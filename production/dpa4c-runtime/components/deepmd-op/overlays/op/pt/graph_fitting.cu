// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Fused energy fitting network for graph-lower inference. The operator is
// descriptor-agnostic: any graph-lowered energy model whose fitting is a
// plain MLP over the flat node axis dispatches here.
//   h_0 = act(x @ W_0 + b_0)                      (+ identity residual when
//   h_l = act(h_{l-1} @ W_l + b_l)                  the layer is square)
//   e   = h_{L-1} @ w_head + b_head + bias_atom_e[atype]   (fp64 output)
// The GEMMs run on cuBLAS in pedantic fp32 (TF32 off); each layer's bias,
// activation and residual collapse into one elementwise epilogue kernel.
// The backward (upstream d_e, a unit vector for the energy reduction) chains
//   dh_{L-1} = d_e * w_head^T
//   dpre_l   = dh_l * act'(pre_l + b_l)
//   dh_{l-1} = dpre_l @ W_l^T  (+ dh_l identity residual)
//   d_x      = dpre_0 @ W_0^T
// with the elementwise steps fused likewise.
//
// The saved state is the pre-activation of every layer, which each GEMM writes
// directly. The backward re-derives the activation derivative from it, so the
// epilogue writes only the activation and the layer costs one full-tensor
// store less than a formulation that also materializes the derivative.
//
// All tensors here are node-scale (atoms, not edges); the fusion removes
// kernel launches and aten glue rather than FLOPs. The head bias arrives as
// a device tensor so that symbolic tracing never reads a value host-side.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <array>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph_ops.h"

namespace {

#define FITTING_CHECK_LAUNCH(what)                                        \
  do {                                                                    \
    cudaError_t err = cudaGetLastError();                                 \
    TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err)); \
  } while (0)

cublasHandle_t cublas_handle(cudaStream_t stream) {
  // A cuBLAS handle is device-bound and unsafe to share across threads, so
  // cache one per device in thread-local storage. Pedantic math keeps the fp32
  // potential-energy surface exact (no TF32, no split-K reordering) for MD.
  thread_local std::unordered_map<int, cublasHandle_t> handles;
  thread_local std::unordered_map<int, cudaStream_t> bound_streams;
  int device = 0;
  cudaGetDevice(&device);
  cublasHandle_t& h = handles[device];
  if (!h) {
    cublasCreate(&h);
#if defined(__HGGCCC__)
    // PPU SDK 2.1 maps PEDANTIC to an inaccurate Tensor path in this API.
    // DEFAULT plus PPU_FP32_TENSOR_OVERRIDE=0 preserves full FP32 GEMM.
    // The deployment environment must set that variable before library load.
    TORCH_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS,
                "PPU fitting: failed to select default FP32 math");
#else
    cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
#endif
  }
  // This handle belongs exclusively to this host thread/device. Bind on
  // creation and every real stream transition; preserve same-stream workspace.
  const auto binding = bound_streams.find(device);
  if (binding == bound_streams.end() || binding->second != stream) {
    TORCH_CHECK(cublasSetStream(h, stream) == CUBLAS_STATUS_SUCCESS,
                "fitting: failed to bind BLAS stream");
    bound_streams[device] = stream;
  }
  return h;
}

// Row-major C(m, n) = A(m, k) @ B(k, n) + beta * C.
void gemm_nn(cudaStream_t stream,
             const float* a,
             const float* b,
             float* c,
             int m,
             int n,
             int k,
             float beta = 0.f) {
  const cublasHandle_t handle = cublas_handle(stream);
  const float alpha = 1.f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n,
              a, k, &beta, c, n);
}

// Row-major C(m, n) = A(m, k) @ B(n, k)^T + beta * C.
void gemm_nt(cudaStream_t stream,
             const float* a,
             const float* b,
             float* c,
             int m,
             int n,
             int k,
             float beta = 0.f) {
  const cublasHandle_t handle = cublas_handle(stream);
  const float alpha = 1.f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b, k,
              a, k, &beta, c, n);
}

// Cache a layout copy only; the potential's weight values are unchanged.
// Keep ownership to prevent pointer reuse, and observe mutation versions.
// An inference tensor has no mutation version and uses the original path.
const float* packed_nano_backward_weight(const torch::Tensor& weight,
                                         int layer,
                                         cudaStream_t stream) {
#if defined(__HGGCCC__)
  struct Entry {
    torch::Tensor original;
    torch::Tensor packed;
    uint32_t version = 0;
    cudaStream_t stream = nullptr;
  };
  thread_local std::array<Entry, 3> entries;
  const auto& counter = weight.unsafeGetTensorImpl()->version_counter();
  if (!counter.enabled()) return nullptr;
  const uint32_t version = counter.current_version();
  Entry& entry = entries[layer];
  if (!entry.original.defined() ||
      entry.original.unsafeGetTensorImpl() != weight.unsafeGetTensorImpl() ||
      entry.version != version || entry.stream != stream) {
    const c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(stream, weight.get_device()));
    auto packed = weight.transpose(0, 1).contiguous();
    entry.original = weight;
    entry.packed = std::move(packed);
    entry.version = version;
    entry.stream = stream;
  }
  return entry.packed.data_ptr<float>();
#else
  return nullptr;
#endif
}

// sigma(z) = (1 + tanh(z/2)) / 2. The identity is not the detour it looks
// like: CUDA's tanhf is a branch-free sequence of one MUFU.EX2 and one
// MUFU.RCP, whereas an accurate expf carries a branchy slow path and the
// intrinsic __expf still needs a reciprocal to finish the sigmoid. Measured
// per kernel instance at one million nodes, the identity beats both, and the
// only faster form is fully approximate, worth 0.5% of the fitting forward
// against an approximate division and exponential.
__device__ __forceinline__ float sigmoid(float z) {
  return 0.5f * (1.f + tanhf(0.5f * z));
}

// Activation codes follow
// deepmd.pt_expt.kernels.triton.dpa1.activation.ACT_CODES (0 = tanh, 1 = silu).
// Value and derivative are separate because the forward needs only the former
// and the backward only the latter.
template <int ACT>
__device__ __forceinline__ float act_value(float z) {
  if constexpr (ACT == 0) {
    return tanhf(z);
  }
  return z * sigmoid(z);
}

template <int ACT>
__device__ __forceinline__ float act_derivative(float z) {
  if constexpr (ACT == 0) {
    const float a = tanhf(z);
    return 1.f - a * a;
  }
  const float s = sigmoid(z);
  return s * (1.f + z * (1.f - s));
}

long ceil_div(long a, long b) { return (a + b - 1) / b; }

// The node axis is long, so deriving a channel from a flat element index
// costs a 64-bit division per thread. The elementwise epilogues instead index
// (channel, node) directly: threadIdx.x selects the float4 lane within a row,
// keeping a warp's accesses contiguous, and the remaining dimensions walk the
// node axis. Layer widths are multiples of four (Python gate).
struct ElementwiseLaunch {
  dim3 grid;
  dim3 block;
};

ElementwiseLaunch elementwise_launch(long n_node, int dout) {
  const int lanes = dout / 4;
  const int rows = std::max(1, 256 / lanes);
  return {dim3((unsigned)ceil_div(n_node, rows)), dim3(lanes, rows)};
}

// y = act(pre + b) (+ x residual). The pre-activation stays where the GEMM
// wrote it, so the layer stores one tensor instead of two.
template <int ACT>
__global__ void layer_epilogue_kernel(long n_node,
                                      int dout,
                                      const float* __restrict__ pre,
                                      const float* __restrict__ b,
                                      const float* __restrict__ x,
                                      int residual,
                                      float* __restrict__ y) {
  const long node = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (node >= n_node) {
    return;
  }
  const int c = (int)threadIdx.x * 4;
  const long t = node * dout + c;
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  float4 yy =
      make_float4(act_value<ACT>(p.x + bb.x), act_value<ACT>(p.y + bb.y),
                  act_value<ACT>(p.z + bb.z), act_value<ACT>(p.w + bb.w));
  if (residual) {
    const float4 xx = *reinterpret_cast<const float4*>(x + t);
    yy.x += xx.x;
    yy.y += xx.y;
    yy.z += xx.z;
    yy.w += xx.w;
  }
  *reinterpret_cast<float4*>(y + t) = yy;
}

// Energy head: e[n] = h[n] @ w_head + b_head + bias_atom_e[atype[n]].
__global__ void head_kernel(long n_node,
                            int width,
                            const float* __restrict__ h,
                            const float* __restrict__ w_head,
                            const float* __restrict__ b_head,
                            const double* __restrict__ bias_atom_e,
                            const long* __restrict__ atype,
                            double* __restrict__ e) {
  const long n = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (n >= n_node) {
    return;
  }
  const float* row = h + n * width;
  float acc = 0.f;
  for (int k = 0; k < width; k += 4) {
    const float4 hv = *reinterpret_cast<const float4*>(row + k);
    const float4 wv = *reinterpret_cast<const float4*>(w_head + k);
    acc += hv.x * wv.x + hv.y * wv.y + hv.z * wv.z + hv.w * wv.w;
  }
  e[n] = (double)(acc + (b_head ? b_head[0] : 0.f)) + bias_atom_e[atype[n]];
}

// Construct the head gradient and apply the final hidden activation VJP in one
// pass. A square final layer also preserves the unmodified head gradient for
// its identity branch.
template <int ACT>
__global__ void seed_backward_epilogue_kernel(
    long n_node,
    int dout,
    const double* __restrict__ d_e,
    const float* __restrict__ w_head,
    const float* __restrict__ pre,
    const float* __restrict__ b,
    float* __restrict__ dpre,
    float* __restrict__ residual_out) {
  const long node = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (node >= n_node) {
    return;
  }
  const int c = (int)threadIdx.x * 4;
  const long t = node * dout + c;
  const float de = (float)d_e[node];
  const float4 wv = *reinterpret_cast<const float4*>(w_head + c);
  const float4 dh = make_float4(de * wv.x, de * wv.y, de * wv.z, de * wv.w);
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  if (residual_out) {
    *reinterpret_cast<float4*>(residual_out + t) = dh;
  }
  *reinterpret_cast<float4*>(dpre + t) =
      make_float4(dh.x * act_derivative<ACT>(p.x + bb.x),
                  dh.y * act_derivative<ACT>(p.y + bb.y),
                  dh.z * act_derivative<ACT>(p.z + bb.z),
                  dh.w * act_derivative<ACT>(p.w + bb.w));
}

// Convert dh to dpre in place: dh *= act'(pre + b). A square residual layer
// also preserves the unmodified dh in its output buffer before the GEMM
// accumulates the weighted branch with beta = 1.
template <int ACT>
__global__ void backward_epilogue_kernel(long n_node,
                                         int dout,
                                         const float* __restrict__ dh,
                                         const float* __restrict__ pre,
                                         const float* __restrict__ b,
                                         float* __restrict__ dpre,
                                         float* __restrict__ residual_out) {
  const long node = blockIdx.x * (long)blockDim.y + threadIdx.y;
  if (node >= n_node) {
    return;
  }
  const int c = (int)threadIdx.x * 4;
  const long t = node * dout + c;
  const float4 d = *reinterpret_cast<const float4*>(dh + t);
  const float4 p = *reinterpret_cast<const float4*>(pre + t);
  const float4 bb =
      b ? *reinterpret_cast<const float4*>(b + c) : make_float4(0, 0, 0, 0);
  if (residual_out) {
    *reinterpret_cast<float4*>(residual_out + t) = d;
  }
  *reinterpret_cast<float4*>(dpre + t) =
      make_float4(d.x * act_derivative<ACT>(p.x + bb.x),
                  d.y * act_derivative<ACT>(p.y + bb.y),
                  d.z * act_derivative<ACT>(p.z + bb.z),
                  d.w * act_derivative<ACT>(p.w + bb.w));
}

/// Dispatch a kernel template over the two supported activation codes.
template <typename Fn>
void dispatch_activation(long act, Fn&& launch) {
  if (act == 0) {
    launch(std::integral_constant<int, 0>{});
  } else {
    launch(std::integral_constant<int, 1>{});
  }
}

// DPA4C-Nano fitting is a single 70->96 hidden layer plus a 96-wide head.
// cuBLAS GEMMs of this size are launch-bound on PPU; a per-node fused kernel
// keeps the 70-vector in registers and writes the same pre-activation contract.
constexpr int kNanoIn = 70;
constexpr int kNanoHid = 96;

bool is_nano_fitting(const FittingLayerPlan& plan, long input_width) {
  return plan.n_layer == 1 && input_width == kNanoIn &&
         plan.width_max == kNanoHid;
}

template <int ACT>
__global__ void nano_fitting_forward_kernel(long n_node,
                                            const float* __restrict__ x,
                                            const float* __restrict__ w0,
                                            const float* __restrict__ b0,
                                            const float* __restrict__ w_head,
                                            const float* __restrict__ b_head,
                                            const double* __restrict__ bias_atom_e,
                                            const long* __restrict__ atype,
                                            float* __restrict__ saved,
                                            double* __restrict__ e) {
  const long n = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (n >= n_node) {
    return;
  }
  const float* row = x + n * kNanoIn;
  float* pre_out = saved + n * kNanoHid;
  float acc = 0.f;
#pragma unroll
  for (int c0 = 0; c0 < kNanoHid; c0 += 8) {
    float pre[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      pre[i] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kNanoIn; ++k) {
      const float xk = row[k];
      const float* wrow = w0 + k * kNanoHid + c0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        pre[i] = fmaf(xk, wrow[i], pre[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      pre_out[c0 + i] = pre[i];
      const float z = pre[i] + (b0 ? b0[c0 + i] : 0.f);
      acc = fmaf(act_value<ACT>(z), w_head[c0 + i], acc);
    }
  }
  e[n] = (double)(acc + (b_head ? b_head[0] : 0.f)) + bias_atom_e[atype[n]];
}

template <int ACT>
__global__ void nano_fitting_backward_kernel(long n_node,
                                             const double* __restrict__ d_e,
                                             const float* __restrict__ saved,
                                             const float* __restrict__ w0,
                                             const float* __restrict__ b0,
                                             const float* __restrict__ w_head,
                                             float* __restrict__ d_x) {
  const long n = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (n >= n_node) {
    return;
  }
  const float de = (float)d_e[n];
  const float* pre = saved + n * kNanoHid;
  float dpre[kNanoHid];
#pragma unroll
  for (int c = 0; c < kNanoHid; ++c) {
    const float z = pre[c] + (b0 ? b0[c] : 0.f);
    dpre[c] = de * w_head[c] * act_derivative<ACT>(z);
  }
  float* dx = d_x + n * kNanoIn;
#pragma unroll
  for (int k = 0; k < kNanoIn; ++k) {
    const float* wrow = w0 + k * kNanoHid;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < kNanoHid; c += 4) {
      acc = fmaf(dpre[c], wrow[c], acc);
      acc = fmaf(dpre[c + 1], wrow[c + 1], acc);
      acc = fmaf(dpre[c + 2], wrow[c + 2], acc);
      acc = fmaf(dpre[c + 3], wrow[c + 3], acc);
    }
    dx[k] = acc;
  }
}

template <int ACT>
__global__ void nano_fitting_energy_gradient_kernel(
    long n_node,
    float* __restrict__ x,
    const float* __restrict__ w0,
    const float* __restrict__ b0,
    const float* __restrict__ w_head,
    const float* __restrict__ b_head,
    const double* __restrict__ bias_atom_e,
    const long* __restrict__ atype,
    const double* __restrict__ seed,
    double* __restrict__ e) {
  const long n = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (n >= n_node) {
    return;
  }
  float row[kNanoIn];
#pragma unroll
  for (int k = 0; k < kNanoIn; ++k) {
    row[k] = x[n * kNanoIn + k];
  }
  float dpre[kNanoHid];
  float acc = 0.f;
#pragma unroll
  for (int c0 = 0; c0 < kNanoHid; c0 += 8) {
    float pre[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      pre[i] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kNanoIn; ++k) {
      const float* wrow = w0 + k * kNanoHid + c0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        pre[i] = fmaf(row[k], wrow[i], pre[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float z = pre[i] + (b0 ? b0[c0 + i] : 0.f);
      acc = fmaf(act_value<ACT>(z), w_head[c0 + i], acc);
      dpre[c0 + i] = act_derivative<ACT>(z) * w_head[c0 + i];
    }
  }
  e[n] = (double)(acc + (b_head ? b_head[0] : 0.f)) + bias_atom_e[atype[n]];
  const float de = (float)seed[n];
#pragma unroll
  for (int c = 0; c < kNanoHid; ++c) {
    dpre[c] *= de;
  }
#pragma unroll
  for (int k = 0; k < kNanoIn; ++k) {
    const float* wrow = w0 + k * kNanoHid;
    float dx = 0.f;
#pragma unroll
    for (int c = 0; c < kNanoHid; c += 4) {
      dx = fmaf(dpre[c], wrow[c], dx);
      dx = fmaf(dpre[c + 1], wrow[c + 1], dx);
      dx = fmaf(dpre[c + 2], wrow[c + 2], dx);
      dx = fmaf(dpre[c + 3], wrow[c + 3], dx);
    }
    x[n * kNanoIn + k] = dx;
  }
}

#include "tiled_fitting.cuh"
#include "k3n_fitting.cuh"

bool launch_nano_forward(cudaStream_t stream,
                         const FittingLayerPlan& plan,
                         const float* x,
                         long input_width,
                         const long* atype,
                         const std::vector<torch::Tensor>& ws,
                         const std::vector<torch::Tensor>& bs,
                         const torch::Tensor& w_head,
                         const torch::Tensor& b_head,
                         const torch::Tensor& bias_atom_e,
                         int64_t act,
                         long run_nodes,
                         float* saved,
                         double* e) {
  if (!is_nano_fitting(plan, input_width) || run_nodes <= 0) {
    return false;
  }
  const float* b0 = bs[0].numel() ? bs[0].data_ptr<float>() : nullptr;
  dispatch_activation(act, [&](auto tag) {
    nano_fitting_forward_kernel<decltype(tag)::value>
        <<<ceil_div(run_nodes, 128), 128, 0, stream>>>(
            run_nodes, x, ws[0].data_ptr<float>(), b0, w_head.data_ptr<float>(),
            b_head.numel() ? b_head.data_ptr<float>() : nullptr,
            bias_atom_e.data_ptr<double>(), atype, saved, e);
  });
  FITTING_CHECK_LAUNCH("nano_fitting forward");
  return true;
}

bool launch_nano_backward(cudaStream_t stream,
                          const FittingLayerPlan& plan,
                          const double* d_e,
                          const float* saved,
                          const std::vector<torch::Tensor>& ws,
                          const std::vector<torch::Tensor>& bs,
                          const torch::Tensor& w_head,
                          int64_t act,
                          long run_nodes,
                          float* d_x) {
  if (!is_nano_fitting(plan, ws[0].size(0)) || run_nodes <= 0) {
    return false;
  }
  const float* b0 = bs[0].numel() ? bs[0].data_ptr<float>() : nullptr;
  dispatch_activation(act, [&](auto tag) {
    nano_fitting_backward_kernel<decltype(tag)::value>
        <<<ceil_div(run_nodes, 128), 128, 0, stream>>>(
            run_nodes, d_e, saved, ws[0].data_ptr<float>(), b0,
            w_head.data_ptr<float>(), d_x);
  });
  FITTING_CHECK_LAUNCH("nano_fitting backward");
  return true;
}

bool launch_nano_energy_gradient(cudaStream_t stream,
                                 const FittingLayerPlan& plan,
                                 float* x,
                                 long input_width,
                                 const long* atype,
                                 const std::vector<torch::Tensor>& ws,
                                 const std::vector<torch::Tensor>& bs,
                                 const torch::Tensor& w_head,
                                 const torch::Tensor& b_head,
                                 const torch::Tensor& bias_atom_e,
                                 int64_t act,
                                 const double* seed,
                                 long run_nodes,
                                 double* e) {
  if (!is_nano_fitting(plan, input_width) || run_nodes <= 0) {
    return false;
  }
  const float* b0 = bs[0].numel() ? bs[0].data_ptr<float>() : nullptr;
  dispatch_activation(act, [&](auto tag) {
    nano_fitting_energy_gradient_kernel<decltype(tag)::value>
        <<<ceil_div(run_nodes, 128), 128, 0, stream>>>(
            run_nodes, x, ws[0].data_ptr<float>(), b0, w_head.data_ptr<float>(),
            b_head.numel() ? b_head.data_ptr<float>() : nullptr,
            bias_atom_e.data_ptr<double>(), atype, seed, e);
  });
  FITTING_CHECK_LAUNCH("nano_fitting energy_gradient");
  return true;
}

}  // namespace

namespace {

FittingLayerPlan validate_fitting_forward_inputs(
    const char* operation,
    const torch::Tensor& x,
    const torch::Tensor& atype,
    const std::vector<torch::Tensor>& ws,
    const torch::Tensor& bias_atom_e) {
  TORCH_CHECK(x.dim() == 2 && x.is_cuda() && x.is_contiguous() &&
                  x.scalar_type() == torch::kFloat32,
              operation, ": x must be contiguous CUDA fp32 with shape (N, D)");
  TORCH_CHECK(atype.dim() == 1 && atype.size(0) == x.size(0) &&
                  atype.is_cuda() && atype.is_contiguous() &&
                  atype.device() == x.device() &&
                  atype.scalar_type() == torch::kInt64,
              operation,
              ": atype must be contiguous CUDA int64 with shape (N,) on "
              "the device of x");
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(
      plan.n_layer > 0 && ws[0].dim() == 2 && ws[0].size(0) == x.size(1),
      operation, ": the first fitting weight must match the input width");
  TORCH_CHECK(bias_atom_e.dim() == 1 && bias_atom_e.is_cuda() &&
                  bias_atom_e.is_contiguous() &&
                  bias_atom_e.device() == x.device() &&
                  bias_atom_e.scalar_type() == torch::kFloat64,
              operation,
              ": bias_atom_e must be contiguous CUDA fp64 on the device of x");
  return plan;
}

}  // namespace

// Evaluate the network over one contiguous run of nodes. Every full-width
// tensor is indexed from the run's first node, so the same code serves the
// whole node axis and a single tile of it. ``saved`` and ``activation`` are
// sized for the run, not for the system.
void fitting_forward_range(cudaStream_t stream,
                           const FittingLayerPlan& plan,
                           const float* x,
                           long input_width,
                           const long* atype,
                           const std::vector<torch::Tensor>& ws,
                           const std::vector<torch::Tensor>& bs,
                           const std::vector<int64_t>& resnets,
                           const torch::Tensor& w_head,
                           const torch::Tensor& b_head,
                           const torch::Tensor& bias_atom_e,
                           int64_t act,
                           long run_nodes,
                           float* saved,
                           float* const activation[2],
                           double* e) {
  if (launch_nano_forward(stream, plan, x, input_width, atype, ws, bs, w_head,
                          b_head, bias_atom_e, act, run_nodes, saved, e)) {
    return;
  }
  const float* cur = x;
  int din = (int)input_width;
  for (int l = 0; l < plan.n_layer; ++l) {
    const int dout = (int)ws[l].size(1);
    float* pre = saved + plan.offset[l] * run_nodes;
    float* y = activation[l & 1];
    gemm_nn(stream, cur, ws[l].data_ptr<float>(), pre, (int)run_nodes, dout,
            din);
    const ElementwiseLaunch shape = elementwise_launch(run_nodes, dout);
    const bool residual = resnets[l] && dout == din;
    dispatch_activation(act, [&](auto tag) {
      layer_epilogue_kernel<decltype(tag)::value>
          <<<shape.grid, shape.block, 0, stream>>>(
              run_nodes, dout, pre,
              bs[l].numel() ? bs[l].data_ptr<float>() : nullptr, cur,
              residual ? 1 : 0, y);
    });
    FITTING_CHECK_LAUNCH("graph_fitting layer");
    cur = y;
    din = dout;
  }
  head_kernel<<<ceil_div(run_nodes, 256), 256, 0, stream>>>(
      run_nodes, din, cur, w_head.data_ptr<float>(),
      b_head.numel() ? b_head.data_ptr<float>() : nullptr,
      bias_atom_e.data_ptr<double>(), atype, e);
  FITTING_CHECK_LAUNCH("graph_fitting head");
}

// Propagate the head cotangent of one run of nodes back to the input. ``dh``
// and ``dh_next`` are scratch of the run's size; ``d_x`` is indexed from the
// run's first node.
void fitting_backward_range(cudaStream_t stream,
                            const FittingLayerPlan& plan,
                            const double* d_e,
                            const float* saved,
                            const std::vector<torch::Tensor>& ws,
                            const std::vector<torch::Tensor>& bs,
                            const std::vector<int64_t>& resnets,
                            const torch::Tensor& w_head,
                            int64_t act,
                            long run_nodes,
                            float* dh,
                            float* dh_next,
                            float* d_x) {
  if (launch_nano_backward(stream, plan, d_e, saved, ws, bs, w_head, act,
                           run_nodes, d_x)) {
    return;
  }
  for (int l = plan.n_layer - 1; l >= 0; --l) {
    const int dout = (int)ws[l].size(1);
    const int din = (int)ws[l].size(0);
    const float* pre = saved + plan.offset[l] * run_nodes;
    const float* b = bs[l].numel() ? bs[l].data_ptr<float>() : nullptr;
    float* out = l > 0 ? dh_next : d_x;
    const bool residual = resnets[l] && dout == din;
    const ElementwiseLaunch shape = elementwise_launch(run_nodes, dout);
    dispatch_activation(act, [&](auto tag) {
      constexpr int kAct = decltype(tag)::value;
      if (l == plan.n_layer - 1) {
        seed_backward_epilogue_kernel<kAct>
            <<<shape.grid, shape.block, 0, stream>>>(
                run_nodes, dout, d_e, w_head.data_ptr<float>(), pre, b, dh,
                residual ? out : nullptr);
      } else {
        backward_epilogue_kernel<kAct><<<shape.grid, shape.block, 0, stream>>>(
            run_nodes, dout, dh, pre, b, dh, residual ? out : nullptr);
      }
    });
    FITTING_CHECK_LAUNCH("graph_fitting backward layer");
    const bool nano_shape = plan.n_layer == 3 && plan.width_max == 96 &&
                            ws[0].size(0) == 70 && dout == 96 &&
                            (din == 70 || din == 96);
    const float* packed = nano_shape
        ? packed_nano_backward_weight(ws[l], l, stream) : nullptr;
    if (packed) {
      gemm_nn(stream, dh, packed, out, (int)run_nodes, din, dout,
              residual ? 1.f : 0.f);
    } else {
      gemm_nt(stream, dh, ws[l].data_ptr<float>(), out, (int)run_nodes, din, dout,
              residual ? 1.f : 0.f);
    }
    if (l > 0) {
      std::swap(dh, dh_next);
    }
  }
}


extern "C" __attribute__((weak)) bool dpa_chain_try_fitting_range(cudaStream_t stream,
    const FittingLayerPlan& plan, float* x, long input_width, const long* atype,
    const std::vector<torch::Tensor>& ws, const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets, const torch::Tensor& w_head,
    const torch::Tensor& b_head, const torch::Tensor& bias_atom_e, int64_t act,
    const double* seed, long n, double* e);
bool fitting_tiled_energy_gradient_range(cudaStream_t stream,
    const FittingLayerPlan& plan, float* x, long input_width, const long* atype,
    const std::vector<torch::Tensor>& ws, const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets, const torch::Tensor& w_head,
    const torch::Tensor& b_head, const torch::Tensor& bias_atom_e, int64_t act,
    const double* seed, long n, double* e) {
  if (dpa_chain_try_fitting_range && dpa_chain_try_fitting_range(stream,plan,x,input_width,atype,ws,bs,resnets,w_head,b_head,bias_atom_e,act,seed,n,e)) return true;
  if (k3n::energy_gradient(stream, x, input_width, atype, ws, bs, resnets,
                           w_head, b_head, bias_atom_e, act, seed, n, e))
    return true;
  const char* setting = std::getenv("DPA_FITTING_TILE_ROWS");
  if (!setting || !setting[0] || setting[0] == '0') return false;
  if(n < 16384) return false;
  if (n <= 0 || input_width != 70 || plan.n_layer != 3 ||
      ws[0].size(1) != 96 || ws[1].size(0) != 96 || ws[1].size(1) != 96 ||
      ws[2].size(0) != 96 || ws[2].size(1) != 96 || resnets.size() != 3)
    return false;
  const int block_rows = std::atoi(setting);
  dispatch_activation(act, [&](auto tag) {
    constexpr int A = decltype(tag)::value;
    if (block_rows == 16)
      launch_tiled_nano<16, 8, A>(stream,n,x,atype,seed,ws,bs,resnets,w_head,b_head,bias_atom_e,e);
    else if (block_rows == 64)
      launch_tiled_nano<64, 8, A>(stream,n,x,atype,seed,ws,bs,resnets,w_head,b_head,bias_atom_e,e);
    else
      launch_tiled_nano<32, 8, A>(stream,n,x,atype,seed,ws,bs,resnets,w_head,b_head,bias_atom_e,e);
  });
  FITTING_CHECK_LAUNCH("tiled nano energy and gradient");
  return true;
}

// Forward: per-atom energy (fp64 (N, 1)) plus the flat saved buffer of the
// layer pre-activations -- chunk l a contiguous (N, width_l) sheet. The
// activations themselves stay in a forward-only ping-pong.
std::tuple<torch::Tensor, torch::Tensor> graph_fitting(
    torch::Tensor x,
    torch::Tensor atype,
    std::vector<torch::Tensor> ws,
    std::vector<torch::Tensor> bs,
    std::vector<int64_t> resnets,
    torch::Tensor w_head,
    torch::Tensor b_head,
    torch::Tensor bias_atom_e,
    int64_t act) {
  const FittingLayerPlan plan = validate_fitting_forward_inputs(
      "graph_fitting", x, atype, ws, bias_atom_e);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const long n_node = x.size(0);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto saved = torch::empty({n_node * plan.saved_width()}, f32);
  auto e = torch::empty({n_node, 1}, x.options().dtype(torch::kFloat64));
  if (n_node == 0) {
    return {e, saved};
  }
  // Two-slot ping-pong for the activations: layer l writes slot ``l & 1``
  // while reading the previous layer's slot, so an activation is overwritten
  // only after the next GEMM has consumed it (kernels run in stream order). A
  // single-layer network requires only its output slot.
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto act_buf = torch::empty({slots, n_node, plan.width_max}, f32);
  float* activation[2] = {
      act_buf[0].data_ptr<float>(),
      slots > 1 ? act_buf[1].data_ptr<float>() : act_buf[0].data_ptr<float>()};
  fitting_forward_range(
      at::cuda::getCurrentCUDAStream(), plan, x.data_ptr<float>(), x.size(1),
      atype.data_ptr<long>(), ws, bs, resnets, w_head, b_head, bias_atom_e, act,
      n_node, saved.data_ptr<float>(), activation, e.data_ptr<double>());
  return {e, saved};
}

// Backward: d_x from the upstream d_e (fp64 (N, 1)). The saved pre-activation
// extent and fitting widths determine the output shape, so the descriptor is
// not retained solely for shape metadata.
void graph_fitting_backward_core(torch::Tensor d_e,
                                 torch::Tensor saved,
                                 std::vector<torch::Tensor> ws,
                                 std::vector<torch::Tensor> bs,
                                 std::vector<int64_t> resnets,
                                 torch::Tensor w_head,
                                 int64_t act,
                                 torch::Tensor d_x) {
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.saved_width() > 0 && saved.numel() % plan.saved_width() == 0,
              "graph_fitting_backward: saved buffer does not match the "
              "fitting widths");
  const long n_node = saved.numel() / plan.saved_width();
  const long input_width = ws[0].size(0);
  TORCH_CHECK(d_x.dim() == 2 && d_x.size(0) == n_node &&
                  d_x.size(1) == input_width &&
                  d_x.scalar_type() == torch::kFloat32 && d_x.is_cuda() &&
                  d_x.is_contiguous(),
              "graph_fitting_backward: output must be contiguous CUDA "
              "fp32 with shape (N, input_width)");
  // Guard the empty system before the division by ``n_node`` below.
  if (n_node == 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(saved.device());
  auto f32 = saved.options().dtype(torch::kFloat32);
  auto d_e_c = d_e.contiguous();
  auto dh = torch::empty({n_node, plan.width_max}, f32);
  auto dh_next = plan.n_layer > 1 ? torch::empty({n_node, plan.width_max}, f32)
                                  : torch::empty({0}, f32);
  fitting_backward_range(at::cuda::getCurrentCUDAStream(), plan,
                         d_e_c.data_ptr<double>(), saved.data_ptr<float>(), ws,
                         bs, resnets, w_head, act, n_node, dh.data_ptr<float>(),
                         plan.n_layer > 1 ? dh_next.data_ptr<float>() : nullptr,
                         d_x.data_ptr<float>());
}

torch::Tensor graph_fitting_backward(torch::Tensor d_e,
                                     torch::Tensor saved,
                                     std::vector<torch::Tensor> ws,
                                     std::vector<torch::Tensor> bs,
                                     std::vector<int64_t> resnets,
                                     torch::Tensor w_head,
                                     int64_t act) {
  const FittingLayerPlan plan = fitting_layer_plan(ws);
  TORCH_CHECK(plan.saved_width() > 0 && saved.numel() % plan.saved_width() == 0,
              "graph_fitting_backward: saved buffer does not match the "
              "fitting widths");
  const long n_node = saved.numel() / plan.saved_width();
  auto d_x = torch::empty({n_node, ws[0].size(0)}, saved.options());
  graph_fitting_backward_core(d_e, saved, std::move(ws), std::move(bs),
                              std::move(resnets), w_head, act, d_x);
  return d_x;
}

// Energy and input gradient in one pass over node tiles.
//
// Inference knows the head cotangent before the forward runs -- it is the
// ownership mask -- so the two directions need not be separated by the whole
// node axis. Walking tiles instead retires each tile's pre-activations as soon
// as its backward consumes them, which replaces the largest node-scale
// allocation of a step with one of tile size. The network is evaluated per
// node, so no tile needs anything from another and nothing is recomputed.
//
// ``tile`` of zero, or any value covering the system, degenerates to a single
// run and reproduces the untiled sequence exactly.
//
// The cotangent replaces the descriptor in place. A tile reads its descriptor
// rows in the first layer and writes their cotangent in the last, and no later
// tile revisits them, so the two never need separate node-scale allocations;
// inference has no further use for the descriptor once the forward has read
// it. Only the energy is returned.
torch::Tensor graph_fitting_energy_gradient(torch::Tensor x,
                                            torch::Tensor atype,
                                            std::vector<torch::Tensor> ws,
                                            std::vector<torch::Tensor> bs,
                                            std::vector<int64_t> resnets,
                                            torch::Tensor w_head,
                                            torch::Tensor b_head,
                                            torch::Tensor bias_atom_e,
                                            int64_t act,
                                            torch::Tensor seed,
                                            int64_t tile) {
  const FittingLayerPlan plan = validate_fitting_forward_inputs(
      "graph_fitting_energy_gradient", x, atype, ws, bias_atom_e);
  const c10::cuda::CUDAGuard device_guard(x.device());
  const long n_node = x.size(0);
  const long input_width = x.size(1);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto e = torch::empty({n_node, 1}, x.options().dtype(torch::kFloat64));
  if (n_node == 0) {
    return e;
  }
  auto seed_c = seed.contiguous();
  TORCH_CHECK(
      seed_c.numel() == n_node && seed_c.scalar_type() == torch::kFloat64 &&
          seed_c.is_cuda() && seed_c.device() == x.device(),
      "graph_fitting_energy_gradient: seed must be CUDA fp64 with one entry "
      "per node on the device of x");

  const long run = tile > 0 ? std::min<long>(tile, n_node) : n_node;
  const int slots = plan.n_layer > 1 ? 2 : 1;
  auto saved = torch::empty({run * plan.saved_width()}, f32);
  // The backward reads only the saved pre-activations, never the activations,
  // so the forward ping-pong and the two cotangent buffers never hold live
  // data at the same time and share one allocation.
  auto scratch = torch::empty({slots, run, plan.width_max}, f32);
  float* slot[2] = {
      scratch[0].data_ptr<float>(),
      slots > 1 ? scratch[1].data_ptr<float>() : scratch[0].data_ptr<float>()};

  auto stream = at::cuda::getCurrentCUDAStream();
  if (run == n_node &&
      launch_nano_energy_gradient(
          stream, plan, x.data_ptr<float>(), input_width,
          atype.data_ptr<long>(), ws, bs, w_head, b_head, bias_atom_e, act,
          seed_c.data_ptr<double>(), n_node, e.data_ptr<double>())) {
    return e;
  }
  for (long begin = 0; begin < n_node; begin += run) {
    const long count = std::min(run, n_node - begin);
    if (fitting_tiled_energy_gradient_range(stream, plan,
        x.data_ptr<float>() + begin * input_width, input_width,
        atype.data_ptr<long>() + begin, ws, bs, resnets, w_head, b_head,
        bias_atom_e, act, seed_c.data_ptr<double>() + begin, count,
        e.data_ptr<double>() + begin)) continue;
    fitting_forward_range(
        stream, plan, x.data_ptr<float>() + begin * input_width, input_width,
        atype.data_ptr<long>() + begin, ws, bs, resnets, w_head, b_head,
        bias_atom_e, act, count, saved.data_ptr<float>(), slot,
        e.data_ptr<double>() + begin);
    fitting_backward_range(stream, plan, seed_c.data_ptr<double>() + begin,
                           saved.data_ptr<float>(), ws, bs, resnets, w_head,
                           act, count, slot[0],
                           plan.n_layer > 1 ? slot[1] : nullptr,
                           x.data_ptr<float>() + begin * input_width);
  }
  return e;
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.impl("graph_fitting", torch::kCUDA, &graph_fitting);
  m.impl("graph_fitting_backward", torch::kCUDA, &graph_fitting_backward);
  m.impl("graph_fitting_energy_gradient", torch::kCUDA,
         &graph_fitting_energy_gradient);
}
