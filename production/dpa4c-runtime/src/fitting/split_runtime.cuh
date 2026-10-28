// Exact SPLIT_FITTING fitting split: forward and input-gradient kernels share the same
// three-product/period-2 math as K3N. The only new intermediate is the three
// FP32 SiLU derivative matrices, indexed by the original 16-node task.
#pragma once

namespace split_fitting {

constexpr int kForwardSharedBytes =
    k3n::kFwdFrags * 2 * k3n::kFragBytes;
constexpr int kBackwardSharedBytes =
    k3n::kBwdFrags * 2 * k3n::kFragBytes;
constexpr int kCheckpointPerTask = 3 * 48 * 32;

template <int Layer, bool PhysicalGradientOnly = false>
__device__ __forceinline__ void backward_from_checkpoint(
    const k3n::Ctx& context, float (&gradient)[6][8]) {
  static_assert(!PhysicalGradientOnly || Layer == 0,
                "physical fitting-gradient pruning is layer-0 only");
  constexpr int image_columns = Layer == 0 ? 5 : 6;
  // Descriptor features 62..69 are fixed type embeddings. The coordinate
  // VJP consumes exactly 0..61, so the private fused large path can omit the
  // fifth 16-column layer-0 output block without changing energy or forces.
  // Preserve the five-block layer-0 register/scheduling shape. Only the
  // live MMA/load tail is shortened; this avoids the PPU compiler regression
  // seen when the template extent itself changed from five to four.
  constexpr int columns = image_columns;
  constexpr int live_columns = PhysicalGradientOnly ? 4 : columns;
  const float scale = k3n::kScaleT * context.HD[97 + Layer];
  const float inverse = 1.0f / scale;
  k3n::FragC accumulator[6];
  float partial[6][8];
#pragma unroll
  for (int block = 0; block < columns; ++block) {
    nvcuda::wmma::fill_fragment(accumulator[block], 0.0f);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      partial[block][element] = 0.0f;
    }
  }
#pragma unroll
  for (int kblock = 0; kblock < 6; ++kblock) {
    const float4 lower = *k3n::sd4(context, Layer, kblock, 0);
    const float4 upper = *k3n::sd4(context, Layer, kblock, 1);
    float operand[8];
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const float derivative =
          k3n::f4get(element < 4 ? lower : upper, element & 3);
      if constexpr (Layer == 2) {
        operand[element] = derivative;
      } else {
        operand[element] = gradient[kblock][element] * derivative;
      }
    }
    k3n::pin8(operand);
    k3n::restart<columns>(kblock, accumulator);
    k3n::FragA high_a, low_a;
    k3n::split_frag(operand, high_a, low_a);
#pragma unroll
    for (int group = 0; group < (columns + 2) / 3; ++group) {
      k3n::FragB high_b[3], low_b[3];
#pragma unroll
      for (int item = 0; item < 3; ++item) {
        if (group * 3 + item < live_columns) {
          k3n::load_fragb(
              high_b[item], low_b[item], context.img,
              k3n::kFwdFrags + k3n::bwd_base(Layer) +
                  kblock * image_columns + group * 3 + item - context.frag0,
              context.lane);
        }
      }
      if (group * 3 + 3 <= live_columns) {
        k3n::mma3<3>(accumulator + group * 3, high_a, low_a, high_b,
                     low_b);
      } else if (group * 3 < live_columns) {
        k3n::mma3<live_columns % 3>(accumulator + group * 3, high_a, low_a,
                                    high_b, low_b);
      }
    }
    k3n::retire<columns>(kblock, 6, accumulator, partial);
  }
#pragma unroll
  for (int block = 0; block < columns; ++block) {
    float4 head = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (Layer == 2) {
      head = k3n::epi4(context, 3 * k3n::kH, block);
    }
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      if constexpr (Layer == 0) {
        gradient[block][element] = partial[block][element] * inverse;
      } else if constexpr (Layer == 2) {
        gradient[block][element] =
            fmaf(partial[block][element], inverse,
                 k3n::f4get(head, element & 3));
      } else {
        gradient[block][element] =
            fmaf(partial[block][element], inverse,
                 gradient[block][element]);
      }
    }
  }
}

__global__ __launch_bounds__(k3n::kMaxWarps * 32, 1)
void split_forward_kernel(long n, const float* x,
                          const long* __restrict__ types,
                          const double* __restrict__ atom_bias,
                          const float* __restrict__ head_bias,
                          const uint4* __restrict__ image,
                          float* __restrict__ checkpoint,
                          double* __restrict__ energy) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto* destination = reinterpret_cast<uint4*>(shared);
  for (int i = threadIdx.x; i < kForwardSharedBytes / 16;
       i += blockDim.x) {
    destination[i] = __ldg(image + i);
  }
  __syncthreads();
  const auto* full = reinterpret_cast<const unsigned char*>(image);
  const float* hd =
      reinterpret_cast<const float*>(full + k3n::kFragArea) + 3 * k3n::kH;
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long stride = static_cast<long>(gridDim.x) * warps;
  const float head = head_bias ? *head_bias : 0.0f;
  for (long task = static_cast<long>(blockIdx.x) * warps + warp;
       task * 16 < n; task += stride) {
    k3n::Ctx context{shared,
                     hd,
                     hd + 100,
                     checkpoint + task * kCheckpointPerTask,
                     0,
                     lane,
                     lane >> 2,
                     lane & 3};
    const long row0 = task * 16 + context.gr;
    const long row1 = row0 + 8;
    float2 xq[5][4];
    float hidden[6][8];
    float derivative2[6][8];
    float partial[2] = {0.0f, 0.0f};
    k3n::load_x(xq, x, row0, row1, n, context.tq);
    k3n::forward<0>(context, hidden, derivative2, partial, xq);
    k3n::forward<1>(context, hidden, derivative2, partial, xq);
    k3n::forward<2, 0>(context, hidden, derivative2, partial, xq);
    k3n::forward<2, 1>(context, hidden, derivative2, partial, xq);
#pragma unroll
    for (int block = 0; block < 6; ++block) {
      *k3n::sd4(context, 2, block, 0) =
          make_float4(derivative2[block][0], derivative2[block][1],
                      derivative2[block][2], derivative2[block][3]);
      *k3n::sd4(context, 2, block, 1) =
          make_float4(derivative2[block][4], derivative2[block][5],
                      derivative2[block][6], derivative2[block][7]);
    }
    float p0 = partial[0];
    float p1 = partial[1];
    p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
    if (context.tq == 0) {
      if (row0 < n) {
        energy[row0] = double(p0 + head) + atom_bias[types[row0]];
      }
      if (row1 < n) {
        energy[row1] = double(p1 + head) + atom_bias[types[row1]];
      }
    }
  }
}

__global__ __launch_bounds__(k3n::kMaxWarps * 32, 1)
void split_backward_kernel(long n, float* x,
                           const double* __restrict__ seed,
                           const uint4* __restrict__ image,
                           float* __restrict__ checkpoint) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto* destination = reinterpret_cast<uint4*>(shared);
  const auto* source = reinterpret_cast<const uint4*>(
      reinterpret_cast<const unsigned char*>(image) + kForwardSharedBytes);
  for (int i = threadIdx.x; i < kBackwardSharedBytes / 16;
       i += blockDim.x) {
    destination[i] = __ldg(source + i);
  }
  __syncthreads();
  const auto* full = reinterpret_cast<const unsigned char*>(image);
  const float* hd =
      reinterpret_cast<const float*>(full + k3n::kFragArea) + 3 * k3n::kH;
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long stride = static_cast<long>(gridDim.x) * warps;
  for (long task = static_cast<long>(blockIdx.x) * warps + warp;
       task * 16 < n; task += stride) {
    k3n::Ctx context{shared,
                     hd,
                     hd + 100,
                     checkpoint + task * kCheckpointPerTask,
                     k3n::kFwdFrags,
                     lane,
                     lane >> 2,
                     lane & 3};
    const long row0 = task * 16 + context.gr;
    const long row1 = row0 + 8;
    float gradient[6][8];
    backward_from_checkpoint<2>(context, gradient);
    backward_from_checkpoint<1>(context, gradient);
    backward_from_checkpoint<0>(context, gradient);
    const float scale0 = row0 < n ? float(seed[row0]) : 0.0f;
    const float scale1 = row1 < n ? float(seed[row1]) : 0.0f;
#pragma unroll
    for (int block = 0; block < 5; ++block) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = block * 16 + context.tq + 4 * j;
        if (k < k3n::kIn) {
          if (row0 < n) {
            x[row0 * k3n::kIn + k] = gradient[block][j] * scale0;
          }
          if (row1 < n) {
            x[row1 * k3n::kIn + k] = gradient[block][4 + j] * scale1;
          }
        }
      }
    }
  }
}

struct SplitCache {
  torch::Tensor checkpoint;
  std::vector<torch::Tensor> retired;
  long tasks = 0;
  int device = -1;
  bool attributes_set = false;
};

inline bool energy_gradient(
    cudaStream_t stream, float* x, long input_width, const long* atype,
    const std::vector<torch::Tensor>& ws,
    const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets, const torch::Tensor& w_head,
    const torch::Tensor& b_head, const torch::Tensor& bias_atom_e, int64_t act,
    const double* seed, long n, double* energy) {
  if (n <= 0 || input_width != k3n::kIn) return false;
  const k3n::FitImage prepared = k3n::prepare_image(
      stream, ws, bs, resnets, w_head, b_head, bias_atom_e, act);
  if (!prepared.img) return false;

  thread_local SplitCache cache;
  const long tasks = (n + 15) / 16;
  const int device = ws[0].get_device();
  if (!cache.checkpoint.defined() || cache.device != device ||
      cache.tasks < tasks) {
    const c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(stream, device));
    if (cache.checkpoint.defined()) {
      cache.retired.push_back(cache.checkpoint);
    }
    cache.checkpoint = torch::empty(
        {tasks * static_cast<long>(kCheckpointPerTask)}, ws[0].options());
    cache.tasks = tasks;
    cache.device = device;
  }
  if (!cache.attributes_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        split_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kForwardSharedBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        split_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kBackwardSharedBytes));
    cache.attributes_set = true;
  }

  int warps = 8;
  int block_cap = 80;
  if (const char* value = std::getenv("SPLIT_FITTING_SPLIT_WARPS")) {
    const int parsed = std::atoi(value);
    if (parsed >= 1 && parsed <= k3n::kMaxWarps) warps = parsed;
  }
  if (const char* value = std::getenv("SPLIT_FITTING_SPLIT_BLOCKS")) {
    const int parsed = std::atoi(value);
    if (parsed >= 1) block_cap = parsed;
  }
  const int blocks =
      static_cast<int>(std::min<long>(block_cap, (tasks + warps - 1) / warps));
  dpa_l2_pin(stream, prepared.img, size_t(k3n::kImageBytes),
             "split_fitting-split-img");
  auto* checkpoint = cache.checkpoint.data_ptr<float>();
  const auto* image = reinterpret_cast<const uint4*>(prepared.img);
  const char* wide = std::getenv("DPA_WIDE_FIT_NODE_FIT_NODE_WIDE");
  if (wide && wide[0] == '1' &&
      deepmd_dpa4c::chain_fit_node::queue_fit(
          {n, x, atype, seed, bias_atom_e.data_ptr<double>(),
           b_head.numel() ? b_head.data_ptr<float>() : nullptr, image,
           checkpoint, energy}, stream)) {
    static bool announced_wide = false;
    if (!announced_wide) {
      std::fprintf(stderr, "WIDE_FIT_NODE_WIDE_FIT_NODE queued nodes=%ld tasks=%ld\n", n, tasks);
      announced_wide = true;
    }
    return true;
  }
  split_forward_kernel<<<blocks, warps * 32, kForwardSharedBytes, stream>>>(
      n, x, atype, bias_atom_e.data_ptr<double>(),
      b_head.numel() ? b_head.data_ptr<float>() : nullptr, image, checkpoint,
      energy);
  split_backward_kernel<<<blocks, warps * 32, kBackwardSharedBytes, stream>>>(
      n, x, seed, image, checkpoint);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  static bool announced = false;
  if (!announced) {
    std::fprintf(stderr,
                 "SPLIT_FITTING_SPLIT active nodes=%ld warps=%d blocks=%d "
                 "checkpoint_bytes=%zu exact_fp32\n",
                 n, warps, blocks,
                 static_cast<size_t>(tasks) * kCheckpointPerTask *
                     sizeof(float));
    announced = true;
  }
  return true;
}

}  // namespace split_fitting
