// Fixed low-register fitting pipeline for the large production route.
// A persistent block executes all of its forward tasks, reuses the same shared
// allocation for the backward weight image, then executes the matching
// backward tasks. No cross-block synchronization is required because fitting
// rows are independent. Arithmetic and the FP32 checkpoint match the validated path.
#pragma once

namespace lowreg_pipeline {

constexpr int kSharedBytes =
    split_fitting::kForwardSharedBytes > split_fitting::kBackwardSharedBytes
        ? split_fitting::kForwardSharedBytes
        : split_fitting::kBackwardSharedBytes;

// SM80 async global-to-shared copy.  Unlike the original uint4 assignment,
// cp.async does not route each 16-byte word through a thread register.  The
// full fixed-model image is issued as four uniform groups (at 256 threads),
// then made visible to every warp before MMA starts.  This changes only the
// transport mechanism; the shared-memory bytes and all arithmetic are
// identical.
__device__ __forceinline__ void copy_word_async(
    uint4* destination, const uint4* source) {
  const unsigned shared_address =
      static_cast<unsigned>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(source));
}

template <int Bytes>
__device__ __forceinline__ void copy_image_async(
    uint4* destination, const uint4* source) {
  static_assert(Bytes % 16 == 0, "fitting image must be uint4 aligned");
  constexpr int words = Bytes / 16;
  constexpr int words_per_group_per_thread = 8;
  for (int base = threadIdx.x; base < words;
       base += blockDim.x * words_per_group_per_thread) {
#pragma unroll
    for (int lane_word = 0; lane_word < words_per_group_per_thread;
         ++lane_word) {
      const int item = base + lane_word * blockDim.x;
      if (item < words)
        copy_word_async(destination + item, source + item);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
}

__global__ __launch_bounds__(k3n::kMaxWarps * 32, 1)
void block_pipeline_kernel(
    long n, float* x, const long* __restrict__ types,
    const double* __restrict__ seed,
    const double* __restrict__ atom_bias,
    const float* __restrict__ head_bias,
    const uint4* __restrict__ image,
    float* __restrict__ checkpoint,
    double* __restrict__ energy) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto* destination = reinterpret_cast<uint4*>(shared);
  copy_image_async<split_fitting::kForwardSharedBytes>(destination, image);

  const auto* full = reinterpret_cast<const unsigned char*>(image);
  const float* hd =
      reinterpret_cast<const float*>(full + k3n::kFragArea) + 3 * k3n::kH;
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long stride = static_cast<long>(gridDim.x) * warps;
  const long first = static_cast<long>(blockIdx.x) * warps + warp;
  const float head = head_bias ? *head_bias : 0.0f;

  // Forward phase. Each block owns the same strided task set in both phases.
  for (long task = first; task * 16 < n; task += stride) {
    k3n::Ctx context{shared,
                     hd,
                     hd + 100,
                     checkpoint + task * split_fitting::kCheckpointPerTask,
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
      if (row0 < n)
        energy[row0] = double(p0 + head) + atom_bias[types[row0]];
      if (row1 < n)
        energy[row1] = double(p1 + head) + atom_bias[types[row1]];
    }
  }

  // Every warp has completed its forward task list before this block reuses
  // the shared image for backward fragments.
  __syncthreads();
  const auto* backward_source = reinterpret_cast<const uint4*>(
      full + split_fitting::kForwardSharedBytes);
  copy_image_async<split_fitting::kBackwardSharedBytes>(destination,
                                                backward_source);

  // Backward phase over exactly the same block-local task set.
  for (long task = first; task * 16 < n; task += stride) {
    k3n::Ctx context{shared,
                     hd,
                     hd + 100,
                     checkpoint + task * split_fitting::kCheckpointPerTask,
                     k3n::kFwdFrags,
                     lane,
                     lane >> 2,
                     lane & 3};
    const long row0 = task * 16 + context.gr;
    const long row1 = row0 + 8;
    float gradient[6][8];
    split_fitting::backward_from_checkpoint<2>(context, gradient);
    split_fitting::backward_from_checkpoint<1>(context, gradient);
    split_fitting::backward_from_checkpoint<0>(context, gradient);
    const float scale0 = row0 < n ? float(seed[row0]) : 0.0f;
    const float scale1 = row1 < n ? float(seed[row1]) : 0.0f;
#pragma unroll
    for (int block = 0; block < 5; ++block) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = block * 16 + context.tq + 4 * j;
        if (k < k3n::kIn) {
          if (row0 < n)
            x[row0 * k3n::kIn + k] = gradient[block][j] * scale0;
          if (row1 < n)
            x[row1 * k3n::kIn + k] = gradient[block][4 + j] * scale1;
        }
      }
    }
  }
}

struct PipelineCache {
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

  thread_local PipelineCache cache;
  const long tasks = (n + 15) / 16;
  const int device = ws[0].get_device();
  if (!cache.checkpoint.defined() || cache.device != device ||
      cache.tasks < tasks) {
    const c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(stream, device));
    if (cache.checkpoint.defined()) cache.retired.push_back(cache.checkpoint);
    cache.checkpoint = torch::empty(
        {tasks * static_cast<long>(split_fitting::kCheckpointPerTask)},
        ws[0].options());
    cache.tasks = tasks;
    cache.device = device;
  }
  if (!cache.attributes_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        block_pipeline_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    cache.attributes_set = true;
  }

  int warps = 8;
  int block_cap = 80;
  if (const char* value = std::getenv("FITTING_PIPELINE_PIPELINE_WARPS")) {
    const int parsed = std::atoi(value);
    if (parsed >= 1 && parsed <= k3n::kMaxWarps) warps = parsed;
  }
  if (const char* value = std::getenv("FITTING_PIPELINE_PIPELINE_BLOCKS")) {
    const int parsed = std::atoi(value);
    if (parsed >= 1) block_cap = parsed;
  }
  const int blocks = static_cast<int>(
      std::min<long>(block_cap, (tasks + warps - 1) / warps));
  dpa_l2_pin(stream, prepared.img, size_t(k3n::kImageBytes),
             "r501-lowreg-coroutine-img");
  if (deepmd_dpa4c::chain_fit_node::queue_fit(
          {n, x, atype, seed, bias_atom_e.data_ptr<double>(),
           b_head.numel() ? b_head.data_ptr<float>() : nullptr,
           reinterpret_cast<const uint4*>(prepared.img),
           cache.checkpoint.data_ptr<float>(), energy}, stream)) {
    return true;
  }
  block_pipeline_kernel<<<blocks, warps * 32, kSharedBytes, stream>>>(
      n, x, atype, seed, bias_atom_e.data_ptr<double>(),
      b_head.numel() ? b_head.data_ptr<float>() : nullptr,
      reinterpret_cast<const uint4*>(prepared.img),
      cache.checkpoint.data_ptr<float>(), energy);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  static bool announced = false;
  if (!announced) {
    announced = true;
    cudaFuncAttributes attributes{};
    int resident = 0;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, block_pipeline_kernel));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, block_pipeline_kernel, warps * 32, kSharedBytes));
    std::fprintf(stderr,
                 "LOWREG_FITTING active nodes=%ld tasks=%ld warps=%d "
                 "blocks=%d checkpoint_bytes=%zu shared=%d regs=%d local=%zu "
                 "resident=%d exact_fp32\\n",
                 n, tasks, warps, blocks,
                 static_cast<size_t>(tasks) * split_fitting::kCheckpointPerTask *
                     sizeof(float),
                 kSharedBytes, attributes.numRegs, attributes.localSizeBytes,
                 resident);
  }
  return true;
}


void launch_range(const deepmd_dpa4c::chain_fit_node::FitPayload& fit,
                         long begin, long count, cudaStream_t stream,
                         int block_cap) {
  const int warps = 8;
  const long tasks = (count + 15) / 16;
  const int blocks = static_cast<int>(
      std::min<long>(block_cap, (tasks + warps - 1) / warps));
  const long checkpoint_task = begin / 16;
  block_pipeline_kernel<<<blocks, warps * 32, kSharedBytes, stream>>>(
      count, fit.x + begin * k3n::kIn, fit.types + begin,
      fit.seed + begin, fit.atom_bias, fit.head_bias, fit.image,
      fit.scratch + checkpoint_task * split_fitting::kCheckpointPerTask,
      fit.energy + begin);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace lowreg_pipeline

