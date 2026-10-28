// SPDX-License-Identifier: LGPL-3.0-or-later
// FP32 multi-row MLP: six matrix products, activation/residual VJPs and head.
// Inputs and derivatives stay in block-local memory between products.
template<int BM, int BK, int ACT>
__global__ void tiled_nano_fitting_kernel(
    long n, float* x, const long* types, const double* seed,
    const float* w0, const float* w1, const float* w2,
    const float* b0, const float* b1, const float* b2,
    const float* head, const float* head_bias, const double* atom_bias,
    int residual1, int residual2, double* energy) {
  constexpr int N = 96;
  constexpr int RM = BM / 8;
  extern __shared__ float memory[];
  float* a = memory;
  float* b = a + BM * N;
  float* derivative = b + BM * N;
  float* panel = derivative + 3 * BM * N;
  const int lane = threadIdx.x & 31;
  const int row_group = threadIdx.x >> 5;
  const long base = static_cast<long>(blockIdx.x) * BM;
  for (int t = threadIdx.x; t < BM * N; t += 256) {
    const int r = t / N, c = t % N;
    a[t] = base + r < n && c < 70 ? x[(base + r) * 70 + c] : 0.f;
  }
  __syncthreads();
  for (int layer = 0; layer < 3; ++layer) {
    const int kdim = layer == 0 ? 70 : N;
    const float* weight = layer == 0 ? w0 : (layer == 1 ? w1 : w2);
    const float* bias = layer == 0 ? b0 : (layer == 1 ? b1 : b2);
    const bool residual = layer == 1 ? residual1 : (layer == 2 ? residual2 : false);
    float acc[RM][3] = {};
    for (int k0 = 0; k0 < kdim; k0 += BK) {
      for (int t = threadIdx.x; t < BK * N; t += 256) {
        const int k = t / N, c = t % N;
        panel[t] = k0 + k < kdim ? weight[(k0 + k) * N + c] : 0.f;
      }
      __syncthreads();
      #pragma unroll
      for (int k = 0; k < BK; ++k) {
        if (k0 + k < kdim) {
          const float bv0 = panel[k * N + lane];
          const float bv1 = panel[k * N + lane + 32];
          const float bv2 = panel[k * N + lane + 64];
          #pragma unroll
          for (int r = 0; r < RM; ++r) {
            const float av = a[(row_group + r * 8) * N + k0 + k];
            acc[r][0] = fmaf(av, bv0, acc[r][0]);
            acc[r][1] = fmaf(av, bv1, acc[r][1]);
            acc[r][2] = fmaf(av, bv2, acc[r][2]);
          }
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < RM; ++r) {
      #pragma unroll
      for (int c = 0; c < 3; ++c) {
        const int col = lane + 32 * c;
        const int t = (row_group + 8 * r) * N + col;
        const float z = acc[r][c] + (bias ? bias[col] : 0.f);
        b[t] = act_value<ACT>(z) + (residual ? a[t] : 0.f);
        derivative[layer * BM * N + t] = act_derivative<ACT>(z);
      }
    }
    __syncthreads();
    float* swap = a; a = b; b = swap;
  }
  // Match the baseline head's four-term grouping and FP64 atomic energy bias.
  if (threadIdx.x < BM && base + threadIdx.x < n) {
    float sum = 0.f;
    const float* row = a + threadIdx.x * N;
    for (int k = 0; k < N; k += 4)
      sum += row[k] * head[k] + row[k+1] * head[k+1]
           + row[k+2] * head[k+2] + row[k+3] * head[k+3];
    energy[base + threadIdx.x] = static_cast<double>(sum + (head_bias ? head_bias[0] : 0.f))
                             + atom_bias[types[base + threadIdx.x]];
  }
  __syncthreads();
  for (int t = threadIdx.x; t < BM * N; t += 256) {
    const int r = t / N, c = t % N;
    a[t] = base + r < n ? static_cast<float>(seed[base + r]) * head[c] : 0.f;
  }
  __syncthreads();
  for (int layer = 2; layer >= 0; --layer) {
    const int outdim = layer == 0 ? 70 : N;
    const float* weight = layer == 0 ? w0 : (layer == 1 ? w1 : w2);
    const bool residual = layer == 1 ? residual1 : (layer == 2 ? residual2 : false);
    float* saved = derivative + layer * BM * N;
    for (int t = threadIdx.x; t < BM * N; t += 256) {
      const float upstream = a[t];
      a[t] = upstream * saved[t];
      saved[t] = residual ? upstream : 0.f;
    }
    __syncthreads();
    float acc[RM][3] = {};
    for (int k0 = 0; k0 < N; k0 += BK) {
      // Read a transposed panel with contiguous accesses in the original W.
      for (int t = threadIdx.x; t < BK * N; t += 256) {
        const int col = t / BK, k = t % BK;
        panel[k * N + col] = col < outdim && k0 + k < N
                          ? weight[col * N + k0 + k] : 0.f;
      }
      __syncthreads();
      #pragma unroll
      for (int k = 0; k < BK; ++k) {
        const float bv0 = panel[k * N + lane];
        const float bv1 = panel[k * N + lane + 32];
        const float bv2 = panel[k * N + lane + 64];
        #pragma unroll
        for (int r = 0; r < RM; ++r) {
          const float av = a[(row_group + r * 8) * N + k0 + k];
          acc[r][0] = fmaf(av, bv0, acc[r][0]);
          acc[r][1] = fmaf(av, bv1, acc[r][1]);
          acc[r][2] = fmaf(av, bv2, acc[r][2]);
        }
      }
      __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < RM; ++r) {
      #pragma unroll
      for (int c = 0; c < 3; ++c) {
        const int row = row_group + 8 * r, col = lane + 32 * c;
        const int t = row * N + col;
        const float value = acc[r][c] + saved[t];
        b[t] = value;
        if (layer == 0 && col < 70 && base + row < n)
          x[(base + row) * 70 + col] = value;
      }
    }
    __syncthreads();
    float* swap = a; a = b; b = swap;
  }
}

template<int BM, int BK, int ACT>
void launch_tiled_nano(cudaStream_t stream, long n, float* x,
    const long* types, const double* seed,
    const std::vector<torch::Tensor>& ws, const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets, const torch::Tensor& head,
    const torch::Tensor& head_bias, const torch::Tensor& atom_bias, double* e) {
  constexpr int shared = (5 * BM * 96 + BK * 96) * sizeof(float);
  if constexpr (shared > 48 * 1024) {
    TORCH_CHECK(cudaFuncSetAttribute(tiled_nano_fitting_kernel<BM, BK, ACT>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared) == cudaSuccess,
      "tiled fitting dynamic shared memory is unavailable");
  }
  tiled_nano_fitting_kernel<BM, BK, ACT><<<ceil_div(n, BM), 256, shared, stream>>>(
    n, x, types, seed, ws[0].data_ptr<float>(), ws[1].data_ptr<float>(), ws[2].data_ptr<float>(),
    bs[0].numel() ? bs[0].data_ptr<float>() : nullptr,
    bs[1].numel() ? bs[1].data_ptr<float>() : nullptr,
    bs[2].numel() ? bs[2].data_ptr<float>() : nullptr,
    head.data_ptr<float>(), head_bias.numel() ? head_bias.data_ptr<float>() : nullptr,
    atom_bias.data_ptr<double>(), resnets[1] != 0, resnets[2] != 0, e);
}
