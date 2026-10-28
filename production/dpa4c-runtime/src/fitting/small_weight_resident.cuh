// RESIDENT_FITTING experiment. Included inside k3n after its device primitives.
namespace chain_resident_fitting {
__global__ __launch_bounds__(128, 1)
void fitting_weight_resident_kernel(long n, float* x, const long* __restrict__ types, const double* __restrict__ seed,
                             const double* __restrict__ atom_bias, const float* __restrict__ head_bias,
                             const uint4* __restrict__ img, float* __restrict__ scratch,
                             double* __restrict__ energy) {
  // RESIDENT_FITTING: independent node tiles share immutable weights through the device
  // cache, avoiding a 212 KB per-CTA shared copy for a single 16-node task.
  const unsigned char* smraw = reinterpret_cast<const unsigned char*>(img);
  const float* HD = reinterpret_cast<const float*>(smraw + kFragArea) + 3 * kH;
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  Ctx c{smraw, HD, HD + 100, scratch + (long(blockIdx.x) * warps + warp) * kScratchPerWarp,
        0, lane, lane >> 2, lane & 3};
  const long stride = long(gridDim.x) * warps;
  const float hb = head_bias ? *head_bias : 0.f;
  for (long task = long(blockIdx.x) * warps + warp; task * 16 < n; task += stride) {
    const long r0 = task * 16 + c.gr, r1 = r0 + 8;
    float2 xq[5][4];
    float h[6][8];
    float d2[6][8];
    float p[2] = {0.f, 0.f};
    load_x(xq, x, r0, r1, n, c.tq);
    forward<0>(c, h, d2, p, xq);
    forward<1>(c, h, d2, p, xq);
    forward<2, 0>(c, h, d2, p, xq);
    forward<2, 1>(c, h, d2, p, xq);
    float p0 = p[0], p1 = p[1];
    p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
    if (c.tq == 0) {
      if (r0 < n) energy[r0] = double(p0 + hb) + atom_bias[types[r0]];
      if (r1 < n) energy[r1] = double(p1 + hb) + atom_bias[types[r1]];
    }
    backward<2>(c, h, d2);
    backward<1>(c, h, d2);
    backward<0>(c, h, d2);
    const float s0 = r0 < n ? float(seed[r0]) : 0.f;
    const float s1 = r1 < n ? float(seed[r1]) : 0.f;
#pragma unroll
    for (int nb = 0; nb < 5; ++nb)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = nb * 16 + c.tq + 4 * j;
        if (k < kIn) {
          if (r0 < n) x[r0 * kIn + k] = h[nb][j] * s0;
          if (r1 < n) x[r1 * kIn + k] = h[nb][4 + j] * s1;
        }
      }
  }
}
} // namespace chain_resident_fitting
