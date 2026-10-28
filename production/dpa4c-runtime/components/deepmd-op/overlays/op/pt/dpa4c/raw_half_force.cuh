// Included inside namespace deepmd_dpa4c from raw_kernel.cuh (after streamed_vjp).
#include "raw_half_force_link.cuh"

extern float* g_half_peer_delta;
extern long* g_half_peer_link;

__global__ void raw_force_half_owner_kernel(Arguments a, DpaDirectNeighborView view, float* f) {
  using P = Profile<8, 2, false>;
  constexpr int Warps = 2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const long i = long(blockIdx.x) * Warps + warp;
  __shared__ float own[Warps][P::StateWidth];
  if (i < a.node_count)
    for (int k = lane; k < P::StateWidth; k += 32) own[warp][k] = a.moment_gradient[i * P::StateWidth + k];
  __syncthreads();
  if (i >= a.node_count) return;
  const long begin = a.destination_row_ptr[i];
  const long live = reinterpret_cast<const int*>(a.edge_mask)[i];
  float force[3] = {};
  const int ti = int(a.atype[i]);
  for (long position = lane; position < live; position += 32) {
    const long p = begin + position;
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, reinterpret_cast<const float4*>(a.edge_vec)[p], xyz);
    const std::uint32_t j = static_cast<std::uint32_t>(g.source);
    if (j <= static_cast<std::uint32_t>(i) || j >= static_cast<std::uint32_t>(a.node_count)) continue;
    float gij[3], gji[3];
    pair_vjp_streamed(a, g, own[warp], ti, a.moment_gradient + long(j) * P::StateWidth, g.source_type, gij,
                      gji);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      force[c] += gij[c] - gji[c];
      atomicAdd(f + long(j) * 3 + c, gji[c] - gij[c]);
    }
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
  }
  if (lane == 0)
#pragma unroll
    for (int c = 0; c < 3; ++c) f[i * 3 + c] += force[c];
}

inline void launch_raw_force_half_c8(const Arguments& a, const DpaDirectNeighborView& view, cudaStream_t s,
                                     float* f, float* v, float* scratch_delta, long total_edges) {
  constexpr int warps = 2;
  (void)scratch_delta;
  (void)total_edges;
  if (scratch_delta != nullptr && g_half_peer_link != nullptr) {
    launch_raw_force_half_link_c8(a, view, s, f, v, scratch_delta, g_half_peer_link);
    return;
  }
  cudaMemsetAsync(f, 0, a.node_count * 3 * sizeof(float), s);
  (void)v;
  raw_force_half_owner_kernel<<<int((a.node_count + warps - 1) / warps), warps * 32, 0, s>>>(a, view, f);
}
