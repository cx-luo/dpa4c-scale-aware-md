// rotation_1-v2 link gather (optional); include from raw_half_force.cuh when enabling DPA_FORCE_HALF_LINK.

__device__ long find_edge_to_link(const Arguments& a, long node, std::uint32_t target) {
  const long begin = a.destination_row_ptr[node];
  const long live = reinterpret_cast<const int*>(a.edge_mask)[node];
  for (long p = begin; p < begin + live; ++p) {
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, reinterpret_cast<const float4*>(a.edge_vec)[p], xyz);
    if (static_cast<std::uint32_t>(g.source) == target) return p;
  }
  return -1;
}

__global__ void raw_force_half_owner_link_kernel(Arguments a, DpaDirectNeighborView view, float* f, float* v,
                                               float* peer_delta, long* peer_link) {
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
  float virial[9] = {};
  const int ti = int(a.atype[i]);
  for (long position = lane; position < live; position += 32) {
    const long p = begin + position;
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, reinterpret_cast<const float4*>(a.edge_vec)[p], xyz);
    const std::uint32_t j = static_cast<std::uint32_t>(g.source);
    if (j <= static_cast<std::uint32_t>(i)) continue;
    float gij[3], gji[3];
    pair_vjp_streamed(a, g, own[warp], ti, a.moment_gradient + long(j) * P::StateWidth, g.source_type, gij,
                      gji);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float dj = gji[c] - gij[c];
      force[c] += gij[c] - gji[c];
      peer_delta[p * 3 + c] = dj;
    }
    const long q = find_edge_to_link(a, long(j), static_cast<std::uint32_t>(i));
    if (q >= 0) peer_link[q] = p;
#pragma unroll
    for (int c = 0; c < 3; ++c)
#pragma unroll
      for (int d = 0; d < 3; ++d) virial[c * 3 + d] += gji[c] * xyz[d];
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
#pragma unroll
    for (int c = 0; c < 9; ++c) virial[c] += __shfl_down_sync(0xffffffffu, virial[c], offset);
  }
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < 3; ++c) f[i * 3 + c] += force[c];
#pragma unroll
    for (int c = 0; c < 9; ++c) v[i * 9 + c] += virial[c];
  }
}

__global__ void raw_force_half_gather_kernel(Arguments a, float* f, const float* peer_delta, const long* peer_link) {
  constexpr int Warps = 2;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const long j = long(blockIdx.x) * Warps + warp;
  if (j >= a.node_count) return;
  const long begin = a.destination_row_ptr[j];
  const long live = reinterpret_cast<const int*>(a.edge_mask)[j];
  float force[3] = {};
  for (long position = lane; position < live; position += 32) {
    const long q = begin + position;
    const long p = peer_link[q];
    if (p < 0) continue;
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += peer_delta[p * 3 + c];
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
  }
  if (lane == 0)
#pragma unroll
    for (int c = 0; c < 3; ++c) f[j * 3 + c] += force[c];
}

inline void launch_raw_force_half_link_c8(const Arguments& a, const DpaDirectNeighborView& view, cudaStream_t s,
                                          float* f, float* v, float* peer_delta, long* peer_link) {
  constexpr int warps = 2;
  cudaMemsetAsync(f, 0, a.node_count * 3 * sizeof(float), s);
  cudaMemsetAsync(v, 0, a.node_count * 9 * sizeof(float), s);
  raw_force_half_owner_link_kernel<<<int((a.node_count + warps - 1) / warps), warps * 32, 0, s>>>(
      a, view, f, v, peer_delta, peer_link);
  raw_force_half_gather_kernel<<<int((a.node_count + warps - 1) / warps), warps * 32, 0, s>>>(a, f, peer_delta,
                                                                                              peer_link);
}
