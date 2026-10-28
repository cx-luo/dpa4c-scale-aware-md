// One-sided variant of direct_force_kernel: E_i VJP only, reaction scattered with atomics.
template <int Warps>
__global__ void direct_force_onesided_kernel(Arguments a, float* f, float* v) {
  using P = Profile<8, 2, false>;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const long i = long(blockIdx.x) * Warps + warp;
  __shared__ float own[Warps][P::StateWidth];
  if (i < a.node_count)
    for (int k = lane; k < P::StateWidth; k += 32) own[warp][k] = a.moment_gradient[i * P::StateWidth + k];
  __syncthreads();
  if (i >= a.node_count) return;
  float force[3] = {}, virial[9] = {};
  const auto* sources = static_cast<const std::uint32_t*>(a.edge_index);
  const int ti = int(a.atype[i]);
  for (long e = a.destination_row_ptr[i] + lane; e < a.destination_row_ptr[i + 1]; e += 32) {
    const std::uint32_t j = sources[e];
    if (j == 0xffffffffu || (a.edge_mask && !a.edge_mask[e])) continue;
    auto g = load_geometry<true, std::uint32_t>(e, a.rcut, a.eps, a.edge_vec, sources, a.atype);
    const auto loc = locate_table<true>(g.radius, a.table_stride, a.table_max, a.interval_count);
    const auto row = table_row(a.table, loc, a.table_width);
    float2 radial[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) radial[c] = evaluate_table_with_derivative(row, c, loc.coordinate, loc.clamped);
    float gij[3];
    direct_vjp(a, g, own[warp], 1, ti, radial, gij);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      force[c] += gij[c];
      atomicAdd(f + long(j) * 3 + c, -gij[c]);
#pragma unroll
      for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * a.edge_vec[e * 3 + d];
    }
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
    for (int c = 0; c < 3; ++c) atomicAdd(f + i * 3 + c, force[c]);
#pragma unroll
    for (int c = 0; c < 9; ++c) v[i * 9 + c] = virial[c];
  }
}
