// One-sided force: each directed edge (i<-j) evaluates only the VJP of E_i.
// The reaction on j is scattered with atomics, so q_j is never read.
// Per-node virial is charged to the edge owner; the frame virial is unchanged.

__device__ __forceinline__ void pair_vjp_one(const Arguments& a, EdgeGeometry g, const float* qi, int ti, int tj,
                                             float (&gij)[3]) {
  using P = Profile<8, 2, false>;
  float basis[9];
  fill_angular_basis(g, basis);
  float dbi[9] = {}, direction[3] = {};
  float rgi = 0.f, egi = 0.f;
  const auto loc = locate_table<true>(g.radius, a.table_stride, a.table_max, a.interval_count);
  const auto row = table_row(a.table, loc, a.table_width);
  const auto* fi = reinterpret_cast<const float2*>(a.pair_film) + (long(ti) * a.type_count + tj) * 8;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    float2 radial = (a.radial_modes & 32) ? make_float2(0.5f, 0.1f) : evaluate_table_with_derivative(row, c, loc.coordinate, loc.clamped);
    float2 wi = __ldg(fi + c);
    float vi = fmaf(wi.x, radial.x, wi.y);
    float ai = 0.f;
    if (c < 4) {
#pragma unroll
      for (int d = 0; d < 3; ++d) ai = fmaf(qi[P::VectorOffset + d * 4 + c], basis[1 + d], ai);
#pragma unroll
      for (int d = 0; d < 5; ++d) ai = fmaf(qi[P::TensorOffset + d * 4 + c], basis[4 + d], ai);
    }
    float si = g.envelope * ai, fg_i = qi[P::ScalarOffset + c] + si;
    rgi = fmaf(g.envelope * fg_i, wi.x * radial.y, rgi);
    egi = fmaf(fg_i + si, vi, egi);
    float api = vi * g.envelope * g.envelope;
    if (c < 4) {
#pragma unroll
      for (int d = 0; d < 3; ++d) dbi[1 + d] = fmaf(qi[P::VectorOffset + d * 4 + c], api, dbi[1 + d]);
#pragma unroll
      for (int d = 0; d < 5; ++d) dbi[4 + d] = fmaf(qi[P::TensorOffset + d * 4 + c], api, dbi[4 + d]);
    }
  }
  const float square = g.envelope * g.envelope;
  egi = fmaf(4.f * square * g.envelope, qi[P::MomentWidth + 1], fmaf(2.f * g.envelope, qi[P::MomentWidth], egi));
  const float de = c3_envelope_derivative(g.radius, a.rcut);
  rgi = fmaf(egi, de, rgi);
  basis_vjp(g, dbi, direction, rgi, gij);
}

template <int Warps>
__global__ void raw_force_onesided_kernel(Arguments a, DpaDirectNeighborView view, float* f, float* v) {
  using P = Profile<8, 2, false>;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const long i = long(blockIdx.x) * Warps + warp;
  __shared__ float own[Warps][P::StateWidth];
  if (i < a.node_count)
    for (int k = lane; k < P::StateWidth; k += 32) own[warp][k] = a.moment_gradient[i * P::StateWidth + k];
  __syncthreads();
  if (i >= a.node_count) return;
  const long begin = a.destination_row_ptr[i];
  const long live = reinterpret_cast<const int*>(a.edge_mask)[i];
  float force[3] = {}, virial[9] = {};
  const int ti = int(a.atype[i]);
  for (long position = lane; position < live; position += 32) {
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, reinterpret_cast<const float4*>(a.edge_vec)[begin + position], xyz);
    const std::uint32_t j = static_cast<std::uint32_t>(g.source);
    float gij[3];
    pair_vjp_one(a, g, own[warp], ti, g.source_type, gij);
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      force[c] += gij[c];
      if (!(a.radial_modes & 16)) atomicAdd(f + long(j) * 3 + c, -gij[c]);
#pragma unroll
      for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * xyz[d];
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

inline void launch_raw_force_onesided_c8(const Arguments& a, const DpaDirectNeighborView& view, cudaStream_t s,
                                         float* f, float* v) {
  cudaMemsetAsync(f, 0, a.node_count * 3 * sizeof(float), s);
  raw_force_onesided_kernel<2><<<(a.node_count + 1) / 2, 64, 0, s>>>(a, view, f, v);
}
