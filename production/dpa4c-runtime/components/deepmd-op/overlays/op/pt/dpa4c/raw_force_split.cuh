// One-sided force with E lanes per edge: lane b of a group owns channels b, b+E, ...
// The row of the radial table is then read by E lanes together instead of one.

template <int E>
__device__ __forceinline__ float group_sum(float value) {
#pragma unroll
  for (int offset = 1; offset < E; offset <<= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

template <int E>
__device__ __forceinline__ void pair_vjp_split(const Arguments& a, EdgeGeometry g, const float* qi, int ti, int tj,
                                               int base, bool valid, float (&gij)[3]) {
  using P = Profile<8, 2, false>;
  float basis[9];
  fill_angular_basis(g, basis);
  float dbi[9] = {}, direction[3] = {};
  float rgi = 0.f, egi = 0.f;
  if (valid) {
    const auto loc = locate_table<true>(g.radius, a.table_stride, a.table_max, a.interval_count);
    const auto row = table_row(a.table, loc, a.table_width);
    const auto* fi = reinterpret_cast<const float2*>(a.pair_film) + (long(ti) * a.type_count + tj) * 8;
#pragma unroll
    for (int tile = 0; tile < 8 / E; ++tile) {
      const int c = base + tile * E;
      const float2 radial = evaluate_table_with_derivative(row, c, loc.coordinate, loc.clamped);
      const float2 wi = __ldg(fi + c);
      const float vi = fmaf(wi.x, radial.x, wi.y);
      float ai = 0.f;
      if (c < 4) {
#pragma unroll
        for (int d = 0; d < 3; ++d) ai = fmaf(qi[P::VectorOffset + d * 4 + c], basis[1 + d], ai);
#pragma unroll
        for (int d = 0; d < 5; ++d) ai = fmaf(qi[P::TensorOffset + d * 4 + c], basis[4 + d], ai);
      }
      const float si = g.envelope * ai, fg_i = qi[P::ScalarOffset + c] + si;
      rgi = fmaf(g.envelope * fg_i, wi.x * radial.y, rgi);
      egi = fmaf(fg_i + si, vi, egi);
      const float api = vi * g.envelope * g.envelope;
      if (c < 4) {
#pragma unroll
        for (int d = 0; d < 3; ++d) dbi[1 + d] = fmaf(qi[P::VectorOffset + d * 4 + c], api, dbi[1 + d]);
#pragma unroll
        for (int d = 0; d < 5; ++d) dbi[4 + d] = fmaf(qi[P::TensorOffset + d * 4 + c], api, dbi[4 + d]);
      }
    }
  }
  rgi = group_sum<E>(rgi);
  egi = group_sum<E>(egi);
#pragma unroll
  for (int d = 1; d < 9; ++d) dbi[d] = group_sum<E>(dbi[d]);
  const float square = g.envelope * g.envelope;
  egi = fmaf(4.f * square * g.envelope, qi[P::MomentWidth + 1], fmaf(2.f * g.envelope, qi[P::MomentWidth], egi));
  const float de = c3_envelope_derivative(g.radius, a.rcut);
  rgi = fmaf(egi, de, rgi);
  basis_vjp(g, dbi, direction, rgi, gij);
}

template <int Warps, int E>
__global__ void raw_force_split_kernel(Arguments a, DpaDirectNeighborView view, float* f, float* v) {
  using P = Profile<8, 2, false>;
  constexpr int Groups = 32 / E;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int base = lane % E, group = lane / E;
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
  const float4* records = reinterpret_cast<const float4*>(a.edge_vec) + begin;
  for (long start = 0; start < live; start += Groups) {
    const long position = start + group;
    const bool valid = position < live;
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, records[valid ? position : 0], xyz);
    float gij[3];
    pair_vjp_split<E>(a, g, own[warp], ti, g.source_type, base, valid, gij);
    if (valid && base == 0) {
      const std::uint32_t j = static_cast<std::uint32_t>(g.source);
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        force[c] += gij[c];
        if (!(a.radial_modes & 16)) atomicAdd(f + long(j) * 3 + c, -gij[c]);
#pragma unroll
        for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * xyz[d];
      }
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

constexpr float kForcePackScale = 16777216.f;

__device__ __forceinline__ void pack_add(unsigned long long* fp, long j, const float (&g)[3], float sign, int part) {
  if (part == 0) {
    const long long qx = __float2ll_rn(sign * g[0] * kForcePackScale);
    const long long qy = __float2ll_rn(sign * g[1] * kForcePackScale);
    atomicAdd(fp + 2 * j, static_cast<unsigned long long>(qx) + (static_cast<unsigned long long>(qy) << 32));
  } else {
    atomicAdd(fp + 2 * j + 1, static_cast<unsigned long long>(__float2ll_rn(sign * g[2] * kForcePackScale)));
  }
}

template <int Warps, int E>
__global__ void raw_force_split_pack_kernel(Arguments a, DpaDirectNeighborView view, unsigned long long* fp, float* v) {
  using P = Profile<8, 2, false>;
  constexpr int Groups = 32 / E;
  static_assert(E >= 2, "pack needs two lanes per edge");
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int base = lane % E, group = lane / E;
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
  const float4* records = reinterpret_cast<const float4*>(a.edge_vec) + begin;
  for (long start = 0; start < live; start += Groups) {
    const long position = start + group;
    const bool valid = position < live;
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, records[valid ? position : 0], xyz);
    float gij[3];
    pair_vjp_split<E>(a, g, own[warp], ti, g.source_type, base, valid, gij);
    if (valid && base < 2) pack_add(fp, long(static_cast<std::uint32_t>(g.source)), gij, -1.f, base);
    if (valid && base == 0) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        force[c] += gij[c];
#pragma unroll
        for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * xyz[d];
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
#pragma unroll
    for (int c = 0; c < 9; ++c) virial[c] += __shfl_down_sync(0xffffffffu, virial[c], offset);
  }
  for (int c = 0; c < 3; ++c) force[c] = __shfl_sync(0xffffffffu, force[c], 0);
  if (lane < 2) pack_add(fp, i, force, 1.f, lane);
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < 9; ++c) v[i * 9 + c] = virial[c];
  }
}

__global__ void unpack_force_kernel(long n, const unsigned long long* fp, float* f) {
  const double inv = 1.0 / 16777216.0;
  for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += long(gridDim.x) * blockDim.x) {
    const unsigned long long xy = fp[2 * i], z = fp[2 * i + 1];
    const long long lo = static_cast<int>(static_cast<unsigned>(xy & 0xffffffffull));
    const long long hi = (static_cast<long long>(xy) - lo) >> 32;
    f[3 * i] = float(double(lo) * inv);
    f[3 * i + 1] = float(double(hi) * inv);
    f[3 * i + 2] = float(double(static_cast<long long>(z)) * inv);
  }
}

inline unsigned long long* force_pack_buffer(long n) {
  static unsigned long long* buffer = nullptr;
  static long capacity = 0;
  if (n > capacity) {
    if (buffer) cudaFree(buffer);
    capacity = n + n / 4 + 64;
    cudaMalloc(&buffer, capacity * 2 * sizeof(unsigned long long));
  }
  return buffer;
}
