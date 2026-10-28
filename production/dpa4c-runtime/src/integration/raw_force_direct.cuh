#pragma once

namespace deepmd_dpa4c { namespace chain_packed_nvt {
using namespace deepmd_dpa4c::chain_reclaimed_widen;

// The outer dispatch remains chain_reclaimed_widen::try_launch_raw_force_c8.  This
// PACKED_NVT-only implementation keeps the exact HERMITEf force/virial arithmetic and
// writes the already-rounded FP32 virial directly into its stable FP64
// consumer buffer.  That makes the later widen graph node unnecessary.
template <int Warps, int E>
__global__ void raw_force_split_pack_direct_virial_kernel(
    Arguments a, DpaDirectNeighborView view, unsigned long long* fp,
    float* v, double* direct_v) {
  using P = Profile<8, 2, false>;
  constexpr int Groups = 32 / E;
  static_assert(E >= 2, "pack needs two lanes per edge");
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int base = lane % E, group = lane / E;
  const long i = a.node_begin + long(blockIdx.x) * Warps + warp;
  __shared__ float own[Warps][P::StateWidth];
  if (i < a.node_begin + a.node_count)
    for (int k = lane; k < P::StateWidth; k += 32)
      own[warp][k] = a.moment_gradient[i * P::StateWidth + k];
  __syncthreads();
  if (i >= a.node_begin + a.node_count) return;
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
    if (valid && base < 2)
      pack_add(fp, long(static_cast<std::uint32_t>(g.source)), gij, -1.f, base);
    if (valid && base == 0) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        force[c] += gij[c];
#pragma unroll
        for (int d = 0; d < 3; ++d)
          virial[c * 3 + d] -= gij[c] * xyz[d];
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c)
      force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
#pragma unroll
    for (int c = 0; c < 9; ++c)
      virial[c] += __shfl_down_sync(0xffffffffu, virial[c], offset);
  }
  for (int c = 0; c < 3; ++c)
    force[c] = __shfl_sync(0xffffffffu, force[c], 0);
  if (lane < 2) pack_add(fp, i, force, 1.f, lane);
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < 9; ++c) {
      v[i * 9 + c] = virial[c];
      direct_v[i * 9 + c] = static_cast<double>(virial[c]);
    }
  }
}

__global__ void materialize_packed_force_kernel(
    long nodes, const unsigned long long* packed, double* force_out) {
  constexpr double inv = 1.0 / 16777216.0;
  for (long atom = long(blockIdx.x) * blockDim.x + threadIdx.x;
       atom < nodes; atom += long(gridDim.x) * blockDim.x) {
    const unsigned long long xy = packed[2 * atom];
    const long long qx = static_cast<int>(static_cast<unsigned>(xy));
    const long long qy = (static_cast<long long>(xy) - qx) >> 32;
    const long long qz = static_cast<long long>(packed[2 * atom + 1]);
    const long base = 3 * atom;
    force_out[base] =
        static_cast<double>(static_cast<float>(double(qx) * inv));
    force_out[base + 1] =
        static_cast<double>(static_cast<float>(double(qy) * inv));
    force_out[base + 2] =
        static_cast<double>(static_cast<float>(double(qz) * inv));
  }
}
}}
