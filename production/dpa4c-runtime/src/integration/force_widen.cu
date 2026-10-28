#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace deepmd_dpa4c { namespace chain_packed_widen {

#include "raw_force_split.cuh"

__device__ unsigned long long* g_packed_widen_packed = nullptr;
__device__ long g_packed_widen_nodes = 0;

__global__ void packed_widen_kernel(long n_force, long n_total,
                                    const float* force,
                                    const float* virial,
                                    double* force_out,
                                    double* virial_out) {
  constexpr double inv = 1.0 / 16777216.0;
  const unsigned long long* packed = g_packed_widen_packed;
  const long nodes = g_packed_widen_nodes;
  const bool fused = packed && nodes > 0 && n_force == nodes * 3;
  const long tid = long(blockIdx.x) * blockDim.x + threadIdx.x;
  const long stride = long(gridDim.x) * blockDim.x;
  if (!fused) {
    for (long i = tid; i < n_force; i += stride)
      force_out[i] = static_cast<double>(force[i]);
  } else {
    // One thread decodes all three components.  This removes the integer /3
    // and reuses the packed xy word instead of loading it in two threads.
    for (long atom = tid; atom < nodes; atom += stride) {
      const unsigned long long xy = packed[2 * atom];
      const long long qx = static_cast<int>(static_cast<unsigned>(xy));
      const long long qy = (static_cast<long long>(xy) - qx) >> 32;
      const long long qz = static_cast<long long>(packed[2 * atom + 1]);
      const long base = 3 * atom;
      // Preserve the old unpack FP32 rounding before the final FP64 widen.
      force_out[base] =
          static_cast<double>(static_cast<float>(double(qx) * inv));
      force_out[base + 1] =
          static_cast<double>(static_cast<float>(double(qy) * inv));
      force_out[base + 2] =
          static_cast<double>(static_cast<float>(double(qz) * inv));
    }
  }
  const long n_virial = n_total - n_force;
  for (long i = tid; i < n_virial; i += stride)
    virial_out[i] = static_cast<double>(virial[i]);
}

namespace {
struct Storage {
  unsigned long long* packed = nullptr;
  long capacity = 0;
  long configured_nodes = 0;
};
Storage& storage() { static Storage s; return s; }

bool enabled() {
  static const bool value = [] {
    const char* p = std::getenv("DPA_PACKED_FORCE_WIDEN");
    return p && p[0] == '1';
  }();
  return value;
}
long maximum_nodes() {
  static const long value = [] {
    const char* p = std::getenv("DPA_PACKED_FORCE_WIDEN_MAX_NODES");
    const long n = p ? std::atol(p) : 4096L;
    return n > 0 ? n : 4096L;
  }();
  return value;
}

bool prepare(long nodes, cudaStream_t stream) {
  auto& s = storage();
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &capture);
  if (nodes > s.capacity) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    unsigned long long* next = nullptr;
    const long capacity = nodes + nodes / 4 + 64;
    if (cudaMalloc(&next, capacity * 2 * sizeof(unsigned long long)) !=
        cudaSuccess) return false;
    s.packed = next;
    s.capacity = capacity;
    s.configured_nodes = 0;
  }
  if (s.configured_nodes != nodes) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    if (cudaMemcpyToSymbolAsync(g_packed_widen_packed, &s.packed, sizeof(s.packed), 0,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess)
      return false;
    if (cudaMemcpyToSymbolAsync(g_packed_widen_nodes, &nodes, sizeof(nodes), 0,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess)
      return false;
    s.configured_nodes = nodes;
  }
  return true;
}

}

void install(long) {}
bool dispatch_widen(long nf,long nt,const float* f,const float* v,double* fo,double* vo,
                    dim3 grid,dim3 block,std::size_t shared,cudaStream_t stream) {
  if (!enabled() || nf <= 0 || nf/3 > maximum_nodes()) return false;
  packed_widen_kernel<<<grid,block,shared,stream>>>(nf,nt,f,v,fo,vo);
  return true;
}

bool try_launch_raw_force_c8(const Arguments& a,
                             const DpaDirectNeighborView& view,
                             cudaStream_t stream, float* force, float* virial) {
  if (!enabled() || a.node_count <= 0 || a.node_count > maximum_nodes() ||
      a.node_begin || a.has_spin || a.lmax != 2 || a.table_width != 8 ||
      a.interval_count != 600 || a.radial_modes != 0 || !force || !virial ||
      !prepare(a.node_count, stream))
    return false;
  auto& s = storage();
  cudaMemsetAsync(s.packed, 0,
                  a.node_count * 2 * sizeof(unsigned long long), stream);
  raw_force_split_pack_kernel<2, 2>
      <<<(a.node_count + 1) / 2, 64, 0, stream>>>(
          a, view, s.packed, virial);
  if (cudaGetLastError() != cudaSuccess) return false;
  static bool shown = false;
  if (!shown) {
    cudaFuncAttributes attr{};
    int resident = 0;
    cudaFuncGetAttributes(&attr, raw_force_split_pack_kernel<2, 2>);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, raw_force_split_pack_kernel<2, 2>, 64, 0);
    std::fprintf(stderr,
                 "PACKED_WIDEN_PACKED_WIDEN active nodes=%ld blocks=%ld regs=%d "
                 "local=%zu shared=%zu resident=%d\n",
                 a.node_count, (a.node_count + 1) / 2, attr.numRegs,
                 attr.localSizeBytes, attr.sharedSizeBytes, resident);
    shown = true;
  }
  return true;
}

}}  // namespace deepmd_dpa4c::chain_packed_widen
