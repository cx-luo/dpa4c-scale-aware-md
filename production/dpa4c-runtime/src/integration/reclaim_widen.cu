#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include "preclear_api.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace deepmd_dpa4c { namespace chain_reclaimed_widen {

#include "raw_force_split.cuh"
}}
#include "raw_force_direct.cuh"
namespace deepmd_dpa4c { namespace chain_reclaimed_widen {

__global__ void reclaim_widen_kernel(long n_force, long n_total,
                                    const float* force,
                                    const float* virial,
                                    double* force_out,
                                    double* virial_out,
                                    unsigned long long* packed,
                                    long nodes) {
  constexpr double inv = 1.0 / 16777216.0;
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
      // Reclaim both packed words while they are already hot.  The next
      // force kernel is ordered after this widen kernel on the same captured
      // stream, so no separate per-step memset node is required.
      // Kernel completion on the captured stream orders these resets before
      // the next replay's atomics.  One thread owns both words for this atom.
      packed[2 * atom] = 0ull;
      packed[2 * atom + 1] = 0ull;
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
  unsigned long long* packed_eager = nullptr;
  unsigned long long* packed_graph = nullptr;
  unsigned long long* current = nullptr;
  long capacity = 0;
  long configured_nodes = 0;
  long nodes = 0;
  long preclear_nodes = 0;
  int* skip_force = nullptr;
  bool installed = false;
  bool armed = false;
  bool launch_direct = false;
  bool graph_direct = false;
  bool last_direct = false;
  double* force_out = nullptr;
  double* virial_out = nullptr;
  long output_nodes = 0;
  cudaStream_t graph_stream = nullptr;
  cudaEvent_t materialized = nullptr;
};
Storage& storage() { static Storage s; return s; }

bool enabled() {
  static const bool value = [] {
    const char* p = std::getenv("DPA_RECLAIM_FORCE_WIDEN");
    return p && p[0] == '1';
  }();
  return value;
}
bool remove_widen_enabled() {
  static const bool value = [] {
    const char* p = std::getenv("DPA_SMALL_NVT_PACKED_FORCE");
    return p && p[0] == '1';
  }();
  return value;
}
long maximum_nodes() {
  static const long value = [] {
    const char* p = std::getenv("DPA_RECLAIM_FORCE_WIDEN_MAX_NODES");
    const long n = p ? std::atol(p) : 4096L;
    return n > 0 ? n : 4096L;
  }();
  return value;
}

bool prepare(long nodes, cudaStream_t stream) {
  auto& s = storage();
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return false;
  if (nodes > s.capacity) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    const long capacity = nodes + nodes / 4 + 64;
    unsigned long long* eager = nullptr;
    unsigned long long* graph = nullptr;
    int* skip_force = nullptr;
    const std::size_t bytes = capacity * 2 * sizeof(unsigned long long);
    if (cudaMalloc(&eager, bytes) != cudaSuccess ||
        cudaMalloc(&graph, bytes) != cudaSuccess ||
        cudaMalloc(&skip_force, sizeof(int)) != cudaSuccess ||
        cudaEventCreateWithFlags(&s.materialized, cudaEventDisableTiming) != cudaSuccess) {
      if (eager) cudaFree(eager);
      if (graph) cudaFree(graph);
      if (skip_force) cudaFree(skip_force);
      return false;
    }
    if (cudaMemset(skip_force, 0, sizeof(int)) != cudaSuccess) {
      cudaFree(eager);
      cudaFree(graph);
      cudaFree(skip_force);
      return false;
    }
    s.packed_eager = eager;
    s.packed_graph = graph;
    s.skip_force = skip_force;
    s.capacity = capacity;
    s.configured_nodes = 0;
  }
  const std::size_t bytes = nodes * 2 * sizeof(unsigned long long);
  if (s.armed && s.preclear_nodes != nodes) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    if (deepmd_dpa4c::chain_packed_nvt::configure_preclear(
            s.packed_graph, nodes) != cudaSuccess)
      return false;
    s.preclear_nodes = nodes;
  }
  if (s.configured_nodes != nodes) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    if (cudaMemsetAsync(s.packed_eager, 0, bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(s.packed_graph, 0, bytes, stream) != cudaSuccess)
      return false;
    s.configured_nodes = nodes;
  }
  if (capture == cudaStreamCaptureStatusNone) {
    // Eager model warmups may execute without a matching widen.  Keep them in
    // a scratch accumulator and preserve the original clear-before-force rule.
    s.current = s.packed_eager;
    s.launch_direct = false;
    s.last_direct = false;
    if (cudaMemsetAsync(s.current, 0, bytes, stream) != cudaSuccess)
      return false;
  } else {
    s.current = s.packed_graph;
    s.launch_direct = s.armed && s.force_out && s.virial_out &&
        s.output_nodes == nodes;
    static bool capture_prepare_shown = false;
    if (!capture_prepare_shown) {
      std::fprintf(stderr,
          "PACKED_NVT_CAPTURE_PREP armed=%d direct=%d nodes=%ld saved_nodes=%ld force=%p virial=%p stream=%p\n",
          s.armed ? 1 : 0, s.launch_direct ? 1 : 0, nodes, s.output_nodes,
          s.force_out, s.virial_out, reinterpret_cast<void*>(stream));
      capture_prepare_shown = true;
    }
  }
  s.nodes = nodes;
  return true;
}

void reclaim_widen_dispatch(long n_force, long n_total,
                            const float* force, const float* virial,
                            double* force_out, double* virial_out, dim3 grid, dim3 block,
                            std::size_t shared, cudaStream_t stream) {
  auto& s = storage();
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    capture = cudaStreamCaptureStatusInvalidated;
  const bool exact_direct = capture != cudaStreamCaptureStatusNone &&
      s.launch_direct && force_out == s.force_out &&
      virial_out == s.virial_out && n_force == s.nodes * 3;
  if (capture != cudaStreamCaptureStatusNone) {
    static bool capture_dispatch_shown = false;
    if (!capture_dispatch_shown) {
      std::fprintf(stderr,
          "PACKED_NVT_CAPTURE_DISPATCH capture=%d launch=%d exact=%d current_force=%p saved_force=%p current_virial=%p saved_virial=%p nforce=%ld nodes=%ld\n",
          int(capture), s.launch_direct ? 1 : 0, exact_direct ? 1 : 0,
          force_out, s.force_out, virial_out, s.virial_out, n_force, s.nodes);
      capture_dispatch_shown = true;
    }
  }
  if (exact_direct) {
    s.graph_direct = true;
    s.last_direct = true;
    s.graph_stream = stream;
    static bool shown = false;
    if (!shown) {
      std::fprintf(stderr,
          "PACKED_NVT_REMOVE_WIDEN captured nodes=%ld force=%p virial=%p stream=%p\n",
          s.nodes, force_out, virial_out, reinterpret_cast<void*>(stream));
      shown = true;
    }
    return;
  }
  if (capture == cudaStreamCaptureStatusNone) {
    s.force_out = force_out;
    s.virial_out = virial_out;
    s.output_nodes = s.nodes;
    s.last_direct = false;
  } else {
    s.graph_direct = false;
    s.last_direct = false;
  }
  reclaim_widen_kernel<<<grid, block, shared, stream>>>(
      n_force, n_total, force, virial, force_out, virial_out,
      s.current, s.nodes);
}

}
void install(long nodes) {
  if (enabled() && nodes > 0 && nodes <= maximum_nodes()) storage().installed=true;
}
bool dispatch_widen(long nf,long nt,const float* f,const float* v,double* fo,double* vo,
                    dim3 grid,dim3 block,std::size_t shared,cudaStream_t stream) {
  if (!storage().installed) return false;
  reclaim_widen_dispatch(nf,nt,f,v,fo,vo,grid,block,shared,stream);
  return true;
}
bool owns_widen() { return storage().installed; }

// Reuse the integration adapter's explicit NVT capability. A model/size or
// species guess must never select this route for NPT or an unknown fix.
__attribute__((visibility("hidden"))) bool small_nvt_armed() {
  return storage().armed && remove_widen_enabled();
}


extern "C" __attribute__((visibility("default")))
int dpa_chain_small_nvt_arm_small_nvt(int enable) {
  auto& s = storage();
  if (!remove_widen_enabled()) return 0;
  if (!enable && s.graph_direct) return -1;
  s.armed = enable != 0;
  if (!s.armed) {
    s.launch_direct = false;
    s.last_direct = false;
    deepmd_dpa4c::chain_packed_nvt::configure_preclear(nullptr, 0);
    s.preclear_nodes = 0;
  }
  return 1;
}

extern "C" __attribute__((visibility("default")))
int dpa_chain_small_nvt_get_packed_force(unsigned long long** packed,
                                    int** state_flag,
                                    long* nodes) {
  auto& s = storage();
  if (!s.installed || !s.armed || !s.graph_direct || !s.last_direct ||
      !s.packed_graph || !s.force_out || s.nodes <= 0)
    return 0;
  if (packed) *packed = s.packed_graph;
  if (state_flag) *state_flag = s.skip_force;
  if (nodes) *nodes = s.nodes;
  return 1;
}

extern "C" __attribute__((visibility("default")))
int dpa_chain_small_nvt_materialize_force(void* consumer_stream_opaque) {
  auto& s = storage();
  if (!s.armed || !s.graph_direct || !s.last_direct) return 0;
  if (!s.packed_graph || !s.force_out || !s.graph_stream || !s.materialized)
    return -1;
  cudaStream_t consumer = reinterpret_cast<cudaStream_t>(consumer_stream_opaque);
  const int threads = 128;
  const int blocks = static_cast<int>((s.nodes + threads - 1) / threads);
  deepmd_dpa4c::chain_packed_nvt::materialize_packed_force_kernel
      <<<blocks, threads, 0, s.graph_stream>>>(
          s.nodes, s.packed_graph, s.force_out);
  if (cudaGetLastError() != cudaSuccess) return -1;
  if (consumer != s.graph_stream) {
    if (cudaEventRecord(s.materialized, s.graph_stream) != cudaSuccess ||
        cudaStreamWaitEvent(consumer, s.materialized, 0) != cudaSuccess)
      return -1;
  }
  return 1;
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
  if (s.launch_direct) {
    deepmd_dpa4c::chain_packed_nvt::raw_force_split_pack_direct_virial_kernel<2, 2>
        <<<(a.node_count + 1) / 2, 64, 0, stream>>>(
            a, view, s.current, virial, s.virial_out);
  } else {
    raw_force_split_pack_kernel<2, 2>
        <<<(a.node_count + 1) / 2, 64, 0, stream>>>(
            a, view, s.current, virial);
  }
  if (cudaGetLastError() != cudaSuccess) return false;
  static bool shown = false;
  if (!shown) {
    cudaFuncAttributes attr{};
    int resident = 0;
    cudaFuncGetAttributes(&attr, raw_force_split_pack_kernel<2, 2>);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, raw_force_split_pack_kernel<2, 2>, 64, 0);
    std::fprintf(stderr,
                 "RECLAIMED_WIDEN_RECLAIM_WIDEN active nodes=%ld blocks=%ld regs=%d "
                 "local=%zu shared=%zu resident=%d\n",
                 a.node_count, (a.node_count + 1) / 2, attr.numRegs,
                 attr.localSizeBytes, attr.sharedSizeBytes, resident);
    shown = true;
  }
  return true;
}

}}  // namespace deepmd_dpa4c::chain_reclaimed_widen
