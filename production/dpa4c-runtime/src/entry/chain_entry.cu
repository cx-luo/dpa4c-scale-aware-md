#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include "fit_node_bridge.h"


namespace deepmd_dpa4c { namespace chain_virial_policy {
void set_need_virial(int, cudaStream_t);
__attribute__((visibility("hidden"))) int graph_route_policy();
__attribute__((visibility("hidden"))) const int* requested_virial_device_pointer();
__attribute__((visibility("hidden"))) void reset_direct_fp32_global_virial();
__attribute__((visibility("hidden"))) bool use_direct_fp32_global_virial();
}}
extern "C" __attribute__((visibility("default")))
int dpa_chain_virial_graph_route_policy() {
  return deepmd_dpa4c::chain_virial_policy::graph_route_policy();
}

extern "C" __attribute__((visibility("default")))
const int* dpa_chain_requested_virial_device_pointer() {
  return deepmd_dpa4c::chain_virial_policy::requested_virial_device_pointer();
}

extern "C" __attribute__((visibility("default")))
int dpa_chain_use_direct_fp32_global_virial() {
  return deepmd_dpa4c::chain_virial_policy::use_direct_fp32_global_virial() ? 1 : 0;
}

extern "C" __attribute__((visibility("default")))
void dpa_chain_set_need_virial_stream(int need_virial, void* stream) {
  // VIRIAL_REQUEST emits a legacy boolean for NVT. VIRIAL_POLICY added policy bits so that the
  // force runtime can select the no-virial specialization between thermo
  // steps. Upgrade only the old 0/1 wire format; encoded callers pass through.
  int encoded = need_virial;
  if ((encoded & ~1) == 0) encoded |= (1 << 2);  // variable NVT policy
  // CENTROID_ROUTE: a full centroid tensor request must retain all nine components.
  // Route NPT policy=2, request=2 to the existing exact full-nine fallback;
  // policy=2 remains reserved for the ordinary six-component global request.
  if (encoded == ((2 << 2) | 2)) encoded = 2;
  deepmd_dpa4c::chain_virial_policy::set_need_virial(
      encoded, reinterpret_cast<cudaStream_t>(stream));
}

namespace deepmd_dpa4c {
namespace chain_reclaimed_widen {
void install(long nodes);
bool owns_widen();
bool try_launch_raw_force_c8(const Arguments&, const DpaDirectNeighborView&,
                             cudaStream_t, float*, float*);
}
namespace chain_packed_widen {
void install(long nodes);
bool try_launch_raw_force_c8(const Arguments&, const DpaDirectNeighborView&,
                             cudaStream_t, float*, float*);
}
namespace chain_virial_policy {
void launch_force(const Arguments&, const DpaDirectNeighborView&,
                         cudaStream_t, float*, float*);
}
namespace chain_hermite_forward {
void launch_raw_forward_c8(const Arguments&, const DpaDirectNeighborView&,
                           cudaStream_t);
}

void launch_raw_force_c8(const Arguments& a, const DpaDirectNeighborView& view,
                         cudaStream_t stream, float* f, float* v) {
  chain_virial_policy::reset_direct_fp32_global_virial();
  chain_reclaimed_widen::install(a.node_count);
  const bool reclaimed_widen = chain_reclaimed_widen::owns_widen();
  if (!reclaimed_widen) chain_packed_widen::install(a.node_count);
  DpaDirectNeighborView force_view = view;
  if (force_view.cutoff_squared < 0.0)
    force_view.cutoff_squared = -force_view.cutoff_squared;
  if (chain_execution_windows::launch_force(a, force_view, stream, f, v)) return;
  if (reclaimed_widen) {
    if (!chain_reclaimed_widen::try_launch_raw_force_c8(a, force_view, stream, f, v))
      chain_virial_policy::launch_force(a, force_view, stream, f, v);
  } else if (!chain_packed_widen::try_launch_raw_force_c8(a, force_view, stream, f, v)) {
    chain_virial_policy::launch_force(a, force_view, stream, f, v);
  }
}

void launch_raw_forward_c8(const Arguments& a, const DpaDirectNeighborView& view,
                           cudaStream_t stream) {
  chain_fit_node::begin_forward(a, stream);
  chain_hermite_forward::launch_raw_forward_c8(a, view, stream);
}
}

namespace deepmd_dpa4c {
namespace chain_small_node {void launch_node(const Arguments&, cudaStream_t);}
void launch_direct_node_c8(const Arguments& a,cudaStream_t stream) {
  if (!chain_fit_node::launch_node(a,stream)) chain_small_node::launch_node(a,stream);
}
}
