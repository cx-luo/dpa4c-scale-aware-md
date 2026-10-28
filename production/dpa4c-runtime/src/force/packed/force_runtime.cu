#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
namespace deepmd_dpa4c {

namespace chain_factored_force {
void launch_raw_force_c8(const Arguments&, const DpaDirectNeighborView&,
                         cudaStream_t, float*, float*);
}
namespace chain_persistent_force {
void launch_raw_force_c8(const Arguments&, const DpaDirectNeighborView&,
                         cudaStream_t, float*, float*);
}
namespace chain_virial_policy {

#include "raw_force_split.cuh"
#include "shared_table_refine.cuh"
#include "radial_math.cuh"
#include "algebra_force.cuh"

__global__ void radial_table_build_split_table(const float* source, float h,
                                      float* destination) {
  const int item = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= 600 * 8) return;
  const int interval = item >> 3;
  const int channel = item & 7;
  const float* row = source + interval * 48;
  const float4 low = reinterpret_cast<const float4*>(row)[channel];
  const float2 high = reinterpret_cast<const float2*>(row + 32)[channel];
  const double hd = double(h);
  const double h2 = hd * hd;
  const double h3 = h2 * hd;
  const float c2 = float(double(low.z) - h2 * double(high.x) -
                         2.0 * h3 * double(high.y));
  const float c3 = float(double(low.w) + 2.0 * hd * double(high.x) +
                         3.0 * h2 * double(high.y));
  destination[item] = low.x;
  destination[600 * 8 + item] = low.y;
  destination[2 * 600 * 8 + item] = c2;
  destination[3 * 600 * 8 + item] = c3;
}

struct HERMITEStorage {
  unsigned long long* packed = nullptr;
  long capacity = 0;
  float* split = nullptr;
  const float* source = nullptr;
  float source_stride = 0.0f;
  int* need_virial = nullptr;
  bool variable_virial = false;
  int virial_policy = 0;  // 1=NVT variable, 2=NPT fixed
  int virial_request = 0;
  bool direct_fp32_global_virial = false;
};

static HERMITEStorage& hermite_storage() {
  static HERMITEStorage value;
  return value;
}

__attribute__((visibility("hidden"))) const int* requested_virial_device_pointer() {
  auto& storage = hermite_storage();
  return storage.variable_virial ? storage.need_virial : nullptr;
}

__attribute__((visibility("hidden"))) void reset_direct_fp32_global_virial() {
  hermite_storage().direct_fp32_global_virial = false;
}

__attribute__((visibility("hidden"))) bool use_direct_fp32_global_virial() {
  return hermite_storage().direct_fp32_global_virial;
}

// The request byte may vary within an NVT graph. The policy selects the
// captured force/output representation and must therefore participate in the
// model graph key, including the NPT-six -> full-centroid transition.
__attribute__((visibility("hidden"))) int graph_route_policy() {
  return hermite_storage().virial_policy;
}

void set_need_virial(int encoded, cudaStream_t stream) {
  auto& storage = hermite_storage();
  const int mode = encoded & 3;
  const int policy = (encoded >> 2) & 3;
  storage.variable_virial = policy == 1;
  storage.virial_policy = policy;
  storage.virial_request = mode;
  if (!storage.need_virial) {
    if (cudaMalloc(&storage.need_virial, sizeof(int)) != cudaSuccess) {
      cudaGetLastError();
      return;
    }
    cudaMemsetAsync(storage.need_virial, 0, sizeof(int), stream);
  }
  // sm_80 is little-endian.  Updating the already-zeroed low byte preserves
  // the exact integer value without a host-lifetime-sensitive 4-byte H2D copy.
  cudaMemsetAsync(storage.need_virial, mode, 1, stream);
}

static bool hermite_prepare(const Arguments& a, cudaStream_t stream, long required_nodes = 0) {
  auto& storage = hermite_storage();
  cudaStreamCaptureStatus capture;
  cudaStreamIsCapturing(stream, &capture);
  const long needed = required_nodes > 0 ? required_nodes : a.node_count;
  if (needed > storage.capacity) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    unsigned long long* next = nullptr;
    if (cudaMalloc(&next, (needed + needed / 4 + 64) * 16) !=
        cudaSuccess) return false;
    // Captured graphs may retain the old pointer; keep it alive deliberately.
    storage.packed = next;
    storage.capacity = needed + needed / 4 + 64;
  }
  if (!storage.split || storage.source != a.table ||
      storage.source_stride != a.table_stride) {
    if (capture != cudaStreamCaptureStatusNone) return false;
    float* next = nullptr;
    if (cudaMalloc(&next, 4 * 600 * 8 * sizeof(float)) != cudaSuccess)
      return false;
    storage.split = next;
    storage.source = a.table;
    storage.source_stride = a.table_stride;
    radial_table_build_split_table<<<19, 256, 0, stream>>>(a.table, a.table_stride,
                                                   storage.split);
    if (cudaGetLastError() != cudaSuccess) return false;
  }
  return true;
}

template <int Warps, bool Conditional>
static bool virial_policy_setup_kernel() {
  static const bool okay = [] {
    auto status = cudaFuncSetAttribute(
        virial_policy_force6<Warps, Conditional>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        3 * 600 * 8 * int(sizeof(float)));
    if (status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  return okay;
}

template <int Warps>
static bool virial_policy_setup_missing() {
  static const bool okay = [] {
    auto status = cudaFuncSetAttribute(
        virial_policy_missing3<Warps>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        3 * 600 * 8 * int(sizeof(float)));
    if (status != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
  }();
  return okay;
}

static bool virial_policy_npt_global_six() {
  static const bool enabled = [] {
    const char* value = std::getenv("DPA_NPT_GLOBAL_VIRIAL_COMPONENTS");
    return value && std::atoi(value) != 0;
  }();
  return enabled;
}

static bool direct_fp32_global_virial_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DPA_DIRECT_FP32_GLOBAL_VIRIAL");
    return value && std::atoi(value) != 0;
  }();
  return enabled;
}

template <int Warps, bool NeedVirial, bool Conditional>
static bool hermite_setup_kernel() {
  static const bool okay = [] {
    auto status = cudaFuncSetAttribute(
        hermite_force<Warps, 1, 2, true, NeedVirial, Conditional>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        3 * 600 * 8 * int(sizeof(float)));
    if (status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  return okay;
}

void launch_force(const Arguments& a, const DpaDirectNeighborView& view,
                  cudaStream_t stream, float* f, float* v) {
  const long minimum = std::getenv("HERMITE_MIN_NODES")
      ? std::max(4096L, std::atol(std::getenv("HERMITE_MIN_NODES"))) : 4096L;
  if (a.node_count < minimum || a.has_spin || a.lmax != 2 ||
      a.table_width != 8 || a.interval_count != 600 || a.radial_modes != 0 ||
      a.node_begin || !hermite_prepare(a, stream)) {
    chain_factored_force::launch_raw_force_c8(a, view, stream, f, v);
    return;
  }
  int warps = std::getenv("HERMITE_WARPS") ? std::atoi(std::getenv("HERMITE_WARPS")) : 12;
  int blocks = std::getenv("HERMITE_BLOCKS") ? std::atoi(std::getenv("HERMITE_BLOCKS")) : 192;
  blocks = std::max(1, blocks);
  auto& storage = hermite_storage();
  cudaMemsetAsync(storage.packed, 0, a.node_count * 16, stream);
  Arguments split_args = a;
  split_args.table = storage.split;
  const float* cubic = storage.split;
  const long dual_minimum = std::getenv("DUAL_FORCE_DUAL_MIN_NODES")
      ? std::max(65536L, std::atol(std::getenv("DUAL_FORCE_DUAL_MIN_NODES")))
      : 65536L;
  const bool dual = a.node_count >= dual_minimum && storage.need_virial && storage.variable_virial;
  const long virial_policy_min_nodes = std::getenv("DPA_VIRIAL_POLICY_MIN_NODES")
      ? std::max(100000L, std::atol(std::getenv("DPA_VIRIAL_POLICY_MIN_NODES"))) : 100000L;
  const bool virial_policy_active = virial_policy_npt_global_six() && a.node_count >= virial_policy_min_nodes &&
      storage.need_virial && storage.virial_policy == 2;
  bool launched = false;
#define HERMITE_LAUNCH(W) do { \
  if (warps == W) { \
    if (virial_policy_active && dual && hermite_setup_kernel<W, false, true>() && virial_policy_setup_kernel<W, true>()) { \
      hermite_force<W, 1, 2, true, false, true><<<blocks, W * 32, 3 * 600 * 8 * int(sizeof(float)), stream>>>( \
          split_args, view, storage.packed, v, cubic, storage.need_virial); \
      virial_policy_force6<W, true><<<blocks, W * 32, 3 * 600 * 8 * int(sizeof(float)), stream>>>( \
          split_args, view, storage.packed, v, cubic, storage.need_virial); \
      launched = true; \
    } else if (virial_policy_active && !dual && virial_policy_setup_kernel<W, false>()) { \
      virial_policy_force6<W, false><<<blocks, W * 32, 3 * 600 * 8 * int(sizeof(float)), stream>>>( \
          split_args, view, storage.packed, v, cubic, nullptr); \
      launched = true; \
    } else if (!virial_policy_active && dual && hermite_setup_kernel<W, false, true>() && hermite_setup_kernel<W, true, true>()) { \
      hermite_force<W, 1, 2, true, false, true><<<blocks, W * 32, 3 * 600 * 8 * int(sizeof(float)), stream>>>( \
          split_args, view, storage.packed, v, cubic, storage.need_virial); \
      hermite_force<W, 1, 2, true, true, true><<<blocks, W * 32, 3 * 600 * 8 * int(sizeof(float)), stream>>>( \
          split_args, view, storage.packed, v, cubic, storage.need_virial); \
      launched = true; \
    } else if (!dual && hermite_setup_kernel<W, true, false>()) { \
      hermite_force<W, 1, 2, true, true, false><<<blocks, W * 32, 3 * 600 * 8 * int(sizeof(float)), stream>>>( \
          split_args, view, storage.packed, v, cubic, nullptr); \
      launched = true; \
    } \
  } \
} while (0)
  HERMITE_LAUNCH(8); HERMITE_LAUNCH(10); HERMITE_LAUNCH(12); HERMITE_LAUNCH(16);
#undef HERMITE_LAUNCH
  if (!launched) {
    storage.direct_fp32_global_virial = false;
    chain_factored_force::launch_raw_force_c8(a, view, stream, f, v);
    return;
  }
  storage.direct_fp32_global_virial = virial_policy_active &&
      storage.virial_request == 1 && direct_fp32_global_virial_enabled();
  unpack_force_kernel<<<std::min<long>((a.node_count + 255) / 256, 4096),
                        256, 0, stream>>>(a.node_count, storage.packed, f);
  static bool announced = false;
  if (!announced) {
    announced = true;
    cudaFuncAttributes attributes{};
    int resident = 0;
    const void* function = warps == 8 ?
        (dual ? reinterpret_cast<const void*>(hermite_force<8, 1, 2, true, true, true>) : reinterpret_cast<const void*>(hermite_force<8, 1, 2, true, true, false>)) :
        warps == 10 ? (dual ? reinterpret_cast<const void*>(hermite_force<10, 1, 2, true, true, true>) : reinterpret_cast<const void*>(hermite_force<10, 1, 2, true, true, false>)) :
        warps == 12 ? (dual ? reinterpret_cast<const void*>(hermite_force<12, 1, 2, true, true, true>) : reinterpret_cast<const void*>(hermite_force<12, 1, 2, true, true, false>)) :
        (dual ? reinterpret_cast<const void*>(hermite_force<16, 1, 2, true, true, true>) : reinterpret_cast<const void*>(hermite_force<16, 1, 2, true, true, false>));
    cudaFuncGetAttributes(&attributes, function);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, function, warps * 32,
        3 * 600 * 8 * int(sizeof(float)));
    std::fprintf(stderr,
        "HERMITE_CUBIC_FORCE active nodes=%ld warps=%d blocks=%d shared=%d "
        "regs=%d local=%zu resident=%d dual_virial=%d\n", a.node_count, warps, blocks,
        3 * 600 * 8 * int(sizeof(float)), attributes.numRegs,
        attributes.localSizeBytes, resident, int(dual));
  }
}

bool launch_window(const Arguments& input, const DpaDirectNeighborView& view,
                   cudaStream_t stream, float* virial, bool reset,
                   long total_nodes, int blocks) {
  if (input.node_count <= 0 || input.node_begin < 0 ||
      input.node_begin + input.node_count > total_nodes || input.has_spin ||
      input.lmax != 2 || input.table_width != 8 ||
      input.interval_count != 600 || input.radial_modes != 0 ||
      !hermite_prepare(input, stream, total_nodes)) return false;
  auto& storage = hermite_storage();
  if (reset) cudaMemsetAsync(storage.packed, 0, total_nodes * 16, stream);
  Arguments a = input;
  a.table = storage.split;
  const float* cubic = storage.split;
  const long dual_window_min = std::getenv("ADAPTIVE_WINDOWS_DUAL_MIN_WINDOW_NODES")
      ? std::max(65536L, std::atol(std::getenv("ADAPTIVE_WINDOWS_DUAL_MIN_WINDOW_NODES")))
      : 131072L;
  // A variable-virial graph launches both conditional force kernels.  That is
  // profitable for a few large windows, but launch overhead dominates dense
  // plans with many small windows. Route by the actual consumer window size.
  const bool dual = total_nodes >= 65536 && input.node_count >= dual_window_min &&
      storage.need_virial && storage.variable_virial;
  const long virial_policy_min_nodes = std::getenv("DPA_VIRIAL_POLICY_MIN_NODES")
      ? std::max(100000L, std::atol(std::getenv("DPA_VIRIAL_POLICY_MIN_NODES"))) : 100000L;
  const bool virial_policy_active = virial_policy_npt_global_six() && total_nodes >= virial_policy_min_nodes &&
      storage.need_virial && storage.virial_policy == 2;
  const int warps = std::getenv("HERMITE_WARPS")
      ? std::atoi(std::getenv("HERMITE_WARPS")) : 12;
  bool launched = false;
#define CONSUMER_PIPELINE_WINDOW(W) do { \
  if (warps == W) { \
    if (virial_policy_active && dual && hermite_setup_kernel<W,false,true>() && virial_policy_setup_kernel<W,true>()) { \
      hermite_force<W,1,2,true,false,true><<<blocks,W*32,3*600*8*int(sizeof(float)),stream>>>( \
          a,view,storage.packed,virial,cubic,storage.need_virial); \
      virial_policy_force6<W,true><<<blocks,W*32,3*600*8*int(sizeof(float)),stream>>>( \
          a,view,storage.packed,virial,cubic,storage.need_virial); \
      launched=true; \
    } else if (virial_policy_active && !dual && virial_policy_setup_kernel<W,false>()) { \
      virial_policy_force6<W,false><<<blocks,W*32,3*600*8*int(sizeof(float)),stream>>>( \
          a,view,storage.packed,virial,cubic,nullptr); \
      launched=true; \
    } else if (!virial_policy_active && dual && hermite_setup_kernel<W,false,true>() && hermite_setup_kernel<W,true,true>()) { \
      hermite_force<W,1,2,true,false,true><<<blocks,W*32,3*600*8*int(sizeof(float)),stream>>>( \
          a,view,storage.packed,virial,cubic,storage.need_virial); \
      hermite_force<W,1,2,true,true,true><<<blocks,W*32,3*600*8*int(sizeof(float)),stream>>>( \
          a,view,storage.packed,virial,cubic,storage.need_virial); \
      launched=true; \
    } else if (!dual && hermite_setup_kernel<W,true,false>()) { \
      hermite_force<W,1,2,true,true,false><<<blocks,W*32,3*600*8*int(sizeof(float)),stream>>>( \
          a,view,storage.packed,virial,cubic,nullptr); \
      launched=true; \
    } \
  } \
} while(0)
  CONSUMER_PIPELINE_WINDOW(8); CONSUMER_PIPELINE_WINDOW(10); CONSUMER_PIPELINE_WINDOW(12); CONSUMER_PIPELINE_WINDOW(16);
#undef CONSUMER_PIPELINE_WINDOW
  storage.direct_fp32_global_virial = launched && virial_policy_active &&
      storage.virial_request == 1 && direct_fp32_global_virial_enabled();
  return launched;
}

bool unpack_window_force(long total_nodes, float* force,
                         cudaStream_t stream) {
  auto& storage = hermite_storage();
  if (!force || !storage.packed || storage.capacity < total_nodes) return false;
  unpack_force_kernel<<<std::min<long>((total_nodes + 255) / 256, 4096),
                        256, 0, stream>>>(total_nodes, storage.packed, force);
  return cudaGetLastError() == cudaSuccess;
}

}
}  // namespace chain_virial_policy
