#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>

#include <mutex>

#include <algorithm>
#include <atomic>
#include <limits>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "fit_node_bridge.h"

#define FITTING_IMAGE_FORCE_K3N4 1
#include "fitting_core.cuh"
#include "split_runtime.cuh"
#include "lowreg_pipeline.cuh"
#include "block_pipeline.cuh"

struct FittingLayerPlan;

using FittingRangeFn = bool (*)(
    cudaStream_t, const FittingLayerPlan&, float*, long, const long*,
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    const std::vector<int64_t>&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, int64_t, const double*, long, double*);


namespace {
__attribute__((visibility("hidden")))
std::atomic<std::uintptr_t> g_unit_seed_begin{0};
__attribute__((visibility("hidden")))
std::atomic<std::uintptr_t> g_unit_seed_end{0};
}

// The direct DeePMD executor owns this persistent torch::ones buffer.  Only
// exact subranges of that buffer may enter the fixed-unit-seed ISA kernel;
// arbitrary fitting callers fall back to the generic split implementation.
extern "C" __attribute__((visibility("default")))
void dpa_chain_set_unit_seed_range(const double* base, long count) {
  g_unit_seed_begin.store(0, std::memory_order_release);
  if (!base || count <= 0 ||
      static_cast<unsigned long>(count) >
          std::numeric_limits<std::uintptr_t>::max() / sizeof(double)) {
    g_unit_seed_end.store(0, std::memory_order_relaxed);
    return;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(base);
  const auto bytes = static_cast<std::uintptr_t>(count) * sizeof(double);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    g_unit_seed_end.store(0, std::memory_order_relaxed);
    return;
  }
  g_unit_seed_end.store(begin + bytes, std::memory_order_relaxed);
  g_unit_seed_begin.store(begin, std::memory_order_release);
}

namespace fitting_pipeline {
__attribute__((visibility("hidden")))
bool unit_seed_range(const double* seed, long count) {
  if (!seed || count <= 0) return false;
  const auto registered = g_unit_seed_begin.load(std::memory_order_acquire);
  if (!registered) return false;
  const auto registered_end = g_unit_seed_end.load(std::memory_order_relaxed);
  if (static_cast<unsigned long>(count) >
      std::numeric_limits<std::uintptr_t>::max() / sizeof(double)) {
    return false;
  }
  const auto candidate = reinterpret_cast<std::uintptr_t>(seed);
  const auto bytes = static_cast<std::uintptr_t>(count) * sizeof(double);
  const auto candidate_end = candidate + bytes;
  const bool matched = candidate >= registered && candidate_end >= candidate &&
                       candidate_end <= registered_end;
  return matched;
}
}

namespace {



bool fitting_range_dispatch(
    cudaStream_t stream, const FittingLayerPlan& plan, float* x,
    long input_width, const long* atype,
    const std::vector<torch::Tensor>& ws,
    const std::vector<torch::Tensor>& bs,
    const std::vector<int64_t>& resnets, const torch::Tensor& w_head,
    const torch::Tensor& b_head, const torch::Tensor& bias_atom_e, int64_t act,
    const double* seed, long n, double* energy) {
  static const long split_min_nodes = [] {
    const char* value = std::getenv("SPLIT_FITTING_SPLIT_MIN_NODES");
    return value ? std::max<long>(1, std::atol(value)) : 4096;
  }();
  static const bool fitting_pipeline_requested = [] {
    const char* value = std::getenv("DPA_FITTING_PIPELINE");
    return value && value[0] == '1';
  }();
  static const long fitting_pipeline_min_nodes = [] {
    const char* value = std::getenv("DPA_FITTING_PIPELINE_MIN_NODES");
    return value ? std::max<long>(65536, std::atol(value)) : 100000;
  }();
  bool handled = false;
  if (fitting_pipeline_requested && n >= fitting_pipeline_min_nodes &&
      fitting_pipeline::unit_seed_range(seed, n)) {
    handled = fitting_pipeline::energy_gradient(stream, x, input_width, atype, ws, bs,
                                    resnets, w_head, b_head, bias_atom_e, act,
                                    seed, n, energy);
  }
  if (!handled && n >= split_min_nodes) {
    handled = split_fitting::energy_gradient(stream, x, input_width, atype, ws, bs,
                                   resnets, w_head, b_head, bias_atom_e, act,
                                   seed, n, energy);
  }
  if (!handled) {
    handled = k3n::energy_gradient(stream, x, input_width, atype, ws, bs,
                                   resnets, w_head, b_head, bias_atom_e, act,
                                   seed, n, energy);
  }
  if (!handled) {
    return false;
  }
  static bool announced = false;
  if (!announced) {
    std::fprintf(stderr,
                 "SPLIT_FITTING exact split fitting hook active nodes=%ld split=%d "
                 "products=%d period=%d col2=%d\n",
                 n, n >= split_min_nodes ? 1 : 0,
                 k3n::kProducts, k3n::kPeriod,
                 k3n::use_col2_env() ? 1 : 0);
    announced = true;
  }
  return true;
}

}  // namespace
extern "C" bool dpa_chain_try_fitting_range(
    cudaStream_t stream, const FittingLayerPlan& plan, float* x,
    long width, const long* types, const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases, const std::vector<int64_t>& residuals,
    const torch::Tensor& head, const torch::Tensor& head_bias,
    const torch::Tensor& atom_bias, int64_t activation, const double* seed,
    long count, double* energy) {
  return fitting_range_dispatch(stream, plan, x, width, types, weights, biases,
       residuals, head, head_bias, atom_bias, activation, seed, count, energy);
}
