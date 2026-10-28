#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "fit_node_bridge.h"

namespace {
inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "EXECUTION_WINDOWS CUDA failure: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
}

namespace fitting_pipeline {
void launch_range(const deepmd_dpa4c::chain_fit_node::FitPayload&, long, long,
                  cudaStream_t, int);
}
namespace deepmd_dpa4c { namespace chain_fit_node {
void launch_fused_range(const FitPayload&, const Arguments&, long, long, cudaStream_t);
}}

namespace deepmd_dpa4c {
namespace chain_small_node { void launch_node(const Arguments&, cudaStream_t); }
namespace chain_virial_policy {
bool launch_window(const Arguments&, const DpaDirectNeighborView&, cudaStream_t,
                   float*, bool, long, int);
bool unpack_window_force(long, float*, cudaStream_t);
}
namespace chain_execution_windows {
bool launch_force(const Arguments& force_arguments,
                  const DpaDirectNeighborView& view, cudaStream_t stream,
                  float* force, float* virial) {
  chain_fit_node::FitPayload fit{};
  Arguments node{};
  if (!chain_fit_node::take_large_pipeline(fit, node, stream)) return false;
  if (fit.n != force_arguments.node_count || fit.n != node.node_count ||
      force_arguments.node_begin != 0 || node.node_begin != 0 ||
      fit.n < 4096) std::abort();
  const long total = fit.n;
  const bool medium = total < 100000;
  const long candidate_slots_per_node = total > 0
      ? (force_arguments.edge_count + total - 1) / total : 0;
  const int density_threshold = std::getenv("EXECUTION_WINDOWS_DENSITY_THRESHOLD")
      ? std::max(1, std::atoi(std::getenv("EXECUTION_WINDOWS_DENSITY_THRESHOLD"))) : 128;
  const int configured_windows = std::getenv("EXECUTION_WINDOWS_WINDOWS")
      ? std::atoi(std::getenv("EXECUTION_WINDOWS_WINDOWS")) : 0;
  const int medium_windows = std::getenv("FIT_NODE_PIPELINE_WINDOWS")
      ? std::atoi(std::getenv("FIT_NODE_PIPELINE_WINDOWS")) : 2;
  const int dense_windows_dense_windows = std::getenv("DPA_DENSE_WINDOWS_DENSE_WINDOWS")
      ? std::atoi(std::getenv("DPA_DENSE_WINDOWS_DENSE_WINDOWS")) : 0;
  const bool dense_windows_dense_override = !medium &&
      candidate_slots_per_node >= density_threshold && dense_windows_dense_windows > 0;
  int windows = medium
      ? std::max(2, std::min(8, medium_windows))
      : (dense_windows_dense_override
          ? std::max(2, std::min(8, dense_windows_dense_windows))
          : (configured_windows > 0
              ? std::max(2, std::min(8, configured_windows))
              : (candidate_slots_per_node < density_threshold ? 2 : 6)));
  const int forward_windows = chain_fit_node::forward_pipeline_windows(stream);
  if (forward_windows) {
    if (forward_windows != windows) std::abort();
    windows = forward_windows;
  }
  const int fit_blocks = std::getenv("CONSUMER_PIPELINE_FIT_BLOCKS")
      ? std::max(1, std::atoi(std::getenv("CONSUMER_PIPELINE_FIT_BLOCKS"))) : 40;
  const int force_blocks = medium
      ? (std::getenv("FIT_NODE_PIPELINE_FORCE_BLOCKS")
          ? std::max(1, std::atoi(std::getenv("FIT_NODE_PIPELINE_FORCE_BLOCKS"))) : 128)
      : (std::getenv("CONSUMER_PIPELINE_FORCE_BLOCKS")
          ? std::max(1, std::atoi(std::getenv("CONSUMER_PIPELINE_FORCE_BLOCKS"))) : 128);

  static cudaStream_t side = nullptr;
  static cudaEvent_t ready[8] = {};
  static cudaEvent_t done = nullptr;
  if (!side) {
    check_cuda(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    for (int i = 0; i < 8; ++i)
      check_cuda(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
    check_cuda(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  }

  auto node_window = [&](long begin, long count) {
    Arguments window = node;
    window.node_begin = 0;
    window.node_count = count;
    window.state += begin * 42;
    window.descriptor_gradient += begin * 70;
    window.moment_gradient += begin * 42;
    window.clear_padding = false;
    return window;
  };
  auto force_window = [&](long begin, long count) {
    Arguments window = force_arguments;
    window.node_begin = begin;
    window.node_count = count;
    window.clear_padding = false;
    return window;
  };

  long previous = 0;
  for (int w = 0; w < windows; ++w) {
    long end = (w + 1 == windows)
        ? total
        : ((total * static_cast<long>(w + 1) / windows) / 16) * 16;
    if (end <= previous) std::abort();
    const long count = end - previous;
    if (forward_windows) chain_fit_node::wait_forward_window(w, stream);
    Arguments nw = node_window(previous, count);
    if(medium) chain_fit_node::launch_fused_range(fit,node,previous,count,stream);
    else {
      fitting_pipeline::launch_range(fit, previous, count, stream, fit_blocks);
      chain_small_node::launch_node(nw, stream);
    }
    check_cuda(cudaEventRecord(ready[w], stream));
    check_cuda(cudaStreamWaitEvent(side, ready[w], 0));
    Arguments fw = force_window(previous, count);
    if (!chain_virial_policy::launch_window(fw, view, side, virial, w == 0, total,
                                   force_blocks)) std::abort();
    previous = end;
  }
  if (!chain_virial_policy::unpack_window_force(total, force, side)) std::abort();
  check_cuda(cudaEventRecord(done, side));
  check_cuda(cudaStreamWaitEvent(stream, done, 0));
  if (forward_windows) {
    chain_fit_node::wait_forward_complete(stream);
    chain_fit_node::consume_forward_pipeline(stream);
  }
  static bool announced = false;
  if (!announced) {
    announced = true;
    std::fprintf(stderr,
                 "%s nodes=%ld candidate_slots_per_node=%ld "
                 "density_threshold=%d windows=%d fit_blocks=%d force_blocks=%d "
                 "exact_multistream_graph_ready forward_pipe=%d\n",
                 medium?"FIT_NODE_PIPELINE_FIT_NODE_WINDOW_PIPE":"EXECUTION_WINDOWS_MULTIWINDOW_PIPE",
                 total, candidate_slots_per_node, density_threshold, windows,
                 fit_blocks, force_blocks, forward_windows ? 1 : 0);
  }
  return true;
}
}  // namespace chain_execution_windows
}  // namespace deepmd_dpa4c
