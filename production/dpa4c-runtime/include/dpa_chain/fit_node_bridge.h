#pragma once
#include "graph_compress_launch.h"
namespace deepmd_dpa4c { namespace chain_fit_node {
struct FitPayload {
  long n;
  float* x;
  const long* types;
  const double* seed;
  const double* atom_bias;
  const float* head_bias;
  const uint4* image;
  float* scratch;
  double* energy;
};
void begin_forward(const Arguments&, cudaStream_t);
bool queue_fit(const FitPayload&, cudaStream_t);
bool launch_node(const Arguments&, cudaStream_t);
bool take_large_pipeline(FitPayload&, Arguments&, cudaStream_t);
void launch_fused_range(const FitPayload&, const Arguments&, long, long, cudaStream_t);
bool wants_forward_pipeline(const Arguments&, cudaStream_t);
void set_forward_pipeline(int, const cudaEvent_t*, cudaEvent_t, cudaStream_t);
int forward_pipeline_windows(cudaStream_t);
void wait_forward_window(int, cudaStream_t);
void wait_forward_complete(cudaStream_t);
void consume_forward_pipeline(cudaStream_t);
}}
struct DpaDirectNeighborView;
namespace deepmd_dpa4c { namespace chain_consumer_pipeline {
bool launch_force(const Arguments&, const DpaDirectNeighborView&,
                  cudaStream_t, float*, float*);
}}

namespace deepmd_dpa4c { namespace chain_execution_windows {
bool launch_force(const Arguments&, const DpaDirectNeighborView&,
                  cudaStream_t, float*, float*);
}}
