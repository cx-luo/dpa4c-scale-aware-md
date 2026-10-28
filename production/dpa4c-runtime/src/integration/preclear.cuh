#pragma once
#include "preclear_api.h"

namespace deepmd_dpa4c { namespace chain_packed_nvt {
__device__ unsigned long long* g_preclear_packed = nullptr;
__device__ long g_preclear_nodes = 0;

cudaError_t configure_preclear(unsigned long long* packed, long nodes) {
  cudaError_t status = cudaMemcpyToSymbol(
      g_preclear_packed, &packed, sizeof(packed), 0, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) return status;
  return cudaMemcpyToSymbol(
      g_preclear_nodes, &nodes, sizeof(nodes), 0, cudaMemcpyHostToDevice);
}

__device__ __forceinline__ void preclear_node(long node) {
  unsigned long long* packed = g_preclear_packed;
  if (packed && node < g_preclear_nodes) {
    packed[2 * node] = 0ull;
    packed[2 * node + 1] = 0ull;
  }
}
}}
