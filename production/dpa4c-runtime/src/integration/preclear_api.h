#pragma once
#include <cuda_runtime.h>

namespace deepmd_dpa4c { namespace chain_packed_nvt {
cudaError_t configure_preclear(unsigned long long* packed, long nodes);
}}
