#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

#include "fit_node_bridge.h"
#include "preclear.cuh"
#define K3N_DEVICE_ONLY 1
#define FITTING_IMAGE_PRODUCTS 3
#define FITTING_IMAGE_KPERIOD 2
#define SPLIT_FITTING_LOAD64 1
#include "fitting_core.cuh"
namespace k3n {
#include "fused_kernel.cuh"
}
#include "host_runtime.cuh"
