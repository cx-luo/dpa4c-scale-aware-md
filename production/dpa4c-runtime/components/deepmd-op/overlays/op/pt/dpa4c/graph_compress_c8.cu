#include "graph_compress_kernel.cuh"
#include "direct_kernel.cuh"
#include "raw_kernel.cuh"
extern "C" int deepmd_dpa4c_candidate_mask_abi() { return 1; }
namespace deepmd_dpa4c { DPA4C_DEFINE_CHANNEL(8) }
