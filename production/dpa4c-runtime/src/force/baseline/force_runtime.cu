#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include "operator_fallback.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
namespace deepmd_dpa4c {
namespace chain_persistent_force {

#include "raw_force_split.cuh"
#include "shared_table_refine.cuh"
static unsigned long long* persistent_force_buffer(long n,cudaStream_t stream) {
  static unsigned long long* buf=nullptr;static long capacity=0;
  if(n>capacity) {
    cudaStreamCaptureStatus status;cudaStreamIsCapturing(stream,&status);
    if(status!=cudaStreamCaptureStatusNone)return nullptr;
    unsigned long long* next=nullptr;
    if(cudaMalloc(&next,(n+n/4+64)*16)!=cudaSuccess)return nullptr;
    // Captured graphs may retain old buffers; keep addresses alive.
    buf=next;capacity=n+n/4+64;
  }
  return buf;
}
void launch_raw_force_c8(const Arguments&a,const DpaDirectNeighborView&view,cudaStream_t stream,float*f,float*v) {

  static const int policy=getenv("PERSISTENT_FORCE_POLICY")?atoi(getenv("PERSISTENT_FORCE_POLICY")):-1;
  static const long min_nodes=getenv("PERSISTENT_FORCE_MIN_NODES")?std::max(4096L,atol(getenv("PERSISTENT_FORCE_MIN_NODES"))):65536L;
  if(a.node_count<min_nodes||a.has_spin||a.lmax!=2||a.table_width!=8||a.interval_count!=600||a.radial_modes!=0||a.node_begin||
     !getenv("DPA_FORCE_PACK")||atoi(getenv("DPA_FORCE_PACK"))!=1||!getenv("DPA_FORCE_ONESIDED")||atoi(getenv("DPA_FORCE_ONESIDED"))!=2) {deepmd_dpa4c::operator_fallback_force(a,view,stream,f,v);return;}
  static int grid=0,setup=0;
  if(!setup) {
    cudaStreamCaptureStatus status;cudaStreamIsCapturing(stream,&status);
    if(status!=cudaStreamCaptureStatusNone){deepmd_dpa4c::operator_fallback_force(a,view,stream,f,v);return;}
    cudaDeviceProp prop{};int device=0;cudaGetDevice(&device);cudaGetDeviceProperties(&prop,device);grid=prop.multiProcessorCount*2;
    auto e0=cudaFuncSetAttribute(persistent_force_persistent_force<24,2,0,0>,cudaFuncAttributeMaxDynamicSharedMemorySize,124800);
    auto e1=cudaFuncSetAttribute(persistent_force_persistent_force<24,2,0,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,124800);
    setup=(e0==cudaSuccess&&e1==cudaSuccess)?1:-1;
    if(setup<0)cudaGetLastError();
  }
  if(setup<0){deepmd_dpa4c::operator_fallback_force(a,view,stream,f,v);return;}
  auto*fp=persistent_force_buffer(a.node_count,stream);if(!fp){deepmd_dpa4c::operator_fallback_force(a,view,stream,f,v);return;}
  const int traversal=policy<0?(a.node_count>=65536):policy;
  static bool logged=false;
  if(!logged){logged=true;fprintf(stderr,"PERSISTENT_FORCE_FORCE active nodes=%ld traversal=%d warps=24 grid=%d shared=124800 FP32_exact_table\n",a.node_count,traversal,grid);}
  cudaMemsetAsync(fp,0,a.node_count*16,stream);
  if(traversal)persistent_force_persistent_force<24,2,0,1><<<grid,768,124800,stream>>>(a,view,fp,v,nullptr);
  else persistent_force_persistent_force<24,2,0,0><<<grid,768,124800,stream>>>(a,view,fp,v,nullptr);
  unpack_force_kernel<<<std::min<long>((a.node_count+255)/256,4096),256,0,stream>>>(a.node_count,fp,f);
}
}
}  // namespace chain_persistent_force
