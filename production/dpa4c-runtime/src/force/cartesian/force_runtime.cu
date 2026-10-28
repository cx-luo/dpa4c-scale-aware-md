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
namespace chain_factored_force {

#include "raw_force_split.cuh"
#include "shared_table_refine.cuh"
#include "radial_math.cuh"
#include "algebra_force.cuh"
static unsigned long long* factored_force_buffer(long n,cudaStream_t stream) {
  static unsigned long long* buf=nullptr;static long capacity=0;
  if(n>capacity){
    cudaStreamCaptureStatus capture;cudaStreamIsCapturing(stream,&capture);
    if(capture!=cudaStreamCaptureStatusNone)return nullptr;
    unsigned long long* next=nullptr;
    if(cudaMalloc(&next,(n+n/4+64)*16)!=cudaSuccess)return nullptr;
    // Existing captured graphs may still reference a prior allocation.
    buf=next;capacity=n+n/4+64;
  }
  return buf;
}
void launch_raw_force_c8(const Arguments&a,const DpaDirectNeighborView&view,cudaStream_t stream,float*f,float*v){
  static const bool factored=getenv("FACTORED_FORCE_FACTORED")&&atoi(getenv("FACTORED_FORCE_FACTORED"));
  static const long min_nodes=getenv("FACTORED_FORCE_MIN_NODES")?std::max(4096L,atol(getenv("FACTORED_FORCE_MIN_NODES"))):4096L;
  if(a.node_count<min_nodes||a.has_spin||a.lmax!=2||a.table_width!=8||a.interval_count!=600||a.radial_modes!=0||a.node_begin||
     !getenv("DPA_FORCE_PACK")||atoi(getenv("DPA_FORCE_PACK"))!=1||!getenv("DPA_FORCE_ONESIDED")||atoi(getenv("DPA_FORCE_ONESIDED"))!=2){chain_persistent_force::launch_raw_force_c8(a,view,stream,f,v);return;}
  static int setup=0,grid=0;
  if(!setup){
    cudaStreamCaptureStatus capture;cudaStreamIsCapturing(stream,&capture);
    if(capture!=cudaStreamCaptureStatusNone){chain_persistent_force::launch_raw_force_c8(a,view,stream,f,v);return;}
    cudaDeviceProp prop{};int dev=0;cudaGetDevice(&dev);cudaGetDeviceProperties(&prop,dev);grid=prop.multiProcessorCount*2;
    auto e0=cudaFuncSetAttribute(factored_force_force<16,1,2,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,124800);
    auto e1=cudaFuncSetAttribute(factored_force_force<16,1,2,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,124800);
    setup=e0==cudaSuccess&&e1==cudaSuccess?1:-1;
    if(setup<0)cudaGetLastError();
  }
  if(setup<0){chain_persistent_force::launch_raw_force_c8(a,view,stream,f,v);return;}
  auto*packed=factored_force_buffer(a.node_count,stream);if(!packed){chain_persistent_force::launch_raw_force_c8(a,view,stream,f,v);return;}
  static bool logged=false;
  if(!logged){logged=true;fprintf(stderr,"FACTORED_FORCE_FORCE active nodes=%ld algorithm=quadratic lanes=1 warps=16 blocks=%d factored=%d shared=124800 FP32\n",a.node_count,grid,int(factored));}
  cudaMemsetAsync(packed,0,a.node_count*16,stream);
  if(factored)factored_force_force<16,1,2,true><<<grid,512,124800,stream>>>(a,view,packed,v,nullptr);
  else factored_force_force<16,1,2,false><<<grid,512,124800,stream>>>(a,view,packed,v,nullptr);
  unpack_force_kernel<<<std::min<long>((a.node_count+255)/256,4096),256,0,stream>>>(a.node_count,packed,f);
}
}
}  // namespace chain_factored_force
