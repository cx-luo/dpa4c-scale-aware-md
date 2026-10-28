#include "graph_compress.cuh"
#include "raw_geometry.cuh"
#include "operator_fallback.h"

#include <cstdio>
#include <cstdlib>

namespace deepmd_dpa4c { namespace node_fallback {
#include "generated_large.cuh"
void launch(const Arguments& a,cudaStream_t stream) {
  if(a.node_count>=4096 && !a.has_spin && a.lmax==2 && a.table_width==8 && a.radial_modes==0) {
    generated_node<<<(a.node_count+127)/128,128,0,stream>>>(a);
  } else operator_fallback_node(a,stream);
}
}}
namespace deepmd_dpa4c { namespace chain_small_node {
#include "node_generated.cuh"
void launch_node(const Arguments& a, cudaStream_t stream) {
  static const bool enabled = [] {const char* s=std::getenv("DPA_SMALL_NODE_SMALL_NODE"); return s&&s[0]=='1';}();

  if(!enabled || a.node_count<=0 || a.node_count>4096 || a.node_begin!=0 ||
      a.has_spin || a.lmax!=2 || a.table_width!=8 || a.radial_modes!=0) {
    deepmd_dpa4c::node_fallback::launch(a,stream);return;
  }
  static const int threads=[] {const char* s=std::getenv("DPA_SMALL_NODE_NODE_THREADS");int n=s?std::atoi(s):32;return n==32||n==64||n==128?n:32;}();
  generated_node<<<(a.node_count+threads-1)/threads,threads,0,stream>>>(a);
  if(cudaGetLastError()!=cudaSuccess) std::abort();
  static bool shown=false;
  if(!shown) {cudaFuncAttributes attr{};int resident=0;cudaFuncGetAttributes(&attr,generated_node);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,generated_node,threads,0);
    std::fprintf(stderr,"SMALL_NODE_SMALL_NODE nodes=%ld threads=%d regs=%d local=%zu shared=%zu resident=%d\n",a.node_count,threads,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);shown=true;}
}
}}
