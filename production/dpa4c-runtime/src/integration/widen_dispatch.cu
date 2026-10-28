#include <cuda_runtime.h>
#include <cstddef>
namespace deepmd_dpa4c {
namespace chain_reclaimed_widen { bool dispatch_widen(long,long,const float*,const float*,double*,double*,dim3,dim3,std::size_t,cudaStream_t); }
namespace chain_packed_widen { bool dispatch_widen(long,long,const float*,const float*,double*,double*,dim3,dim3,std::size_t,cudaStream_t); }
}
extern "C" bool dpa_chain_try_widen(long nf,long nt,const float* f,const float* v,double* fo,double* vo,int blocks,int threads,cudaStream_t stream) {
  using namespace deepmd_dpa4c;
  if(chain_reclaimed_widen::dispatch_widen(nf,nt,f,v,fo,vo,dim3(blocks),dim3(threads),0,stream))return true;
  return chain_packed_widen::dispatch_widen(nf,nt,f,v,fo,vo,dim3(blocks),dim3(threads),0,stream);
}
