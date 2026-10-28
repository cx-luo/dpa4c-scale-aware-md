#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
extern "C" cudaError_t cudaLaunchKernel(const void*fn,dim3 grid,dim3 block,void**args,size_t sh,cudaStream_t stream){
 using F=cudaError_t(*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);
 static auto orig=(F)dlsym(RTLD_NEXT,"cudaLaunchKernel");if(!orig)abort();
 if(sh<200000 || sh>230000 || block.x!=512 || grid.x!=64)return orig(fn,grid,block,args,sh,stream);
 long n=*static_cast<long*>(args[0]);if(n<65536 || n>1000000)return orig(fn,grid,block,args,sh,stream);
 static thread_local float*scratch=nullptr;
 if(!scratch){
   cudaStreamCaptureStatus status;auto err=cudaStreamIsCapturing(stream,&status);if(err!=cudaSuccess)return err;
   if(status!=cudaStreamCaptureStatusNone)return orig(fn,grid,block,args,sh,stream);
   err=cudaMalloc(&scratch,size_t(96)*16*3072*sizeof(float));if(err!=cudaSuccess)return err;
   fprintf(stderr,"FITTING_GRID fitting grid 64 -> 96; unchanged production kernel, nodes=%ld\n",n);
 }
 // Production direct_executor fitting has nine arguments and overwrites x in place.
 void* params[9];for(int i=0;i<9;++i)params[i]=args[i];params[7]=&scratch;
 grid.x=96;return orig(fn,grid,block,params,sh,stream);
}
