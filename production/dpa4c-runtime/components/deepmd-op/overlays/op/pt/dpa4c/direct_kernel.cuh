// C8/L2 single-hop exact VJP; same spline, FiLM, masses and angular basis.
#include <cstdlib>
namespace deepmd_dpa4c {
__device__ __forceinline__ void direct_vjp(
    const Arguments& a, const EdgeGeometry& geom, const float* q, long stride,
    int center_type, const float2 (&radial)[8], float (&result)[3]) {
  using P=Profile<8,2,false>;
  float basis[9];fill_angular_basis(geom,basis);
  float db[9]={},direction[3]={};
  float rg=0.0f,eg=0.0f;
  const auto* film=reinterpret_cast<const float2*>(a.pair_film)+
      (long(center_type)*a.type_count+geom.source_type)*8;
  #pragma unroll
  for(int c=0;c<8;++c) {
    const float2 f=__ldg(film+c);
    const float value=fmaf(f.x,radial[c].x,f.y);
    const float derivative=f.x*radial[c].y;
    float angular=0.0f;
    if(c<4) {
      #pragma unroll
      for(int d=0;d<3;++d) angular=fmaf(q[(P::VectorOffset+d*4+c)*stride],basis[1+d],angular);
      #pragma unroll
      for(int d=0;d<5;++d) angular=fmaf(q[(P::TensorOffset+d*4+c)*stride],basis[4+d],angular);
    }
    const float scaled=geom.envelope*angular;
    const float fg=q[(P::ScalarOffset+c)*stride]+scaled;
    rg=fmaf(geom.envelope*fg,derivative,rg);
    eg=fmaf(fg+scaled,value,eg);
    const float ap=value*geom.envelope*geom.envelope;
    if(c<4) {
      #pragma unroll
      for(int d=0;d<3;++d) db[1+d]=fmaf(q[(P::VectorOffset+d*4+c)*stride],ap,db[1+d]);
      #pragma unroll
      for(int d=0;d<5;++d) db[4+d]=fmaf(q[(P::TensorOffset+d*4+c)*stride],ap,db[4+d]);
    }
  }
  const float squared=geom.envelope*geom.envelope;
  eg=fmaf(4.0f*squared*geom.envelope,q[(P::MomentWidth+1)*stride],
      fmaf(2.0f*geom.envelope,q[P::MomentWidth*stride],eg));
  rg=fmaf(eg,c3_envelope_derivative(geom.radius,a.rcut),rg);
  basis_vjp(geom,db,direction,rg,result);
}

template<int Warps> __global__ void direct_force_kernel(Arguments a,float* f,float* v) {
  using P=Profile<8,2,false>;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  const long i=long(blockIdx.x)*Warps+warp;
  __shared__ float own[Warps][P::StateWidth];
  if(i<a.node_count) for(int k=lane;k<P::StateWidth;k+=32) own[warp][k]=a.moment_gradient[i*P::StateWidth+k];
  __syncthreads();
  if(i>=a.node_count)return;
  float force[3]={},virial[9]={};
  const auto* sources=static_cast<const std::uint32_t*>(a.edge_index);
  const int ti=int(a.atype[i]);
  for(long e=a.destination_row_ptr[i]+lane;e<a.destination_row_ptr[i+1];e+=32) {
    const std::uint32_t j=sources[e];
    if(j==0xffffffffu || (a.edge_mask && !a.edge_mask[e]))continue;
    auto g=load_geometry<true,std::uint32_t>(e,a.rcut,a.eps,a.edge_vec,sources,a.atype);
    const auto loc=locate_table<true>(g.radius,a.table_stride,a.table_max,a.interval_count);
    const auto row=table_row(a.table,loc,a.table_width);
    float2 radial[8];
    #pragma unroll
    for(int c=0;c<8;++c)radial[c]=evaluate_table_with_derivative(row,c,loc.coordinate,loc.clamped);
    float gij[3],gji[3];
    direct_vjp(a,g,own[warp],1,ti,radial,gij);
    const int tj=g.source_type;
    g.ux=-g.ux;g.uy=-g.uy;g.uz=-g.uz;g.source_type=ti;
    direct_vjp(a,g,a.moment_gradient+long(j)*P::StateWidth,1,tj,radial,gji);
    #pragma unroll
    for(int c=0;c<3;++c) {
      force[c]+=gij[c]-gji[c];
      #pragma unroll
      for(int d=0;d<3;++d)virial[c*3+d]+=gji[c]*a.edge_vec[e*3+d];
    }
  }
  #pragma unroll
  for(int offset=16;offset;offset>>=1) {
    #pragma unroll
    for(int c=0;c<3;++c)force[c]+=__shfl_down_sync(0xffffffffu,force[c],offset);
    #pragma unroll
    for(int c=0;c<9;++c)virial[c]+=__shfl_down_sync(0xffffffffu,virial[c],offset);
  }
  if(lane==0) {
    #pragma unroll
    for(int c=0;c<3;++c)f[i*3+c]=force[c];
    #pragma unroll
    for(int c=0;c<9;++c)v[i*9+c]=virial[c];
  }
}
void operator_fallback_node(const Arguments& a,cudaStream_t s) {
  using L=BackwardLauncher<8,2,false,false,true,std::uint32_t>;
  // K1 = C2 = 4 lanes carry the probe stages, so four is the narrowest legal width.
  static const int lanes=[]{const char* e=std::getenv("DPA_NODE_LANES");return e&&e[0]?std::atoi(e):0;}();
  static const long lanes_min=[]{const char* e=std::getenv("DPA_NODE_LANES_MIN");return e&&e[0]?std::atol(e):16384L;}();
  if(lanes==4 && a.node_count>=lanes_min){L::launch_node_backward<4>(a,s);return;}
  if(L::prefer_wide_nodes())L::launch_node_backward<kNodeLanesWide>(a,s);
  else L::launch_node_backward<kNodeLanesNarrow>(a,s);
}
#include "direct_force_onesided.cuh"
void launch_direct_force_c8(const Arguments& a,cudaStream_t s,float* f,float* v) {
 const char* one=std::getenv("DPA_FORCE_ONESIDED");
 if(one && one[0]=='1'){
   cudaMemsetAsync(f,0,a.node_count*3*sizeof(float),s);
   direct_force_onesided_kernel<4><<<(a.node_count+3)/4,128,0,s>>>(a,f,v);
   return;
 }
  const char* w=std::getenv("DPA_DIRECT_WARPS");
  if(w && w[0]=='1')direct_force_kernel<1><<<a.node_count,32,0,s>>>(a,f,v);
  else if(w && w[0]=='2')direct_force_kernel<2><<<(a.node_count+1)/2,64,0,s>>>(a,f,v);
  else if(w && w[0]=='8')direct_force_kernel<8><<<(a.node_count+7)/8,256,0,s>>>(a,f,v);
  else direct_force_kernel<4><<<(a.node_count+3)/4,128,0,s>>>(a,f,v);
}
}
