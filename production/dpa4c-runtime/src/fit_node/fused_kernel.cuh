// Included inside k3n, after the unchanged fitting primitives.
namespace chain_fit_node {
#include "node_body.cuh"
template<bool SharedScratch, bool StoreGradient, int GradientWidth, bool PhysicalGradientOnly = false, bool SharedLayer2 = false>
__global__ __launch_bounds__(32, 1)
void fitting_node_fused_kernel(deepmd_dpa4c::chain_fit_node::FitPayload f,
                               deepmd_dpa4c::Arguments a) {
  static_assert(GradientWidth == 62 || GradientWidth == 72);
  static_assert(!PhysicalGradientOnly || !StoreGradient);
  constexpr bool shared_l2=SharedLayer2 && SharedScratch;
  __shared__ float gradient_storage[shared_l2 ? 1 : 16 * GradientWidth];
  __shared__ float derivative[SharedScratch ? (shared_l2 ? 3 * 48 * 32 : kScratchPerWarp) : 1];
  // Layer-2 derivative storage is dead before node VJP consumes the gradient.
  float* tile_gradient=shared_l2 ? derivative : gradient_storage;
  const unsigned char* image = reinterpret_cast<const unsigned char*>(f.image);
  const float* HD = reinterpret_cast<const float*>(image + kFragArea) + 3*kH;
  const int lane=threadIdx.x;
  Ctx c{image,HD,HD+100,SharedScratch ? derivative : f.scratch+long(blockIdx.x)*kScratchPerWarp,
        0,lane,lane>>2,lane&3};
  const long r0=long(blockIdx.x)*16+c.gr,r1=r0+8;
  {
    float2 xq[5][4];float h[6][8],d2[6][8],p[2]={0.f,0.f};
    load_x(xq,f.x,r0,r1,f.n,c.tq);
    forward<0>(c,h,d2,p,xq);forward<1>(c,h,d2,p,xq);
    forward<2,0,shared_l2>(c,h,d2,p,xq);forward<2,1,shared_l2>(c,h,d2,p,xq);
    float p0=p[0],p1=p[1];
    p0+=__shfl_xor_sync(0xffffffffu,p0,1);p0+=__shfl_xor_sync(0xffffffffu,p0,2);
    p1+=__shfl_xor_sync(0xffffffffu,p1,1);p1+=__shfl_xor_sync(0xffffffffu,p1,2);
    const float hb=f.head_bias?*f.head_bias:0.f;
    if(c.tq==0){
      if(r0<f.n)f.energy[r0]=double(p0+hb)+f.atom_bias[f.types[r0]];
      if(r1<f.n)f.energy[r1]=double(p1+hb)+f.atom_bias[f.types[r1]];
    }
    backward<2,shared_l2>(c,h,d2);backward<1>(c,h,d2);backward<0>(c,h,d2);
    if constexpr(shared_l2)__syncwarp();
    const float s0=r0<f.n?float(f.seed[r0]):0.f,s1=r1<f.n?float(f.seed[r1]):0.f;
#pragma unroll
    for(int nb=0;nb<5;++nb){
#pragma unroll
      for(int j=0;j<4;++j){
        const int k=nb*16+c.tq+4*j;
        if(k<kIn){
          const float g0=h[nb][j]*s0,g1=h[nb][4+j]*s1;
          // Features 62..69 are fixed type embeddings. node_vjp uses only
          // 0..61, as its existing GradientWidth=62 route also guarantees.
          // Keep the full 70-feature energy forward and every used derivative.
          if constexpr(PhysicalGradientOnly){
            if(k<62) {
              tile_gradient[c.gr*GradientWidth+k]=g0;
              tile_gradient[(c.gr+8)*GradientWidth+k]=g1;
            }
          }else if constexpr(GradientWidth == 72){
            tile_gradient[c.gr*GradientWidth+k]=g0;
            tile_gradient[(c.gr+8)*GradientWidth+k]=g1;
          }else if(k<GradientWidth){
            tile_gradient[c.gr*GradientWidth+k]=g0;
            tile_gradient[(c.gr+8)*GradientWidth+k]=g1;
          }
          if constexpr(StoreGradient){if(r0<f.n)f.x[r0*kIn+k]=g0;if(r1<f.n)f.x[r1*kIn+k]=g1;}
        }
      }
    }
  }
  __syncwarp();
  const long node=long(blockIdx.x)*16+lane;
  if(lane<16 && node<a.node_count){
    deepmd_dpa4c::chain_packed_nvt::preclear_node(node);
    node_vjp(a,node,tile_gradient+lane*GradientWidth);
  }
}


// Medium narrow-NVT specialization.  Keep the production fused kernel above
// unchanged, materialize only the 62 physical descriptor gradients, and let a
// separate 128-thread consumer perform node_vjp.
__global__ __launch_bounds__(32, 1)
void fitting_node_outlined_kernel(
    deepmd_dpa4c::chain_fit_node::FitPayload f,
    deepmd_dpa4c::Arguments a) {
  __shared__ float derivative[kScratchPerWarp];
  const unsigned char* image = reinterpret_cast<const unsigned char*>(f.image);
  const float* HD = reinterpret_cast<const float*>(image + kFragArea) + 3*kH;
  const int lane=threadIdx.x;
  Ctx c{image,HD,HD+100,derivative,0,lane,lane>>2,lane&3};
  const long r0=long(blockIdx.x)*16+c.gr,r1=r0+8;
  float2 xq[5][4];float h[6][8],d2[6][8],p[2]={0.f,0.f};
  load_x(xq,f.x,r0,r1,f.n,c.tq);
  forward<0>(c,h,d2,p,xq);forward<1>(c,h,d2,p,xq);
  forward<2,0,false>(c,h,d2,p,xq);forward<2,1,false>(c,h,d2,p,xq);
  float p0=p[0],p1=p[1];
  p0+=__shfl_xor_sync(0xffffffffu,p0,1);p0+=__shfl_xor_sync(0xffffffffu,p0,2);
  p1+=__shfl_xor_sync(0xffffffffu,p1,1);p1+=__shfl_xor_sync(0xffffffffu,p1,2);
  const float hb=f.head_bias?*f.head_bias:0.f;
  if(c.tq==0){
    if(r0<f.n)f.energy[r0]=double(p0+hb)+f.atom_bias[f.types[r0]];
    if(r1<f.n)f.energy[r1]=double(p1+hb)+f.atom_bias[f.types[r1]];
  }
  backward<2,false>(c,h,d2);backward<1>(c,h,d2);backward<0>(c,h,d2);
  const float s0=r0<f.n?float(f.seed[r0]):0.f,s1=r1<f.n?float(f.seed[r1]):0.f;
#pragma unroll
  for(int nb=0;nb<4;++nb){
#pragma unroll
    for(int j=0;j<4;++j){
      const int k=nb*16+c.tq+4*j;
      if(k<62){
        const float g0=h[nb][j]*s0,g1=h[nb][4+j]*s1;
        if(r0<f.n)f.x[r0*kIn+k]=g0;
        if(r1<f.n)f.x[r1*kIn+k]=g1;
      }
    }
  }
}

__global__ void outlined_node_vjp_kernel(deepmd_dpa4c::Arguments a) {
  const long node=long(blockIdx.x)*blockDim.x+threadIdx.x;
  if(node<a.node_count){
    deepmd_dpa4c::chain_packed_nvt::preclear_node(node);
    node_vjp(a,node,a.descriptor_gradient+node*70);
  }
}
} // namespace chain_fit_node
