
template<int Layout>
__device__ __forceinline__ float2 persistent_force_radial(const float* original,const float* cache,int index,int channel,float x,bool clamped) {
  float4 low; float2 high;
  if constexpr(Layout<0) {
    TableLocation loc{index,x,clamped};return evaluate_table_with_derivative(table_row(original,loc,8),channel,x,clamped);
  } else if constexpr(Layout==0) {
    const float* p=cache+index*52;
    low=reinterpret_cast<const float4*>(p)[channel];high=reinterpret_cast<const float2*>(p+32)[channel];
  } else {
    const float* p=cache+index*49+channel;
    low=make_float4(p[0],p[8],p[16],p[24]);high=make_float2(p[32],p[40]);
  }
  const float value=low.x+(low.y+(low.z+(low.w+(high.x+high.y*x)*x)*x)*x)*x;
  const float derivative=low.y+(2.f*low.z+(3.f*low.w+(4.f*high.x+5.f*high.y*x)*x)*x)*x;
  return make_float2(value,clamped?0.f:derivative);
}
template <int E, int Layout>
__device__ __forceinline__ void persistent_force_pair_vjp(const Arguments& a, EdgeGeometry g, const float* qi, int ti, int tj,
                                               int base, bool valid, float (&gij)[3], const float* shared_table) {
  using P = Profile<8, 2, false>;
  float basis[9];
  fill_angular_basis(g, basis);
  float dbi[9] = {}, direction[3] = {};
  float rgi = 0.f, egi = 0.f;
  if (valid) {
    const auto loc = locate_table<true>(g.radius, a.table_stride, a.table_max, a.interval_count);

    const auto* fi = reinterpret_cast<const float2*>(a.pair_film) + (long(ti) * a.type_count + tj) * 8;
#pragma unroll
    for (int tile = 0; tile < 8 / E; ++tile) {
      const int c = base + tile * E;
      const float2 radial = persistent_force_radial<Layout>(a.table, shared_table, loc.index, c, loc.coordinate, loc.clamped);
      const float2 wi = __ldg(fi + c);
      const float vi = fmaf(wi.x, radial.x, wi.y);
      float ai = 0.f;
      if (c < 4) {
#pragma unroll
        for (int d = 0; d < 3; ++d) ai = fmaf(qi[P::VectorOffset + d * 4 + c], basis[1 + d], ai);
#pragma unroll
        for (int d = 0; d < 5; ++d) ai = fmaf(qi[P::TensorOffset + d * 4 + c], basis[4 + d], ai);
      }
      const float si = g.envelope * ai, fg_i = qi[P::ScalarOffset + c] + si;
      rgi = fmaf(g.envelope * fg_i, wi.x * radial.y, rgi);
      egi = fmaf(fg_i + si, vi, egi);
      const float api = vi * g.envelope * g.envelope;
      if (c < 4) {
#pragma unroll
        for (int d = 0; d < 3; ++d) dbi[1 + d] = fmaf(qi[P::VectorOffset + d * 4 + c], api, dbi[1 + d]);
#pragma unroll
        for (int d = 0; d < 5; ++d) dbi[4 + d] = fmaf(qi[P::TensorOffset + d * 4 + c], api, dbi[4 + d]);
      }
    }
  }
  rgi = group_sum<E>(rgi);
  egi = group_sum<E>(egi);
#pragma unroll
  for (int d = 1; d < 9; ++d) dbi[d] = group_sum<E>(dbi[d]);
  const float square = g.envelope * g.envelope;
  egi = fmaf(4.f * square * g.envelope, qi[P::MomentWidth + 1], fmaf(2.f * g.envelope, qi[P::MomentWidth], egi));
  const float de = c3_envelope_derivative(g.radius, a.rcut);
  rgi = fmaf(egi, de, rgi);
  basis_vjp(g, dbi, direction, rgi, gij);
}


template<int Warps,int E,int Layout,int Traversal>
__global__ void persistent_force_persistent_force(Arguments a,DpaDirectNeighborView view,unsigned long long* fp,float* v,const float* transposed) {
  using P=Profile<8,2,false>;
  constexpr int Groups=32/E;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,base=lane%E,group=lane/E;
  extern __shared__ float shared_table[];
  __shared__ float own[Warps][P::StateWidth];
  const float* cache=Layout==2?transposed:shared_table;
  if constexpr(Layout==0) {
    for(int k=threadIdx.x;k<600*48;k+=blockDim.x)shared_table[(k/48)*52+k%48]=a.table[k];
  } else if constexpr(Layout==1) {
    for(int k=threadIdx.x;k<600*48;k+=blockDim.x) {
      int idx=k/48,rem=k%48,c=rem/6,coef=rem%6;
      int old=coef<4?4*c+coef:32+2*c+coef-4;
      shared_table[idx*49+coef*8+c]=a.table[idx*48+old];
    }
  }
  __syncthreads();
  const long tiles=(a.node_count+Warps-1)/Warps;
  const long first=Traversal ? (tiles*blockIdx.x/gridDim.x)*Warps : long(blockIdx.x)*Warps;
  const long stop=Traversal ? min(a.node_count,(tiles*(blockIdx.x+1)/gridDim.x)*Warps) : a.node_count;
  const long stride=Traversal ? Warps : long(gridDim.x)*Warps;
  for(long i=a.node_begin+first+warp;i<a.node_begin+stop;i+=stride) {
    for(int k=lane;k<P::StateWidth;k+=32)own[warp][k]=a.moment_gradient[i*P::StateWidth+k];
    __syncwarp();
  const long begin = a.destination_row_ptr[i];
  const long live = reinterpret_cast<const int*>(a.edge_mask)[i];
  float force[3] = {}, virial[9] = {};
  const int ti = int(a.atype[i]);
  const float4* records = reinterpret_cast<const float4*>(a.edge_vec) + begin;
  for (long start = 0; start < live; start += Groups) {
    const long position = start + group;
    const bool valid = position < live;
    float xyz[3];
    EdgeGeometry g = geometry_from_cache(a, records[valid ? position : 0], xyz);
    float gij[3];
    persistent_force_pair_vjp<E,Layout>(a, g, own[warp], ti, g.source_type, base, valid, gij, cache);
    if (valid && base < 2) pack_add(fp, long(static_cast<std::uint32_t>(g.source)), gij, -1.f, base);
    if (valid && base == 0) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        force[c] += gij[c];
#pragma unroll
        for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * xyz[d];
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
#pragma unroll
    for (int c = 0; c < 9; ++c) virial[c] += __shfl_down_sync(0xffffffffu, virial[c], offset);
  }
  for (int c = 0; c < 3; ++c) force[c] = __shfl_sync(0xffffffffu, force[c], 0);
  if (lane < 2) pack_add(fp, i, force, 1.f, lane);
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < 9; ++c) v[i * 9 + c] = virial[c];
  }

  __syncwarp();
  }
}
