template <int E, bool Factored>
__device__ __forceinline__ void factored_force_pair_early(const Arguments& a, EdgeGeometry g, const float* qi, int ti, int tj,
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
      const float2 radial = persistent_force_radial<0>(a.table, shared_table, loc.index, c, loc.coordinate, loc.clamped);
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
  const float de = Factored ? envelope_derivative_factored(g.radius,a.rcut) : c3_envelope_derivative(g.radius,a.rcut);
  rgi = fmaf(egi, de, rgi);
  basis_vjp(g, dbi, direction, rgi, gij);
}


template <int E, bool Factored>
__device__ __forceinline__ void factored_force_pair_late(const Arguments& a, EdgeGeometry g, const float* qi, int ti, int tj,
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
      const float2 radial = persistent_force_radial<0>(a.table, shared_table, loc.index, c, loc.coordinate, loc.clamped);
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
  const float square = g.envelope * g.envelope;
  if(base==0) egi = fmaf(4.f * square * g.envelope, qi[P::MomentWidth + 1], fmaf(2.f * g.envelope, qi[P::MomentWidth], egi));
  const float de = Factored ? envelope_derivative_factored(g.radius,a.rcut) : c3_envelope_derivative(g.radius,a.rcut);
  rgi = fmaf(egi, de, rgi);
  basis_vjp(g, dbi, direction, rgi, gij);
#pragma unroll
  for(int k=0;k<3;++k)gij[k]=group_sum<E>(gij[k]);
}



// Exact 3+1 split: c0/c1/c2 use 57.6 KB shared memory and the 19.2 KB
// c3 plane uses the read-only cache.  Coefficients and Horner order match HERMITE.
__device__ __forceinline__ float2 hermite_radial(
    const float* cache, const float* c3_plane, int index, int channel,
    float x, bool clamped) {
  const int item = index * 8 + channel;
  const float c0 = cache[item];
  const float c1 = cache[600 * 8 + item];
  const float c2 = cache[2 * 600 * 8 + item];
  const float c3 = __ldg(c3_plane + item);
  const float value = c0 + (c1 + (c2 + c3 * x) * x) * x;
  const float derivative = c1 + (2.f * c2 + 3.f * c3 * x) * x;
  return make_float2(value, clamped ? 0.f : derivative);
}

// Algebraic Lmax=2 VJP using v.u + u^T Q u - q20/2.
// The -q20/2 constant is retained explicitly; no |u|^2 == 1 assumption.
// qi tensor rows 0,1,3,4 have been multiplied by sqrt(3) once per node.
template<int E, bool Factored, bool HoistEnvelope=false>
__device__ __forceinline__ void factored_force_pair_quadratic(
    const Arguments& a, EdgeGeometry g, const float* qi, int ti, int tj,
    int base, bool valid, float (&gij)[3],const float* shared_table) {
  constexpr bool Late=true;
  float bx=0.f,by=0.f,bz=0.f,rg=0.f,eg=0.f;
  const float ux=g.ux,uy=g.uy,uz=g.uz;
  if(valid){
    auto loc=locate_table<true>(g.radius,a.table_stride,a.table_max,a.interval_count);

    auto fi=reinterpret_cast<const float2*>(a.pair_film)+(long(ti)*a.type_count+tj)*8;
    #pragma unroll
    for(int tile=0;tile<8/E;++tile){
      const int c=base+tile*E;
      const float* c3_plane = a.table + 3 * 600 * 8;
      float2 radial=hermite_radial(shared_table,c3_plane,loc.index,c,
                               loc.coordinate,loc.clamped);
      float2 wi=__ldg(fi+c);float phi=fmaf(wi.x,radial.x,wi.y);
      float ang=0.f,dx=0.f,dy=0.f,dz=0.f;
      if(c<4){
        float vx=qi[8+c],vy=qi[12+c],vz=qi[16+c];
        float qxy=qi[20+c],qyz=qi[24+c],q20=qi[28+c],qxz=qi[32+c],qxx=qi[36+c];
        float tx=fmaf(qxx,ux,fmaf(qxy,uy,qxz*uz));
        float ty=fmaf(qxy,ux,fmaf(-qxx,uy,qyz*uz));
        float tz=fmaf(qxz,ux,fmaf(qyz,uy,3.f*q20*uz));
        ang=fmaf(ux,vx+.5f*tx,fmaf(uy,vy+.5f*ty,fmaf(uz,vz+.5f*tz,-.5f*q20)));
        dx=vx+tx;dy=vy+ty;dz=vz+tz;
      }
      float si=g.envelope*ang, fg=qi[c]+si;
      rg=fmaf(HoistEnvelope ? fg : g.envelope*fg,wi.x*radial.y,rg);
      eg=fmaf(fg+si,phi,eg);
      if(c<4){float api=HoistEnvelope ? phi : phi*g.envelope*g.envelope;
        bx=fmaf(api,dx,bx);by=fmaf(api,dy,by);bz=fmaf(api,dz,bz);
      }
    }
  }
  // The scalar primal is e*S(r) + e^2*A(u,r) + q40*e^2 + q41*e^4.
  // Hoist the common e and e^2 from its radial and angular adjoints only
  // in fixed virial consumers. Conditional dual kernels retain their original
  // arithmetic and resource shape. Preserve edge and channel order.
  if constexpr(HoistEnvelope) {
    const float envelope_square=g.envelope*g.envelope;
    bx*=envelope_square;by*=envelope_square;bz*=envelope_square;
    rg*=g.envelope;
  }
  if constexpr(!Late){
    rg=group_sum<E>(rg);eg=group_sum<E>(eg);
    bx=group_sum<E>(bx);by=group_sum<E>(by);bz=group_sum<E>(bz);
  }
  if(!Late || base==0){float sq=g.envelope*g.envelope;
    eg=fmaf(4.f*sq*g.envelope,qi[41],fmaf(2.f*g.envelope,qi[40],eg));
  }
  rg=fmaf(eg,(Factored ? envelope_derivative_factored(g.radius,a.rcut) : c3_envelope_derivative(g.radius,a.rcut)),rg);
  float dot=fmaf(ux,bx,fmaf(uy,by,uz*bz));
  gij[0]=fmaf(bx-ux*dot,g.inverse_radius,rg*ux);
  gij[1]=fmaf(by-uy*dot,g.inverse_radius,rg*uy);
  gij[2]=fmaf(bz-uz*dot,g.inverse_radius,rg*uz);
  if constexpr(Late){
    #pragma unroll
    for(int c=0;c<3;++c)gij[c]=group_sum<E>(gij[c]);
  }
}
__device__ __forceinline__ void physical_gradient_copy_word_async(
    float4* destination, const float4* source) {
  const unsigned shared_address =
      static_cast<unsigned>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void physical_gradient_copy_force_table_async(
    float* destination_float, const float* source_float) {
  auto* destination = reinterpret_cast<float4*>(destination_float);
  const auto* source = reinterpret_cast<const float4*>(source_float);
  constexpr int words = (3 * 600 * 8) / 4;
  constexpr int words_per_group_per_thread = 4;
  for (int base = threadIdx.x; base < words;
       base += blockDim.x * words_per_group_per_thread) {
#pragma unroll
    for (int lane_word = 0; lane_word < words_per_group_per_thread;
         ++lane_word) {
      const int item = base + lane_word * blockDim.x;
      if (item < words)
        physical_gradient_copy_word_async(destination + item, source + item);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
}

template<int Warps,int E,int Algorithm,bool Factored,bool NeedVirial,bool Conditional>
__global__ void hermite_force(Arguments a,DpaDirectNeighborView view,unsigned long long* fp,float* v,const float* transposed,const int* requested_virial) {
  using P=Profile<8,2,false>;
  constexpr int Layout=0,Traversal=1;
  constexpr int Groups=32/E;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,base=lane%E,group=lane/E;
  if constexpr (Conditional) {
    int requested = lane == 0 ? __ldg(requested_virial) : 0;
    requested = __shfl_sync(0xffffffffu, requested, 0);
    if ((requested != 0) != NeedVirial) return;
  }
  extern __shared__ float shared_table[];
  __shared__ float own[Warps][P::StateWidth];
  const float* cache=shared_table;
  physical_gradient_copy_force_table_async(shared_table, transposed);
  const long tiles=(a.node_count+Warps-1)/Warps;
  const long first=Traversal ? (tiles*blockIdx.x/gridDim.x)*Warps : long(blockIdx.x)*Warps;
  const long stop=Traversal ? min(a.node_count,(tiles*(blockIdx.x+1)/gridDim.x)*Warps) : a.node_count;
  const long stride=Traversal ? Warps : long(gridDim.x)*Warps;
  for(long i=a.node_begin+first+warp;i<a.node_begin+stop;i+=stride) {
    for(int k=lane;k<P::StateWidth;k+=32){
      float q=a.moment_gradient[i*P::StateWidth+k];
      if constexpr(Algorithm==2){if(k>=20&&k<40&&(k<28||k>=32))q*=1.7320508075688772935f;}
      own[warp][k]=q;
    }
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
    if constexpr(Algorithm==0)factored_force_pair_early<E,Factored>(a,g,own[warp],ti,g.source_type,base,valid,gij,cache);
    else if constexpr(Algorithm==1)factored_force_pair_late<E,Factored>(a,g,own[warp],ti,g.source_type,base,valid,gij,cache);
    else factored_force_pair_quadratic<E,Factored,NeedVirial && !Conditional>(a,g,own[warp],ti,g.source_type,base,valid,gij,cache);
    if(valid){
      if constexpr(E==1){pack_add(fp,long(static_cast<std::uint32_t>(g.source)),gij,-1.f,0);pack_add(fp,long(static_cast<std::uint32_t>(g.source)),gij,-1.f,1);}
      else if(base<2)pack_add(fp,long(static_cast<std::uint32_t>(g.source)),gij,-1.f,base);
    }
    if (valid && base == 0) {
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        force[c] += gij[c];
#pragma unroll
        if constexpr (NeedVirial) {
          for (int d = 0; d < 3; ++d) virial[c * 3 + d] -= gij[c] * xyz[d];
        }
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
#pragma unroll
    for (int c = 0; c < 3; ++c) force[c] += __shfl_down_sync(0xffffffffu, force[c], offset);
#pragma unroll
    if constexpr (NeedVirial) {
      for (int c = 0; c < 9; ++c) virial[c] += __shfl_down_sync(0xffffffffu, virial[c], offset);
    }
  }
  for (int c = 0; c < 3; ++c) force[c] = __shfl_sync(0xffffffffu, force[c], 0);
  if (lane < 2) pack_add(fp, i, force, 1.f, lane);
  if constexpr (NeedVirial) {
    if (lane == 0) {
#pragma unroll
      for (int c = 0; c < 9; ++c) v[i * 9 + c] = virial[c];
    }
  }

  __syncwarp();
  }
}


// VIRIAL_POLICY NPT path: global LAMMPS pressure consumes only xx, yy, zz, xy, xz, yz.
// Keep those six entries in their original nine-wide slots while removing the
// three unused per-atom centroid components from the hot force kernel.  This
// probe is selected only by an explicit environment flag; the production
// source and per-atom virial path remain unchanged.
template<int Warps,bool Conditional>
__global__ void virial_policy_force6(Arguments a,DpaDirectNeighborView view,
                            unsigned long long* fp,float* v,
                            const float* transposed,
                            const int* requested_virial) {
  using P=Profile<8,2,false>;
  constexpr int Groups=32;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  if constexpr(Conditional) {
    int requested=lane==0?__ldg(requested_virial):0;
    requested=__shfl_sync(0xffffffffu,requested,0);
    if(requested==0)return;
  }
  extern __shared__ float shared_table[];
  __shared__ float own[Warps][P::StateWidth];
  physical_gradient_copy_force_table_async(shared_table,transposed);
  const long tiles=(a.node_count+Warps-1)/Warps;
  const long first=(tiles*blockIdx.x/gridDim.x)*Warps;
  const long stop=min(a.node_count,(tiles*(blockIdx.x+1)/gridDim.x)*Warps);
  for(long i=a.node_begin+first+warp;i<a.node_begin+stop;i+=Warps) {
    for(int k=lane;k<P::StateWidth;k+=32){
      float q=a.moment_gradient[i*P::StateWidth+k];
      if(k>=20&&k<40&&(k<28||k>=32))q*=1.7320508075688772935f;
      own[warp][k]=q;
    }
    __syncwarp();
    const long begin=a.destination_row_ptr[i];
    const long live=reinterpret_cast<const int*>(a.edge_mask)[i];
    float force[3]={},virial[6]={};
    const int ti=int(a.atype[i]);
    const float4* records=reinterpret_cast<const float4*>(a.edge_vec)+begin;
    for(long start=0;start<live;start+=Groups){
      const long position=start+lane;
      const bool valid=position<live;
      float xyz[3];
      EdgeGeometry g=geometry_from_cache(a,records[valid?position:0],xyz);
      float gij[3];
      factored_force_pair_quadratic<1,true,true>(a,g,own[warp],ti,g.source_type,0,valid,gij,shared_table);
      if(valid){
        pack_add(fp,long(static_cast<std::uint32_t>(g.source)),gij,-1.f,0);
        pack_add(fp,long(static_cast<std::uint32_t>(g.source)),gij,-1.f,1);
        force[0]+=gij[0];force[1]+=gij[1];force[2]+=gij[2];
        // Preserve the exact expressions used by the nine-component kernel.
        virial[0]-=gij[0]*xyz[0]; // slot 0: xx
        virial[1]-=gij[1]*xyz[1]; // slot 4: yy
        virial[2]-=gij[2]*xyz[2]; // slot 8: zz
        virial[3]-=gij[1]*xyz[0]; // slot 3: xy
        virial[4]-=gij[2]*xyz[0]; // slot 6: xz
        virial[5]-=gij[2]*xyz[1]; // slot 7: yz
      }
    }
#pragma unroll
    for(int offset=16;offset;offset>>=1){
#pragma unroll
      for(int c=0;c<3;++c)force[c]+=__shfl_down_sync(0xffffffffu,force[c],offset);
#pragma unroll
      for(int c=0;c<6;++c)virial[c]+=__shfl_down_sync(0xffffffffu,virial[c],offset);
    }
    for(int c=0;c<3;++c)force[c]=__shfl_sync(0xffffffffu,force[c],0);
    if(lane<2)pack_add(fp,i,force,1.f,lane);
    if(lane==0){
      v[i*9+0]=virial[0];v[i*9+4]=virial[1];v[i*9+8]=virial[2];
      v[i*9+3]=virial[3];v[i*9+6]=virial[4];v[i*9+7]=virial[5];
    }
    __syncwarp();
  }
}


// VIRIAL_POLICY exact per-atom fallback.  The six-component primary kernel writes the
// entries consumed by global pressure.  If cvflag_atom requests the complete
// centroid tensor, this second kernel recomputes the same edge VJP and writes
// only the three remaining slots.  It exits before table transport unless the
// stream-ordered request flag is exactly 2.
template<int Warps>
__global__ void virial_policy_missing3(Arguments a,DpaDirectNeighborView view,float* v,
                              const float* transposed,
                              const int* requested_virial) {
  using P=Profile<8,2,false>;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  int requested=lane==0?__ldg(requested_virial):0;
  requested=__shfl_sync(0xffffffffu,requested,0);
  if(requested!=2)return;
  extern __shared__ float shared_table[];
  __shared__ float own[Warps][P::StateWidth];
  physical_gradient_copy_force_table_async(shared_table,transposed);
  const long tiles=(a.node_count+Warps-1)/Warps;
  const long first=(tiles*blockIdx.x/gridDim.x)*Warps;
  const long stop=min(a.node_count,(tiles*(blockIdx.x+1)/gridDim.x)*Warps);
  for(long i=a.node_begin+first+warp;i<a.node_begin+stop;i+=Warps){
    for(int k=lane;k<P::StateWidth;k+=32){
      float q=a.moment_gradient[i*P::StateWidth+k];
      if(k>=20&&k<40&&(k<28||k>=32))q*=1.7320508075688772935f;
      own[warp][k]=q;
    }
    __syncwarp();
    const long begin=a.destination_row_ptr[i];
    const long live=reinterpret_cast<const int*>(a.edge_mask)[i];
    float virial[3]={};
    const int ti=int(a.atype[i]);
    const float4* records=reinterpret_cast<const float4*>(a.edge_vec)+begin;
    for(long start=0;start<live;start+=32){
      const long position=start+lane;const bool valid=position<live;
      float xyz[3];
      EdgeGeometry g=geometry_from_cache(a,records[valid?position:0],xyz);
      float gij[3];
      factored_force_pair_quadratic<1,true>(a,g,own[warp],ti,g.source_type,0,valid,gij,shared_table);
      if(valid){
        virial[0]-=gij[0]*xyz[1]; // slot 1
        virial[1]-=gij[0]*xyz[2]; // slot 2
        virial[2]-=gij[1]*xyz[2]; // slot 5
      }
    }
#pragma unroll
    for(int offset=16;offset;offset>>=1)
#pragma unroll
      for(int c=0;c<3;++c)virial[c]+=__shfl_down_sync(0xffffffffu,virial[c],offset);
    if(lane==0){v[i*9+1]=virial[0];v[i*9+2]=virial[1];v[i*9+5]=virial[2];}
    __syncwarp();
  }
}
