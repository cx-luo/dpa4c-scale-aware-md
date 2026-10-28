// When set, each kept edge is also written here (tile-local buffer).
__device__ float4* g_device_control_tile_dst = nullptr;

// Pipelined row compaction: identical output to compact_emit_row (same kept
// order, same float conversion), but each chunk of D*32 candidates issues all
// index loads, then all coordinate gathers, before the first ballot.
// With ktype, the kept edges' source types are written alongside (packed rows only).
template <int D>
__device__ __forceinline__ int compact_emit_row_pf(
    const DpaDirectNeighborView& v,long center,long begin,long end,float4* kept,float4* records,
    unsigned char* ktype,const Arguments& a) {
  const int lane=threadIdx.x&31;
  const long atom=v.model_to_atom[center];
  const double xi=__ldg(v.x+atom*v.stride_atom);
  const double yi=__ldg(v.x+atom*v.stride_atom+v.stride_xyz);
  const double zi=__ldg(v.x+atom*v.stride_atom+2*v.stride_xyz);
  int count=0;
  for(long chunk=begin;chunk<end;chunk+=D*32) {
    long j[D];std::uint32_t src[D];
#pragma unroll
    for(int k=0;k<D;++k) {
      const long e=chunk+k*32+lane;
      j[k]=-1;src[k]=0;
      if(e<end){j[k]=v.candidate_atom[e];src[k]=v.source[e];}
    }
    double px[D],py[D],pz[D];int ty[D];
#pragma unroll
    for(int k=0;k<D;++k) {
      px[k]=py[k]=pz[k]=0.0;ty[k]=0;
      if(j[k]>=0){
        const double* p=v.x+j[k]*v.stride_atom;
        px[k]=__ldg(p);py[k]=__ldg(p+v.stride_xyz);pz[k]=__ldg(p+2*v.stride_xyz);
        if(ktype)ty[k]=int(a.atype[src[k]]);
      }
    }
#pragma unroll
    for(int k=0;k<D;++k) {
      if(chunk+k*32>=end)break;
      bool active=false;float x=0,y=0,z=0;
      if(j[k]>=0) {
        const double dx=px[k]-xi,dy=py[k]-yi,dz=pz[k]-zi;
        active=keep_neighbor_edge(dx,dy,dz,v.cutoff_squared,a.edge_rmax_sq);
        if(active){x=float(dx*v.inverse_distance_unit);y=float(dy*v.inverse_distance_unit);z=float(dz*v.inverse_distance_unit);}
      }
      const unsigned mask=__ballot_sync(0xffffffffu,active);
      if(active){
        const int at=count+__popc(mask&((1u<<lane)-1u));
        float4 record=make_float4(x,y,z,__uint_as_float(src[k]));
        if(kept)kept[at]=record;
        if(ktype)ktype[at]=static_cast<unsigned char>(ty[k]);
        if(records)records[begin+at]=record;
        if(g_device_control_tile_dst) g_device_control_tile_dst[at]=record;
      }
      count+=__popc(mask);
    }
  }
  __syncwarp();
  return count;
}

// Identical to compact_emit_row_pf but never writes the global record array.
// The shared-memory copy still feeds the forward edge loop; the force pass
// would have to recompute geometry from the neighbor list instead.
template <int D>
__device__ __forceinline__ int compact_emit_row_pf_norec(
    const DpaDirectNeighborView& v,long center,long begin,long end,float4* kept,const Arguments& a) {
  const int lane=threadIdx.x&31;
  const long atom=v.model_to_atom[center];
  const double xi=__ldg(v.x+atom*v.stride_atom);
  const double yi=__ldg(v.x+atom*v.stride_atom+v.stride_xyz);
  const double zi=__ldg(v.x+atom*v.stride_atom+2*v.stride_xyz);
  int count=0;
  for(long chunk=begin;chunk<end;chunk+=D*32) {
    long j[D];std::uint32_t src[D];
#pragma unroll
    for(int k=0;k<D;++k) {
      const long e=chunk+k*32+lane;
      j[k]=-1;src[k]=0;
      if(e<end){j[k]=v.candidate_atom[e];src[k]=v.source[e];}
    }
    double px[D],py[D],pz[D];
#pragma unroll
    for(int k=0;k<D;++k) {
      px[k]=py[k]=pz[k]=0.0;
      if(j[k]>=0){
        const double* p=v.x+j[k]*v.stride_atom;
        px[k]=__ldg(p);py[k]=__ldg(p+v.stride_xyz);pz[k]=__ldg(p+2*v.stride_xyz);
      }
    }
#pragma unroll
    for(int k=0;k<D;++k) {
      if(chunk+k*32>=end)break;
      bool active=false;float x=0,y=0,z=0;
      if(j[k]>=0) {
        const double dx=px[k]-xi,dy=py[k]-yi,dz=pz[k]-zi;
        active=keep_neighbor_edge(dx,dy,dz,v.cutoff_squared,a.edge_rmax_sq);
        if(active){x=float(dx*v.inverse_distance_unit);y=float(dy*v.inverse_distance_unit);z=float(dz*v.inverse_distance_unit);}
      }
      const unsigned mask=__ballot_sync(0xffffffffu,active);
      if(active && kept){
        const int at=count+__popc(mask&((1u<<lane)-1u));
        const float4 record=make_float4(x,y,z,__uint_as_float(src[k]));
        kept[at]=record;
        if(g_device_control_tile_dst) g_device_control_tile_dst[at]=record;
      }
      count+=__popc(mask);
    }
  }
  __syncwarp();
  return count;
}

__device__ __forceinline__ EdgeGeometry geometry_from_record_type(
    const Arguments& a,const float4& record,float (&xyz)[3],int source_type) {
  EdgeGeometry g;
  xyz[0]=record.x;xyz[1]=record.y;xyz[2]=record.z;
  const float square=record.x*record.x+record.y*record.y+record.z*record.z+a.eps*a.eps;
  g.inverse_radius=rsqrtf(square);g.radius=square*g.inverse_radius;
  g.ux=record.x*g.inverse_radius;g.uy=record.y*g.inverse_radius;g.uz=record.z*g.inverse_radius;
  g.envelope=c3_envelope(g.radius,a.rcut);g.source=__float_as_uint(record.w);
  g.source_type=source_type;return g;
}
