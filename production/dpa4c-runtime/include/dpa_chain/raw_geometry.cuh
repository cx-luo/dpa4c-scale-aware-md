#include "exact_cut.cuh"
#include "direct_neighbor_view.h"
namespace deepmd_dpa4c {
__device__ __forceinline__ bool load_direct_geometry(
    const Arguments& a,const DpaDirectNeighborView& v,long center,long edge,
    EdgeGeometry& g,float (&xyz)[3]) {
  const long i=v.model_to_atom[center],j=v.candidate_atom[edge];
  const double dx=v.x[j*v.stride_atom]-v.x[i*v.stride_atom];
  const double dy=v.x[j*v.stride_atom+v.stride_xyz]-v.x[i*v.stride_atom+v.stride_xyz];
  const double dz=v.x[j*v.stride_atom+2*v.stride_xyz]-v.x[i*v.stride_atom+2*v.stride_xyz];
  if(!(filtered_exact_cut(dx,dy,dz,v.cutoff_squared)))return false;
  xyz[0]=float(dx*v.inverse_distance_unit);
  xyz[1]=float(dy*v.inverse_distance_unit);
  xyz[2]=float(dz*v.inverse_distance_unit);
  const float square=xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2]+a.eps*a.eps;
  g.inverse_radius=rsqrtf(square);g.radius=square*g.inverse_radius;
  g.ux=xyz[0]*g.inverse_radius;g.uy=xyz[1]*g.inverse_radius;g.uz=xyz[2]*g.inverse_radius;
  g.envelope=c3_envelope(g.radius,a.rcut);
  g.source=v.source[edge];g.source_type=int(a.atype[g.source]);
  return true;
}

// Every lane participates. Indices are packed in original candidate order.
// High-degree rows use the original exact traversal; capacity never prunes edges.
__device__ __forceinline__ int compact_direct_row(
    const DpaDirectNeighborView& v,long center,long begin,long end,float4* kept) {
  const int lane=threadIdx.x&31;
  const long atom=v.model_to_atom[center];
  const double xi=v.x[atom*v.stride_atom];
  const double yi=v.x[atom*v.stride_atom+v.stride_xyz];
  const double zi=v.x[atom*v.stride_atom+2*v.stride_xyz];
  int count=0;
  for(long base=begin;base<end;base+=32) {
    const long e=base+lane;bool active=false;float x=0,y=0,z=0;
    if(e<end) {
      const long j=v.candidate_atom[e];
      const double dx=v.x[j*v.stride_atom]-xi;
      const double dy=v.x[j*v.stride_atom+v.stride_xyz]-yi;
      const double dz=v.x[j*v.stride_atom+2*v.stride_xyz]-zi;
      active=filtered_exact_cut(dx,dy,dz,v.cutoff_squared);
      if(active){x=float(dx*v.inverse_distance_unit);y=float(dy*v.inverse_distance_unit);z=float(dz*v.inverse_distance_unit);}
    }
    const unsigned mask=__ballot_sync(0xffffffffu,active);
    if(active)kept[count+__popc(mask&((1u<<lane)-1u))]=make_float4(x,y,z,__uint_as_float(v.source[e]));
    count+=__popc(mask);
  }
  __syncwarp();
  return count;
}

__device__ __forceinline__ int compact_emit_row(
    const DpaDirectNeighborView& v,long center,long begin,long end,float4* kept,float4* records) {
  const int lane=threadIdx.x&31;
  const long atom=v.model_to_atom[center];
  const double xi=v.x[atom*v.stride_atom];
  const double yi=v.x[atom*v.stride_atom+v.stride_xyz];
  const double zi=v.x[atom*v.stride_atom+2*v.stride_xyz];
  int count=0;
  for(long base=begin;base<end;base+=32) {
    const long e=base+lane;bool active=false;float x=0,y=0,z=0;
    if(e<end) {
      const long j=v.candidate_atom[e];
      const double dx=v.x[j*v.stride_atom]-xi;
      const double dy=v.x[j*v.stride_atom+v.stride_xyz]-yi;
      const double dz=v.x[j*v.stride_atom+2*v.stride_xyz]-zi;
      active=filtered_exact_cut(dx,dy,dz,v.cutoff_squared);
      if(active){x=float(dx*v.inverse_distance_unit);y=float(dy*v.inverse_distance_unit);z=float(dz*v.inverse_distance_unit);}
    }
    const unsigned mask=__ballot_sync(0xffffffffu,active);
    if(active){const int at=count+__popc(mask&((1u<<lane)-1u));float4 record=make_float4(x,y,z,__uint_as_float(v.source[e]));if(kept)kept[at]=record;if(records)records[begin+at]=record;}
    count+=__popc(mask);
  }
  __syncwarp();
  return count;
}

__device__ __forceinline__ EdgeGeometry geometry_from_cache(
    const Arguments& a,const float4& record,float (&xyz)[3]) {
  EdgeGeometry g;
  xyz[0]=record.x;xyz[1]=record.y;xyz[2]=record.z;
  const float square=record.x*record.x+record.y*record.y+record.z*record.z+a.eps*a.eps;
  g.inverse_radius=rsqrtf(square);g.radius=square*g.inverse_radius;
  g.ux=record.x*g.inverse_radius;g.uy=record.y*g.inverse_radius;g.uz=record.z*g.inverse_radius;
  g.envelope=c3_envelope(g.radius,a.rcut);g.source=__float_as_uint(record.w);
  g.source_type=int(a.atype[g.source]);return g;
}
}
