namespace deepmd_dpa4c {
__global__ void prepare_geometry_kernel(const long* rows,DpaDirectNeighborView v,
 long nodes,float4* records,int* counts) {
 const int lane=threadIdx.x&31;const long i=long(blockIdx.x)*(blockDim.x/32)+threadIdx.x/32;
 if(i>=nodes)return;
 const long atom=v.model_to_atom[i];
 const double xi=v.x[atom*v.stride_atom],yi=v.x[atom*v.stride_atom+v.stride_xyz],zi=v.x[atom*v.stride_atom+2*v.stride_xyz];
 const long begin=rows[i],end=rows[i+1];int live=0;
 for(long base=begin;base<end;base+=32){
   const long e=base+lane;bool active=false;float x=0.f,y=0.f,z=0.f;
   if(e<end){const long j=v.candidate_atom[e];
     const double dx=v.x[j*v.stride_atom]-xi,dy=v.x[j*v.stride_atom+v.stride_xyz]-yi,dz=v.x[j*v.stride_atom+2*v.stride_xyz]-zi;
     active=filtered_exact_cut(dx,dy,dz,v.cutoff_squared);
     if(active){x=float(dx*v.inverse_distance_unit);y=float(dy*v.inverse_distance_unit);z=float(dz*v.inverse_distance_unit);}
   }
   unsigned mask=__ballot_sync(0xffffffffu,active);
   if(active)records[begin+live+__popc(mask&((1u<<lane)-1u))]=make_float4(x,y,z,__uint_as_float(v.source[e]));
   live+=__popc(mask);
 }
 if(lane==0)counts[i]=live;
}
void launch_prepare_geometry_c8(const long* rows,const DpaDirectNeighborView& view,long nodes,
 float4* records,int* counts,cudaStream_t s){prepare_geometry_kernel<<<(nodes+3)/4,128,0,s>>>(rows,view,nodes,records,counts);}
}
