namespace deepmd_dpa4c {
__device__ __forceinline__ void pair_vjp_streamed(const Arguments& a,EdgeGeometry g,
 const float* qi,int ti,const float* qj,int tj,float (&gij)[3],float (&gji)[3]){
 using P=Profile<8,2,false>;float basis[9];fill_angular_basis(g,basis);
 float dbi[9]={},dbj[9]={},direction[3]={};float rgi=0.f,rgj=0.f,egi=0.f,egj=0.f;
 const auto loc=locate_table<true>(g.radius,a.table_stride,a.table_max,a.interval_count);const auto row=table_row(a.table,loc,a.table_width);
 const auto* fi=reinterpret_cast<const float2*>(a.pair_film)+(long(ti)*a.type_count+tj)*8;
 const auto* fj=reinterpret_cast<const float2*>(a.pair_film)+(long(tj)*a.type_count+ti)*8;
 #pragma unroll
 for(int c=0;c<8;++c){
   float2 radial=evaluate_table_with_derivative(row,c,loc.coordinate,loc.clamped),wi=__ldg(fi+c),wj=__ldg(fj+c);
   float vi=fmaf(wi.x,radial.x,wi.y),vj=fmaf(wj.x,radial.x,wj.y);float ai=0.f,aj=0.f;
   if(c<4){
     #pragma unroll
     for(int d=0;d<3;++d){int at=P::VectorOffset+d*4+c;ai=fmaf(qi[at],basis[1+d],ai);aj=fmaf(qj[at],-basis[1+d],aj);}
     #pragma unroll
     for(int d=0;d<5;++d){int at=P::TensorOffset+d*4+c;ai=fmaf(qi[at],basis[4+d],ai);aj=fmaf(qj[at],basis[4+d],aj);}
   }
   float si=g.envelope*ai,sj=g.envelope*aj,fg_i=qi[P::ScalarOffset+c]+si,fg_j=qj[P::ScalarOffset+c]+sj;
   rgi=fmaf(g.envelope*fg_i,wi.x*radial.y,rgi);rgj=fmaf(g.envelope*fg_j,wj.x*radial.y,rgj);
   egi=fmaf(fg_i+si,vi,egi);egj=fmaf(fg_j+sj,vj,egj);
   float api=vi*g.envelope*g.envelope,apj=vj*g.envelope*g.envelope;
   if(c<4){
     #pragma unroll
     for(int d=0;d<3;++d){int at=P::VectorOffset+d*4+c;dbi[1+d]=fmaf(qi[at],api,dbi[1+d]);dbj[1+d]=fmaf(qj[at],apj,dbj[1+d]);}
     #pragma unroll
     for(int d=0;d<5;++d){int at=P::TensorOffset+d*4+c;dbi[4+d]=fmaf(qi[at],api,dbi[4+d]);dbj[4+d]=fmaf(qj[at],apj,dbj[4+d]);}
   }
 }
 const float square=g.envelope*g.envelope;
 egi=fmaf(4.f*square*g.envelope,qi[P::MomentWidth+1],fmaf(2.f*g.envelope,qi[P::MomentWidth],egi));
 egj=fmaf(4.f*square*g.envelope,qj[P::MomentWidth+1],fmaf(2.f*g.envelope,qj[P::MomentWidth],egj));
 const float de=c3_envelope_derivative(g.radius,a.rcut);rgi=fmaf(egi,de,rgi);rgj=fmaf(egj,de,rgj);
 basis_vjp(g,dbi,direction,rgi,gij);g.ux=-g.ux;g.uy=-g.uy;g.uz=-g.uz;basis_vjp(g,dbj,direction,rgj,gji);
}
}
