// Derivative of (1-x)^4*(1+4x+10x^2+20x^3+35x^4).
// Polynomial identity; same cutoff and envelope, FP32 reassociation only.
__device__ __forceinline__ float envelope_derivative_factored(float r,float rc){
  if(r<=0.f || r>=rc)return 0.f;
  float x=__fdividef(r,rc),u=1.f-x,x2=x*x,u2=u*u;
  return __fdividef(-280.f*x2*x2*u2*u,rc);
}
template<bool Estrin>
__device__ __forceinline__ float2 radial_value_slope(const TableRow&row,int c,float x,bool clamped){
  if constexpr(!Estrin)return evaluate_table_with_derivative(row,c,x,clamped);
  float4 a=__ldg(row.quartet+c);float2 b=__ldg(row.pair+c);float x2=x*x;
  float v=fmaf(x2,fmaf(x2,fmaf(b.y,x,b.x),fmaf(a.w,x,a.z)),fmaf(a.y,x,a.x));
  float d=fmaf(x2,fmaf(x2,5.f*b.y,fmaf(4.f*b.x,x,3.f*a.w)),fmaf(2.f*a.z,x,a.y));
  return make_float2(v,clamped?0.f:d);
}
