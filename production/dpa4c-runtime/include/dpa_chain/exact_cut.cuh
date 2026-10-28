#pragma once
#include <cuda_runtime.h>
namespace deepmd_dpa4c {
// Exact predicate filter, NOT an approximate cutoff. At r^2 near 36, rounding
// three FP32 components and their squared sum changes r^2 by < 4e-5 under
// IEEE RN (including either contracted or uncontracted products). The 1e-3
// uncertainty interval is >25 times wider; even its FP32 endpoint rounding
// leaves this margin. All ambiguous values use the original double expression.
// Large/NaN values and alternative cutoffs also take the original expression.
__device__ __forceinline__ bool filtered_exact_cut(double dx,double dy,double dz,double cut2) {
  if(cut2==36.0) {
    const float x=float(dx),y=float(dy),z=float(dz);
    const float s=x*x+y*y+z*z;
    if(s>=1.0f && s<=100.0f) {
      const bool inside=s<35.999f,outside=s>36.001f;
      if(inside || outside)return inside;
    }
  }
  return dx*dx+dy*dy+dz*dz<cut2;
}
}
