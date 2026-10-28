#pragma once
#include <cuda_runtime.h>
#include "dpa_fast_step_kokkos.h"
namespace DpaSmallEpoch {
struct Ev {
  double e, v[6], kinetic;
  KOKKOS_INLINE_FUNCTION Ev(): e(0), kinetic(0) { for(int k=0;k<6;++k)v[k]=0; }
  KOKKOS_INLINE_FUNCTION Ev& operator+=(const Ev& b) {
    e+=b.e; for(int k=0;k<6;++k)v[k]+=b.v[k]; kinetic+=b.kinetic; return *this;
  }
  KOKKOS_INLINE_FUNCTION void operator+=(const volatile Ev& b) volatile {
    e+=b.e; for(int k=0;k<6;++k)v[k]+=b.v[k]; kinetic+=b.kinetic;
  }
};
struct Args {
  LAMMPS_NS::DpaFastStep::Arr3<double> velocity;
  const int *mask, *type;
  const double *rmass, *mass;
  double dtf, press[3];
  int groupbit, tgroupbit;
};
KOKKOS_INLINE_FUNCTION double final_kinetic(const Args& a, int i, double f0,double f1,double f2) {
  double m=a.rmass?a.rmass[i]:a.mass[a.type[i]];
  if(a.mask[i]&a.groupbit) {
    double d=a.dtf/m;
    a.velocity(i,0)+=d*f0; a.velocity(i,1)+=d*f1; a.velocity(i,2)+=d*f2;
    for(int k=0;k<3;++k) a.velocity(i,k)*=a.press[k];
    for(int k=0;k<3;++k) a.velocity(i,k)*=a.press[k];
  }
  if(!(a.mask[i]&a.tgroupbit))return 0;
  return (a.velocity(i,0)*a.velocity(i,0)+a.velocity(i,1)*a.velocity(i,1)+a.velocity(i,2)*a.velocity(i,2))*m;
}
using Prepare=bool(*)(void*,double*,double*,double,bool,bool,Args*,Ev**);
using Publish=void(*)(cudaStream_t);
}
