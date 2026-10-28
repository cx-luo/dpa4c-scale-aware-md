// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stan Moore (SNL)
                         Mitch Murphy (alphataubio@gmail.com)
------------------------------------------------------------------------- */

#include "fix_nh_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "compute_temp_kokkos.h"
#include "domain_kokkos.h"
#include "dpa_fast_step_kokkos.h"
#include "error.h"
#include "force.h"
#include "irregular.h"
#include "kspace.h"
#include "neighbor.h"
#include "output.h"
#include "modify.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <type_traits>
#include <typeinfo>

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {
template<class DeviceType, class V, class F, class M, class R, class DT>
void nh_small_v_rmass(int n, V vv, F ff, M mm, R rr, DT dt, int gb) {
  Kokkos::parallel_for("FixNHKokkos::nve_v_rmass", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i) {
      if (mm[i] & gb) {
        const F_FLOAT dtfm = dt / rr[i];
        vv(i,0) += dtfm*ff(i,0); vv(i,1) += dtfm*ff(i,1); vv(i,2) += dtfm*ff(i,2);
      }
    });
}
template<class DeviceType, class V, class F, class M, class R, class T, class DT>
void nh_small_v_mass(int n, V vv, F ff, M mm, R mass, T type, DT dt, int gb) {
  Kokkos::parallel_for("FixNHKokkos::nve_v_mass", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i) {
      if (mm[i] & gb) {
        const F_FLOAT dtfm = dt / mass[type[i]];
        vv(i,0) += dtfm*ff(i,0); vv(i,1) += dtfm*ff(i,1); vv(i,2) += dtfm*ff(i,2);
      }
    });
}
template<class DeviceType, class X, class V, class M, class DT>
void nh_small_x(int n, X xx, V vv, M mm, DT dt, int gb) {
  Kokkos::parallel_for("FixNHKokkos::nve_x", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i) {
      if (mm[i] & gb) {
        xx(i,0) += dt*vv(i,0); xx(i,1) += dt*vv(i,1); xx(i,2) += dt*vv(i,2);
      }
    });
}
template<class DeviceType, class V, class M, class Factor>
void nh_small_temp(int n, V vv, M mm, Factor factor, int gb) {
  Kokkos::parallel_for("FixNHKokkos::nh_v_temp", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i) {
      if (mm[i] & gb) { vv(i,0) *= factor; vv(i,1) *= factor; vv(i,2) *= factor; }
    });
}
template<class DeviceType, class A, class B>
double max_abs_diff3(int n, A a, B b) {
  double m = 0.0;
  Kokkos::parallel_reduce("FixNHKokkos::verify_diff", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i, double &mm) {
      for (int k = 0; k < 3; ++k) {
        const double d = fabs(a(i,k) - b(i,k));
        if (d > mm) mm = d;
      }
    }, Kokkos::Max<double>(m));
  return m;
}
// nh_v_temp (optional), nve_v and nve_x in one pass, same per-atom arithmetic
template<int RMASS, class DeviceType, class X, class V, class F, class M, class R, class T>
void nh_small_vx(int n, X xx, V vv, F ff, M mm, R ms, T type, bool scale, double fe,
                 double dtf, double dtv, int gb) {
  Kokkos::parallel_for("FixNHKokkos::nve_vx", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i) {
      if (mm[i] & gb) {
        if (scale) { vv(i,0) *= fe; vv(i,1) *= fe; vv(i,2) *= fe; }
        const F_FLOAT dtfm = dtf / (RMASS ? ms[i] : ms[type[i]]);
        vv(i,0) += dtfm*ff(i,0); vv(i,1) += dtfm*ff(i,1); vv(i,2) += dtfm*ff(i,2);
        xx(i,0) += dtv*vv(i,0); xx(i,1) += dtv*vv(i,1); xx(i,2) += dtv*vv(i,2);
      }
    });
}
struct NHBoxMap { double lo[3]; double m[6]; };
// nh_v_press (non-triclinic), nve_v, remap (x2lamda with a_inv, lamda2x with b),
// nve_x, remap (x2lamda with b_inv, lamda2x with c): DomainKokkos arithmetic
template<int RMASS, class DeviceType, class X, class V, class F, class M, class R, class T>
void nh_npt_vx(int n, X xx, V vv, F ff, M mm, R ms, T type, double p0, double p1, double p2,
               double dtf, double dtv, int gb, NHBoxMap ai, NHBoxMap b, NHBoxMap bi, NHBoxMap c) {
  Kokkos::parallel_for("FixNHKokkos::npt_vx", Kokkos::RangePolicy<DeviceType>(0,n),
    KOKKOS_LAMBDA(const int i) {
      const int m = mm[i];
      if (m & gb) {
        vv(i,0) *= p0; vv(i,1) *= p1; vv(i,2) *= p2;
        vv(i,0) *= p0; vv(i,1) *= p1; vv(i,2) *= p2;
        const F_FLOAT dtfm = dtf / (RMASS ? ms[i] : ms[type[i]]);
        vv(i,0) += dtfm*ff(i,0); vv(i,1) += dtfm*ff(i,1); vv(i,2) += dtfm*ff(i,2);
      }
      {
        F_FLOAT delta[3];
        delta[0] = xx(i,0) - ai.lo[0];
        delta[1] = xx(i,1) - ai.lo[1];
        delta[2] = xx(i,2) - ai.lo[2];
        xx(i,0) = ai.m[0]*delta[0] + ai.m[5]*delta[1] + ai.m[4]*delta[2];
        xx(i,1) = ai.m[1]*delta[1] + ai.m[3]*delta[2];
        xx(i,2) = ai.m[2]*delta[2];
        const double xi1 = xx(i,1);
        const double xi2 = xx(i,2);
        xx(i,0) = b.m[0]*xx(i,0) + b.m[5]*xi1 + b.m[4]*xi2 + b.lo[0];
        xx(i,1) = b.m[1]*xi1 + b.m[3]*xi2 + b.lo[1];
        xx(i,2) = b.m[2]*xi2 + b.lo[2];
      }
      if (m & gb) {
        xx(i,0) += dtv*vv(i,0); xx(i,1) += dtv*vv(i,1); xx(i,2) += dtv*vv(i,2);
      }
      {
        F_FLOAT delta[3];
        delta[0] = xx(i,0) - bi.lo[0];
        delta[1] = xx(i,1) - bi.lo[1];
        delta[2] = xx(i,2) - bi.lo[2];
        xx(i,0) = bi.m[0]*delta[0] + bi.m[5]*delta[1] + bi.m[4]*delta[2];
        xx(i,1) = bi.m[1]*delta[1] + bi.m[3]*delta[2];
        xx(i,2) = bi.m[2]*delta[2];
        const double xi1 = xx(i,1);
        const double xi2 = xx(i,2);
        xx(i,0) = c.m[0]*xx(i,0) + c.m[5]*xi1 + c.m[4]*xi2 + c.lo[0];
        xx(i,1) = c.m[1]*xi1 + c.m[3]*xi2 + c.lo[1];
        xx(i,2) = c.m[2]*xi2 + c.lo[2];
      }
    });
}
} // namespace

static constexpr double DELTAFLIP = 0.1;
static constexpr double TILTMAX = 1.5;
static constexpr double EPSILON = 1.0e-6;

enum{NOBIAS,BIAS};
enum{NONE,XYZ,XY,YZ,XZ};
enum{ISO,ANISO,TRICLINIC};

/* ----------------------------------------------------------------------
   NVT,NPH,NPT integrators for improved Nose-Hoover equations of motion
 ---------------------------------------------------------------------- */

template<class DeviceType>
FixNHKokkos<DeviceType>::FixNHKokkos(LAMMPS *lmp, int narg, char **arg) : FixNH(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *)atom;
  domainKK = (DomainKokkos *) domain;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::init()
{
  FixNH::init();

  atomKK->k_mass.modify<LMPHostType>();
  atomKK->k_mass.sync<DeviceType>();
}

/* ----------------------------------------------------------------------
   compute T,P before integrator starts
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::setup(int /*vflag*/)
{
  // tdof needed by compute_temp_target()

  atomKK->sync(temperature->execution_space,temperature->datamask_read);
  t_current = temperature->compute_scalar();
  atomKK->modified(temperature->execution_space,temperature->datamask_modify);
  atomKK->sync(execution_space,temperature->datamask_modify);

  tdof = temperature->dof;

  // t_target is needed by NPH and NPT in compute_scalar()
  // If no thermostat or using fix nphug,
  // t_target must be defined by other means.

  if (tstat_flag && strcmp(style,"nphug") != 0) {
    compute_temp_target();
  } else if (pstat_flag) {

    // t0 = reference temperature for masses
    // set equal to either ptemp or the current temperature
    // cannot be done in init() b/c temperature cannot be called there
    // is b/c Modify::init() inits computes after fixes due to dof dependence
    // error if T less than 1e-6
    // if it was read in from a restart file, leave it be

    if (t0 == 0.0) {
      if (p_temp_flag) {
        t0 = p_temp;
      } else {
        atomKK->sync(temperature->execution_space,temperature->datamask_read);
        t0 = temperature->compute_scalar();
        atomKK->modified(temperature->execution_space,temperature->datamask_modify);
        atomKK->sync(execution_space,temperature->datamask_modify);
        if (t0 < EPSILON)
          error->all(FLERR,"Current temperature too close to zero, consider using ptemp keyword");
      }
    }
    t_target = t0;
  }

  if (pstat_flag) compute_press_target();

  atomKK->sync(temperature->execution_space,temperature->datamask_read);
  t_current = temperature->compute_scalar();
  atomKK->modified(temperature->execution_space,temperature->datamask_modify);
  atomKK->sync(execution_space,temperature->datamask_modify);

  tdof = temperature->dof;

  if (pstat_flag) {
    //atomKK->sync(pressure->execution_space,pressure->datamask_read);
    //atomKK->modified(pressure->execution_space,pressure->datamask_modify);
    if (pstyle == ISO) pressure->compute_scalar();
    else pressure->compute_vector();
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  // masses and initial forces on thermostat variables

  if (tstat_flag) {
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++)
      eta_mass[ich] = boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++) {
      eta_dotdot[ich] = (eta_mass[ich-1]*eta_dot[ich-1]*eta_dot[ich-1] -
                         boltz * t_target) / eta_mass[ich];
    }
  }

  // masses and initial forces on barostat variables

  if (pstat_flag) {
    double kt = boltz * t_target;
    double nkt = (atom->natoms + 1) * kt;

    for (int i = 0; i < 3; i++)
      if (p_flag[i])
        omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);

    if (pstyle == TRICLINIC) {
      for (int i = 3; i < 6; i++)
        if (p_flag[i]) omega_mass[i] = nkt/(p_freq[i]*p_freq[i]);
    }

  // masses and initial forces on barostat thermostat variables

    if (mpchain) {
      etap_mass[0] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_mass[ich] = boltz * t_target / (p_freq_max*p_freq_max);
      for (int ich = 1; ich < mpchain; ich++)
        etap_dotdot[ich] =
          (etap_mass[ich-1]*etap_dot[ich-1]*etap_dot[ich-1] -
           boltz * t_target) / etap_mass[ich];
    }

  }

  nhc_device = nhc_device_allowed();
  nhc_upload_pending = nhc_device;
  if (DpaFastStep::enabled(DpaFastStep::kDeviceNhc) && comm->me == 0) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      auto *tkk = dynamic_cast<ComputeTempKokkos<DeviceType> *>(temperature);
      utils::logmesg(lmp, "DPA_NHC_DEVICE {} fast={} tstat={} pstat={} nc={} m={} dyn={} dof={} np={}\n",
                     nhc_device ? 1 : 0, fast_step_allowed() ? 1 : 0, tstat_flag, pstat_flag, nc_tchain,
                     mtchain, tkk ? (tkk->dynamic_dof() ? 1 : 0) : -1, temperature->dof, comm->nprocs);
    }
  }
}

/* ----------------------------------------------------------------------
   1st half of Verlet update
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  if (nhc_device) {
    if (nhc_upload_pending) {
      nhc_upload();
      nhc_upload_pending = false;
    }
    compute_temp_target();
    nhc_device_initial();
    return;
  }

  // update eta_press_dot

  if (pstat_flag && mpchain) nhc_press_integrate();

  const bool fast = fast_step_allowed() && DpaFastStep::enabled(DpaFastStep::kInitial) &&
    nc_tchain == 1 && (!pstat_flag || fast_npt_allowed());

  // update eta_dot

  if (tstat_flag) {
    compute_temp_target();
    defer_temp_scale = fast;
    nhc_temp_integrate();
    defer_temp_scale = false;
  }

  if (fast && !pstat_flag) {
    fused_nvt_initial();
    return;
  }

  // need to recompute pressure to account for change in KE
  // t_current is up-to-date, but compute_temperature is not
  // compute appropriately coupled elements of mvv_current

  if (pstat_flag) {
    if (fast) {
      fused_scale_temp();
      atomKK->sync(pressure->execution_space,pressure->datamask_read);
      pressure->compute_scalar();
    } else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      atomKK->sync(pressure->execution_space,pressure->datamask_read);
      if (pstyle == ISO) {
        temperature->compute_scalar();
        pressure->compute_scalar();
      } else {
        temperature->compute_vector();
        pressure->compute_vector();
      }
    }
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->modified(pressure->execution_space,pressure->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,pressure->datamask_modify);
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    if (fast) {
      fused_npt_initial();
      if (kspace_flag) force->kspace->setup();
      return;
    }
    nh_v_press();
  }

  nve_v();

  // remap simulation box by 1/2 step

  if (pstat_flag) remap();

  nve_x();

  // remap simulation box by 1/2 step
  // redo KSpace coeffs since volume has changed

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

/* ----------------------------------------------------------------------
   2nd half of Verlet update
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::final_integrate()
{
  if (nhc_device) {
    tdof = temperature->dof;
    nhc_device_final();
    return;
  }
  if (!fast_nve_v_temp()) {
    nve_v();

    // re-compute temp before nh_v_press()
    // only needed for temperature computes with BIAS on reneighboring steps:
    //   b/c some biases store per-atom values (e.g. temp/profile)
    //   per-atom values are invalid if reneigh/comm occurred
    //     since temp->compute() in initial_integrate()

    if (which == BIAS && neighbor->ago == 0) {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      t_current = temperature->compute_scalar();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }

    if (pstat_flag) nh_v_press();

    // compute new T,P after velocities rescaled by nh_v_press()
    // compute appropriately coupled elements of mvv_current

    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    t_current = temperature->compute_scalar();
    atomKK->modified(temperature->execution_space,temperature->datamask_modify);
    atomKK->sync(execution_space,temperature->datamask_modify);
  }
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) {
      atomKK->sync(pressure->execution_space,pressure->datamask_read);
      pressure->compute_scalar();
      atomKK->modified(pressure->execution_space,pressure->datamask_modify);
      atomKK->sync(execution_space,pressure->datamask_modify);
    } else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      atomKK->sync(pressure->execution_space,pressure->datamask_read);
      temperature->compute_vector();
      pressure->compute_vector();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->modified(pressure->execution_space,pressure->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,pressure->datamask_modify);
    }
    couple();
    pressure->addstep(update->ntimestep+1);
  }

  if (pstat_flag) nh_omega_dot();

  // update eta_dot
  // update eta_press_dot

  if (tstat_flag) nhc_temp_integrate();
  if (pstat_flag && mpchain) nhc_press_integrate();
}

/* ----------------------------------------------------------------------
   change box size
   remap all atoms or dilate group atoms depending on allremap flag
   if rigid bodies exist, scale rigid body centers-of-mass
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::remap()
{
  int nlocal = atom->nlocal;

  // convert pertinent atoms and rigid bodies to lamda coords

  x = atomKK->k_x.template view<DeviceType>();

  if (allremap) domainKK->x2lamda(nlocal);
  else {
    for ( int i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domainKK->x2lamda(&x(i,0), &x(i,0));
  }

  if (rfix.size() > 0)
    error->all(FLERR,"Cannot (yet) use rigid bodies with fix nh and Kokkos");
  // for (auto &ifix : rfix) ifix->deform(0);

  remap_box();

  // convert pertinent atoms and rigid bodies back to box coords

  if (allremap) domainKK->lamda2x(nlocal);
  else {
    for ( int i = 0; i < nlocal; i++)
      if (mask[i] & dilate_group_bit)
        domainKK->lamda2x(&x(i,0), &x(i,0));
  }

  // for (auto &ifix : rfix) ifix->deform(1);
}

/* ----------------------------------------------------------------------
   box part of remap(): no atom coordinates are read or written
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::remap_box()
{
  double oldlo,oldhi;
  double expfac;

  double *h = domain->h;

  // omega is not used, except for book-keeping

  for (int i = 0; i < 6; i++) omega[i] += dto*omega_dot[i];

  // reset global and local box to new size/shape

  // this operation corresponds to applying the
  // translate and scale operations
  // corresponding to the solution of the following ODE:
  //
  // h_dot = omega_dot * h
  //
  // where h_dot, omega_dot and h are all upper-triangular
  // 3x3 tensors. In Voigt notation, the elements of the
  // RHS product tensor are:
  // h_dot = [0*0, 1*1, 2*2, 1*3+3*2, 0*4+5*3+4*2, 0*5+5*1]
  //
  // Ordering of operations preserves time symmetry.

  double dto2 = dto/2.0;
  double dto4 = dto/4.0;
  double dto8 = dto/8.0;

  // off-diagonal components, first half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }
  }

  // scale diagonal components
  // scale tilt factors with cell, if set

  if (p_flag[0]) {
    oldlo = domain->boxlo[0];
    oldhi = domain->boxhi[0];
    expfac = exp(dto*omega_dot[0]);
    domain->boxlo[0] = (oldlo-fixedpoint[0])*expfac + fixedpoint[0];
    domain->boxhi[0] = (oldhi-fixedpoint[0])*expfac + fixedpoint[0];
  }

  if (p_flag[1]) {
    oldlo = domain->boxlo[1];
    oldhi = domain->boxhi[1];
    expfac = exp(dto*omega_dot[1]);
    domain->boxlo[1] = (oldlo-fixedpoint[1])*expfac + fixedpoint[1];
    domain->boxhi[1] = (oldhi-fixedpoint[1])*expfac + fixedpoint[1];
    if (scalexy) h[5] *= expfac;
  }

  if (p_flag[2]) {
    oldlo = domain->boxlo[2];
    oldhi = domain->boxhi[2];
    expfac = exp(dto*omega_dot[2]);
    domain->boxlo[2] = (oldlo-fixedpoint[2])*expfac + fixedpoint[2];
    domain->boxhi[2] = (oldhi-fixedpoint[2])*expfac + fixedpoint[2];
    if (scalexz) h[4] *= expfac;
    if (scaleyz) h[3] *= expfac;
  }

  // off-diagonal components, second half

  if (pstyle == TRICLINIC) {

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

    if (p_flag[3]) {
      expfac = exp(dto4*omega_dot[1]);
      h[3] *= expfac;
      h[3] += dto2*(omega_dot[3]*h[2]);
      h[3] *= expfac;
    }

    if (p_flag[5]) {
      expfac = exp(dto4*omega_dot[0]);
      h[5] *= expfac;
      h[5] += dto2*(omega_dot[5]*h[1]);
      h[5] *= expfac;
    }

    if (p_flag[4]) {
      expfac = exp(dto8*omega_dot[0]);
      h[4] *= expfac;
      h[4] += dto4*(omega_dot[5]*h[3]+omega_dot[4]*h[2]);
      h[4] *= expfac;
    }

  }

  domain->yz = h[3];
  domain->xz = h[4];
  domain->xy = h[5];

  // tilt factor to cell length ratio can not exceed TILTMAX in one step

  if (domain->yz < -TILTMAX*domain->yprd ||
      domain->yz > TILTMAX*domain->yprd ||
      domain->xz < -TILTMAX*domain->xprd ||
      domain->xz > TILTMAX*domain->xprd ||
      domain->xy < -TILTMAX*domain->xprd ||
      domain->xy > TILTMAX*domain->xprd)
    error->all(FLERR,"Fix npt/nph has tilted box too far in one step - "
               "periodic cell is too far from equilibrium state");

  domain->set_global_box();
  domain->set_local_box();
}

/* ----------------------------------------------------------------------
   perform half-step barostat scaling of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nh_v_press()
{
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2));
  factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2));
  factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2));

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->remove_bias_all_kk();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->remove_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }

  atomKK->sync(execution_space,V_MASK | MASK_MASK);

  copymode = 1;
  if (pstyle == TRICLINIC)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nh_v_press<1> >(0,nlocal),*this);
  else
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNH_nh_v_press<0> >(0,nlocal),*this);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->restore_bias_all();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->restore_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }
}

template<class DeviceType>
template<int TRICLINIC_FLAG>
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nh_v_press<TRICLINIC_FLAG>, const int &i) const {
  if (mask[i] & groupbit) {
    v(i,0) *= factor[0];
    v(i,1) *= factor[1];
    v(i,2) *= factor[2];
    if (TRICLINIC_FLAG) {
      v(i,0) += -dthalf*(v(i,1)*omega_dot[5] + v(i,2)*omega_dot[4]);
      v(i,1) += -dthalf*v(i,2)*omega_dot[3];
    }
    v(i,0) *= factor[0];
    v(i,1) *= factor[1];
    v(i,2) *= factor[2];
  }
}

/* ----------------------------------------------------------------------
   perform half-step update of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nve_v()
{
  atomKK->sync(execution_space,X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);

  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  copymode = 1;
  if (rmass.data())
    nh_small_v_rmass<DeviceType>(nlocal,v,f,mask,rmass,dtf,groupbit);
  else
    nh_small_v_mass<DeviceType>(nlocal,v,f,mask,mass,type,dtf,groupbit);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);
}

template<class DeviceType>
template<int RMASS>
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nve_v<RMASS>, const int &i) const {
  if (RMASS) {
    if (mask[i] & groupbit) {
      const F_FLOAT dtfm = dtf / rmass[i];
      v(i,0) += dtfm*f(i,0);
      v(i,1) += dtfm*f(i,1);
      v(i,2) += dtfm*f(i,2);
    }
  } else {
    if (mask[i] & groupbit) {
      const F_FLOAT dtfm = dtf / mass[type[i]];
      v(i,0) += dtfm*f(i,0);
      v(i,1) += dtfm*f(i,1);
      v(i,2) += dtfm*f(i,2);
    }
  }
}

/* ----------------------------------------------------------------------
   perform full-step update of positions
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nve_x()
{
  atomKK->sync(execution_space,X_MASK | V_MASK | MASK_MASK);
  atomKK->modified(execution_space,X_MASK);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  // x update by full step only for atoms in group

  copymode = 1;
  nh_small_x<DeviceType>(nlocal,x,v,mask,dtv,groupbit);
  copymode = 0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nve_x, const int &i) const {
  if (mask[i] & groupbit) {
    x(i,0) += dtv * v(i,0);
    x(i,1) += dtv * v(i,1);
    x(i,2) += dtv * v(i,2);
  }
}

/* ----------------------------------------------------------------------
   perform half-step thermostat scaling of velocities
-----------------------------------------------------------------------*/

template<class DeviceType>
void FixNHKokkos<DeviceType>::nh_v_temp()
{
  if (defer_temp_scale) {
    pending_factor_eta = factor_eta;
    temp_scale_pending = true;
    return;
  }

  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->remove_bias_all_kk();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->remove_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }

  atomKK->sync(execution_space,V_MASK | MASK_MASK);

  copymode = 1;
  nh_small_temp<DeviceType>(nlocal,v,mask,factor_eta,groupbit);
  copymode = 0;

  atomKK->modified(execution_space,V_MASK);

  if (which == BIAS) {
    if (temperature->kokkosable) temperature->restore_bias_all();
    else {
      atomKK->sync(temperature->execution_space,temperature->datamask_read);
      temperature->restore_bias_all();
      atomKK->modified(temperature->execution_space,temperature->datamask_modify);
      atomKK->sync(execution_space,temperature->datamask_modify);
    }
  }
}

/* ----------------------------------------------------------------------
   short device paths, see dpa_fast_step_kokkos.h
------------------------------------------------------------------------- */

template<class DeviceType>
bool FixNHKokkos<DeviceType>::fast_step_allowed() const
{
  if (!DpaFastStep::is_cuda_device<DeviceType>() || which != NOBIAS) return false;
  if (igroup == atomKK->firstgroup && atomKK->nfirst != atomKK->nlocal) return false;
  if (typeid(*temperature) != typeid(ComputeTempKokkos<DeviceType>) ||
      temperature->execution_space != execution_space)
    return false;
  return true;
}

template<class DeviceType>
bool FixNHKokkos<DeviceType>::fast_npt_allowed() const
{
  return pstyle == ISO && allremap && rfix.size() == 0 && domain->triclinic == 0;
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::press_factors()
{
  factor[0] = exp(-dt4*(omega_dot[0]+mtk_term2));
  factor[1] = exp(-dt4*(omega_dot[1]+mtk_term2));
  factor[2] = exp(-dt4*(omega_dot[2]+mtk_term2));
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::fused_nve_vx(bool scale, double scale_factor)
{
  atomKK->sync(execution_space,X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atomKK->nlocal;

  copymode = 1;
  if (rmass.data())
    nh_small_vx<1,DeviceType>(nlocal,x,v,f,mask,rmass,type,scale,scale_factor,dtf,dtv,groupbit);
  else
    nh_small_vx<0,DeviceType>(nlocal,x,v,f,mask,mass,type,scale,scale_factor,dtf,dtv,groupbit);
  copymode = 0;

  atomKK->modified(execution_space,X_MASK | V_MASK);
}

/* ----------------------------------------------------------------------
   one deterministic reduction kernel: optional nh_v_temp (pending factor),
   nve_v and nh_v_press, then the temperature per-atom sum; returns local sum
------------------------------------------------------------------------- */

template<class DeviceType>
double FixNHKokkos<DeviceType>::v_update_temp(bool scale, bool nve, bool press, bool read)
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (DpaFastStep::is_cuda_device<DeviceType>()) {
    atomKK->sync(execution_space,X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
    atomKK->k_mass.sync<DeviceType>();

    v = atomKK->k_v.view<DeviceType>();
    f = atomKK->k_f.view<DeviceType>();
    rmass = atomKK->k_rmass.view<DeviceType>();
    mass = atomKK->k_mass.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();
    mask = atomKK->k_mask.view<DeviceType>();
    const int nlocal = atomKK->nlocal;

    const DpaFastStep::VUpdate u{pending_factor_eta, dtf, {factor[0], factor[1], factor[2]}};
    const auto va = DpaFastStep::arr3(v);
    const DpaFastStep::Arr3<const double> fa{f.data(), (long) f.stride_0(), (long) f.stride_1()};
    const int tgb = temperature->groupbit;
    const int key = (rmass.data() ? 8 : 0) | (scale ? 4 : 0) | (nve ? 2 : 0) | (press ? 1 : 0);
    switch (key) {
#define DPA_V_CASE(K) case K: \
      DpaFastStep::launch_v_update_temp<(K>>3)&1,(K>>2)&1,(K>>1)&1,K&1>( \
        nlocal,va,fa,mask.data(),rmass.data(),mass.data(),type.data(),u,groupbit,tgb); break;
      DPA_V_CASE(0) DPA_V_CASE(1) DPA_V_CASE(2) DPA_V_CASE(3)
      DPA_V_CASE(4) DPA_V_CASE(5) DPA_V_CASE(6) DPA_V_CASE(7)
      DPA_V_CASE(8) DPA_V_CASE(9) DPA_V_CASE(10) DPA_V_CASE(11)
      DPA_V_CASE(12) DPA_V_CASE(13) DPA_V_CASE(14) DPA_V_CASE(15)
#undef DPA_V_CASE
    }
    if (scale || nve || press) atomKK->modified(execution_space,V_MASK);
    if (!read) return 0.0;
    return DpaFastStep::read_result(DpaFastStep::scratch());
  }
#endif
  return 0.0;
}

/* ----------------------------------------------------------------------
   final_integrate: nve_v [+ nh_v_press] + temperature in one kernel
------------------------------------------------------------------------- */

template<class DeviceType>
bool FixNHKokkos<DeviceType>::fast_nve_v_temp()
{
  if (!fast_step_allowed() || !DpaFastStep::enabled(DpaFastStep::kFinal)) return false;
  if (pstat_flag && !fast_npt_allowed()) return false;
  auto *tkk = static_cast<ComputeTempKokkos<DeviceType> *>(temperature);

  const bool check = DpaFastStep::verify();
  atomKK->sync(execution_space,V_MASK);
  auto vk = atomKK->k_v.view<DeviceType>();
  decltype(vk) v0, v1;
  if (check) {
    v0 = decltype(vk)("v0",vk.extent(0));
    v1 = decltype(vk)("v1",vk.extent(0));
    Kokkos::deep_copy(v0,vk);
  }

  if (pstat_flag) press_factors();
  t_current = tkk->scalar_from_local_sum(v_update_temp(false,true,pstat_flag));

  if (check) {
    static DpaFastStep::VerifyStats stats("final_integrate dv rel_dT");
    const double t_fast = t_current;
    vk = atomKK->k_v.view<DeviceType>();
    Kokkos::deep_copy(v1,vk);
    Kokkos::deep_copy(vk,v0);
    nve_v();
    if (pstat_flag) nh_v_press();
    atomKK->sync(temperature->execution_space,temperature->datamask_read);
    t_current = temperature->compute_scalar();
    const double dv = max_abs_diff3<DeviceType>(atomKK->nlocal,vk,v1);
    const double rel = t_current != 0.0 ? fabs(t_fast - t_current) / fabs(t_current) : fabs(t_fast);
    stats.add(dv != 0.0 || rel > 1e-13, dv, rel);
  }
  return true;
}

/* ----------------------------------------------------------------------
   NVT chain on the device (DPA_LMP_FAST_STEP bit 16)
------------------------------------------------------------------------- */

template<class DeviceType>
bool FixNHKokkos<DeviceType>::nhc_device_allowed() const
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (DpaFastStep::is_cuda_device<DeviceType>()) {
    if (!DpaFastStep::enabled(DpaFastStep::kDeviceNhc) || DpaFastStep::verify()) return false;
    if (!DpaFastStep::within_device_nhc_limit(atom->natoms)) return false;
    if (!DpaFastStep::enabled(DpaFastStep::kInitial) || !DpaFastStep::enabled(DpaFastStep::kFinal)) return false;
    if (!fast_step_allowed() || !tstat_flag || pstat_flag || nc_tchain != 1) return false;
    if (mtchain < 1 || mtchain > DpaFastStep::kMaxChain) return false;
    auto *tkk = static_cast<ComputeTempKokkos<DeviceType> *>(temperature);
    if (tkk->dynamic_dof() || temperature->dof <= 0.0 || comm->nprocs != 1) return false;
    return true;
  }
#endif
  return false;
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::nhc_upload()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (DpaFastStep::is_cuda_device<DeviceType>()) {
    DpaFastStep::NhcState h{};
    for (int i = 0; i < mtchain; i++) {
      h.eta[i] = eta[i];
      h.eta_dotdot[i] = eta_dotdot[i];
      h.eta_mass[i] = eta_mass[i];
    }
    for (int i = 0; i <= mtchain; i++) h.eta_dot[i] = eta_dot[i];
    h.t_current = t_current;
    h.factor_eta = factor_eta;
    auto &b = DpaFastStep::nhc_buffers();
    cudaMemcpy(b.state + b.current, &h, sizeof(h), cudaMemcpyHostToDevice);
  }
#endif
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::nhc_download()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (DpaFastStep::is_cuda_device<DeviceType>()) {
    DpaFastStep::NhcState h{};
    auto &b = DpaFastStep::nhc_buffers();
    cudaMemcpy(&h, b.state + b.current, sizeof(h), cudaMemcpyDeviceToHost);
    for (int i = 0; i < mtchain; i++) {
      eta[i] = h.eta[i];
      eta_dotdot[i] = h.eta_dotdot[i];
      eta_mass[i] = h.eta_mass[i];
    }
    for (int i = 0; i <= mtchain; i++) eta_dot[i] = h.eta_dot[i];
    t_current = h.t_current;
    factor_eta = h.factor_eta;
  }
#endif
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::post_run()
{
  if (nhc_device && !nhc_upload_pending) {
    nhc_download();
    nhc_upload_pending = true;
  }
}

template<class DeviceType>
DpaFastStep::NhcParams FixNHKokkos<DeviceType>::nhc_params()
{
  DpaFastStep::NhcParams p{};
#if defined(KOKKOS_ENABLE_CUDA)
  if (eta_mass_flag) {
    eta_mass[0] = tdof * boltz * t_target / (t_freq*t_freq);
    for (int ich = 1; ich < mtchain; ich++)
      eta_mass[ich] = boltz * t_target / (t_freq*t_freq);
  }
  p.tdof_boltz = tdof * boltz;
  p.ke_target = ke_target;
  p.boltz_t_target = boltz * t_target;
  p.dt4 = dt4;
  p.dt8 = dt8;
  p.dthalf = dthalf;
  p.tdrag_factor = tdrag_factor;
  p.tfactor = force->mvv2e / (temperature->dof * force->boltz);
  for (int i = 0; i < mtchain; i++) {
    p.eta_mass[i] = eta_mass[i];
    p.inv_eta_mass[i] = 1.0 / eta_mass[i];
  }
#endif
  return p;
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::nhc_device_initial()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (DpaFastStep::is_cuda_device<DeviceType>()) {
    atomKK->sync(execution_space,X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
    atomKK->k_mass.sync<DeviceType>();
    x = atomKK->k_x.view<DeviceType>();
    v = atomKK->k_v.view<DeviceType>();
    f = atomKK->k_f.view<DeviceType>();
    rmass = atomKK->k_rmass.view<DeviceType>();
    mass = atomKK->k_mass.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();
    mask = atomKK->k_mask.view<DeviceType>();
    int nlocal = atomKK->nlocal;
    if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;
    const DpaFastStep::NhcParams p = nhc_params();
    auto &b = DpaFastStep::nhc_buffers();
    const auto *in = b.state + b.current;
    auto *out = b.state + (b.current ^ 1);
    b.current ^= 1;
    const DpaFastStep::Arr3<const double> fa{f.data(), (long) f.stride_0(), (long) f.stride_1()};
    const int grid = DpaFastStep::grid_for(nlocal);
    auto launch = [&](auto chain) {
      constexpr int M = decltype(chain)::value;
      if (rmass.data())
        DpaFastStep::nhc_initial_kernel<1, M><<<grid, DpaFastStep::kBlock, 0, DpaFastStep::stream()>>>(
            nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v), fa, mask.data(), rmass.data(), mass.data(),
            type.data(), dtf, dtv, groupbit, in, out, p);
      else
        DpaFastStep::nhc_initial_kernel<0, M><<<grid, DpaFastStep::kBlock, 0, DpaFastStep::stream()>>>(
            nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v), fa, mask.data(), rmass.data(), mass.data(),
            type.data(), dtf, dtv, groupbit, in, out, p);
    };
    switch (mtchain) {
      case 1: launch(std::integral_constant<int, 1>{}); break;
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 3: launch(std::integral_constant<int, 3>{}); break;
      case 4: launch(std::integral_constant<int, 4>{}); break;
      default: launch(std::integral_constant<int, 5>{}); break;
    }
    atomKK->modified(execution_space,X_MASK | V_MASK);
  }
#endif
}

template<class DeviceType>
void FixNHKokkos<DeviceType>::nhc_device_final()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (DpaFastStep::is_cuda_device<DeviceType>()) {
    v_update_temp(false, true, false, false);

    // output and end_of_step consumers reuse the pre-scaling temperature
    // stored by compute_scalar(), as on the host path
    if (update->ntimestep == output->next || modify->n_end_of_step) {
      auto *tkk = static_cast<ComputeTempKokkos<DeviceType> *>(temperature);
      t_current = tkk->scalar_from_local_sum(DpaFastStep::read_result(DpaFastStep::scratch()));
    }

    v = atomKK->k_v.view<DeviceType>();
    mask = atomKK->k_mask.view<DeviceType>();
    int nlocal = atomKK->nlocal;
    if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;
    const DpaFastStep::NhcParams p = nhc_params();
    auto &b = DpaFastStep::nhc_buffers();
    const auto *in = b.state + b.current;
    auto *out = b.state + (b.current ^ 1);
    b.current ^= 1;
    const double *sum = DpaFastStep::scratch().result;
    auto launch = [&](auto chain) {
      constexpr int M = decltype(chain)::value;
      DpaFastStep::nhc_final_kernel<M><<<DpaFastStep::grid_for(nlocal), DpaFastStep::kBlock, 0,
                                         DpaFastStep::stream()>>>(
          nlocal, DpaFastStep::arr3(v), mask.data(), groupbit, sum, in, out, p);
    };
    switch (mtchain) {
      case 1: launch(std::integral_constant<int, 1>{}); break;
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 3: launch(std::integral_constant<int, 3>{}); break;
      case 4: launch(std::integral_constant<int, 4>{}); break;
      default: launch(std::integral_constant<int, 5>{}); break;
    }
    atomKK->modified(execution_space,V_MASK);
  }
#endif
}

/* ----------------------------------------------------------------------
   NVT initial_integrate: pending nh_v_temp + nve_v + nve_x
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::fused_nvt_initial()
{
  const bool scale = temp_scale_pending;
  temp_scale_pending = false;
  if (!DpaFastStep::verify()) {
    fused_nve_vx(scale, pending_factor_eta);
    return;
  }
  static DpaFastStep::VerifyStats stats("nvt initial_integrate dx dv");
  atomKK->sync(execution_space,X_MASK | V_MASK);
  auto xk = atomKK->k_x.view<DeviceType>();
  auto vk = atomKK->k_v.view<DeviceType>();
  decltype(xk) x0("x0",xk.extent(0)), x1("x1",xk.extent(0));
  decltype(vk) v0("v0",vk.extent(0)), v1("v1",vk.extent(0));
  Kokkos::deep_copy(x0,xk); Kokkos::deep_copy(v0,vk);
  fused_nve_vx(scale, pending_factor_eta);
  Kokkos::deep_copy(x1,xk); Kokkos::deep_copy(v1,vk);
  Kokkos::deep_copy(xk,x0); Kokkos::deep_copy(vk,v0);
  if (scale) {
    const double saved = factor_eta;
    factor_eta = pending_factor_eta;
    nh_v_temp();
    factor_eta = saved;
  }
  nve_v();
  nve_x();
  const int n = atomKK->nlocal;
  const double dx = max_abs_diff3<DeviceType>(n,xk,x1), dv = max_abs_diff3<DeviceType>(n,vk,v1);
  stats.add(dx != 0.0 || dv != 0.0, dx, dv);
}

/* ----------------------------------------------------------------------
   NPT initial_integrate: pending nh_v_temp + temperature->compute_scalar()
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::fused_scale_temp()
{
  auto *tkk = static_cast<ComputeTempKokkos<DeviceType> *>(temperature);
  const bool scale = temp_scale_pending;
  temp_scale_pending = false;
  if (!DpaFastStep::verify()) {
    tkk->scalar_from_local_sum(v_update_temp(scale,false,false));
    return;
  }
  static DpaFastStep::VerifyStats stats("npt initial temperature dv rel_dT");
  atomKK->sync(execution_space,V_MASK);
  auto vk = atomKK->k_v.view<DeviceType>();
  decltype(vk) v0("v0",vk.extent(0)), v1("v1",vk.extent(0));
  Kokkos::deep_copy(v0,vk);
  const double t_fast = tkk->scalar_from_local_sum(v_update_temp(scale,false,false));
  vk = atomKK->k_v.view<DeviceType>();
  Kokkos::deep_copy(v1,vk);
  Kokkos::deep_copy(vk,v0);
  if (scale) {
    const double saved = factor_eta;
    factor_eta = pending_factor_eta;
    nh_v_temp();
    factor_eta = saved;
  }
  atomKK->sync(temperature->execution_space,temperature->datamask_read);
  const double t_ref = temperature->compute_scalar();
  const double dv = max_abs_diff3<DeviceType>(atomKK->nlocal,vk,v1);
  const double rel = t_ref != 0.0 ? fabs(t_fast - t_ref) / fabs(t_ref) : fabs(t_fast);
  stats.add(dv != 0.0 || rel > 1e-13, dv, rel);
}

/* ----------------------------------------------------------------------
   NPT initial_integrate: nh_v_press + nve_v + remap + nve_x + remap
   box updates are computed on the host first, then applied per atom
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::fused_npt_initial()
{
  const bool check = DpaFastStep::verify();
  struct HostState { double lo[3], hi[3], xy, xz, yz, omega[6]; };
  auto save_state = [&]() {
    HostState s;
    for (int k = 0; k < 3; ++k) { s.lo[k] = domain->boxlo[k]; s.hi[k] = domain->boxhi[k]; }
    s.xy = domain->xy; s.xz = domain->xz; s.yz = domain->yz;
    for (int k = 0; k < 6; ++k) s.omega[k] = omega[k];
    return s;
  };
  auto restore_state = [&](const HostState &s) {
    for (int k = 0; k < 3; ++k) { domain->boxlo[k] = s.lo[k]; domain->boxhi[k] = s.hi[k]; }
    domain->xy = s.xy; domain->xz = s.xz; domain->yz = s.yz;
    for (int k = 0; k < 6; ++k) omega[k] = s.omega[k];
    domain->set_global_box();
    domain->set_local_box();
  };

  atomKK->sync(execution_space,X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
  auto xk = atomKK->k_x.view<DeviceType>();
  auto vk = atomKK->k_v.view<DeviceType>();
  decltype(xk) x0, x1;
  decltype(vk) v0, v1;
  HostState s0{};
  if (check) {
    x0 = decltype(xk)("x0",xk.extent(0)); x1 = decltype(xk)("x1",xk.extent(0));
    v0 = decltype(vk)("v0",vk.extent(0)); v1 = decltype(vk)("v1",vk.extent(0));
    Kokkos::deep_copy(x0,xk); Kokkos::deep_copy(v0,vk);
    s0 = save_state();
  }

  press_factors();
  NHBoxMap a_inv, b, b_inv, c;
  auto fill = [&](NHBoxMap &m, const double *mat) {
    for (int k = 0; k < 3; ++k) m.lo[k] = domain->boxlo[k];
    for (int k = 0; k < 6; ++k) m.m[k] = mat[k];
  };
  fill(a_inv, domain->h_inv);
  remap_box();
  fill(b, domain->h);
  fill(b_inv, domain->h_inv);
  remap_box();
  fill(c, domain->h);

  x = xk;
  v = vk;
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  const int nlocal = atomKK->nlocal;

  copymode = 1;
  if (rmass.data())
    nh_npt_vx<1,DeviceType>(nlocal,x,v,f,mask,rmass,type,factor[0],factor[1],factor[2],
                            dtf,dtv,groupbit,a_inv,b,b_inv,c);
  else
    nh_npt_vx<0,DeviceType>(nlocal,x,v,f,mask,mass,type,factor[0],factor[1],factor[2],
                            dtf,dtv,groupbit,a_inv,b,b_inv,c);
  copymode = 0;
  atomKK->modified(execution_space,X_MASK | V_MASK);

  if (check) {
    static DpaFastStep::VerifyStats stats("npt initial_integrate dx dv");
    const HostState s1 = save_state();
    Kokkos::deep_copy(x1,xk); Kokkos::deep_copy(v1,vk);
    Kokkos::deep_copy(xk,x0); Kokkos::deep_copy(vk,v0);
    restore_state(s0);
    nh_v_press();
    nve_v();
    remap();
    nve_x();
    remap();
    const HostState s2 = save_state();
    double dbox = 0.0;
    for (int k = 0; k < 3; ++k)
      dbox = MAX(dbox, MAX(fabs(s1.lo[k]-s2.lo[k]), fabs(s1.hi[k]-s2.hi[k])));
    const double dx = max_abs_diff3<DeviceType>(nlocal,xk,x1), dv = max_abs_diff3<DeviceType>(nlocal,vk,v1);
    stats.add(dx != 0.0 || dv != 0.0 || dbox != 0.0, dx, dv);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNHKokkos<DeviceType>::operator()(TagFixNH_nh_v_temp, const int &i) const {
  if (mask[i] & groupbit) {
    v(i,0) *= factor_eta;
    v(i,1) *= factor_eta;
    v(i,2) *= factor_eta;
  }
}

/* ----------------------------------------------------------------------
  if any tilt ratios exceed limits, set flip = 1 and compute new tilt values
  do not flip in x or y if non-periodic (can tilt but not flip)
    this is b/c the box length would be changed (dramatically) by flip
  if yz tilt exceeded, adjust C vector by one B vector
  if xz tilt exceeded, adjust C vector by one A vector
  if xy tilt exceeded, adjust B vector by one A vector
  check yz first since it may change xz, then xz check comes after
  if any flip occurs, create new box in domain
  image_flip() adjusts image flags due to box shape change induced by flip
  remap() puts atoms outside the new box back into the new box
  perform irregular on atoms in lamda coords to migrate atoms to new procs
  important that image_flip comes before remap, since remap may change
    image flags to new values, making eqs in doc of Domain:image_flip incorrect
------------------------------------------------------------------------- */

template<class DeviceType>
void FixNHKokkos<DeviceType>::pre_exchange()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;

  // flip is only triggered when tilt exceeds 0.5 by DELTAFLIP
  // this avoids immediate re-flipping due to tilt oscillations

  double xtiltmax = (0.5+DELTAFLIP)*xprd;
  double ytiltmax = (0.5+DELTAFLIP)*yprd;

  int flipxy,flipxz,flipyz;
  flipxy = flipxz = flipyz = 0;

  if (domain->yperiodic) {
    if (domain->yz < -ytiltmax) {
      domain->yz += yprd;
      domain->xz += domain->xy;
      flipyz = 1;
    } else if (domain->yz >= ytiltmax) {
      domain->yz -= yprd;
      domain->xz -= domain->xy;
      flipyz = -1;
    }
  }

  if (domain->xperiodic) {
    if (domain->xz < -xtiltmax) {
      domain->xz += xprd;
      flipxz = 1;
    } else if (domain->xz >= xtiltmax) {
      domain->xz -= xprd;
      flipxz = -1;
    }
    if (domain->xy < -xtiltmax) {
      domain->xy += xprd;
      flipxy = 1;
    } else if (domain->xy >= xtiltmax) {
      domain->xy -= xprd;
      flipxy = -1;
    }
  }

  int flip = 0;
  if (flipxy || flipxz || flipyz) flip = 1;

  if (flip) {
    domain->set_global_box();
    domain->set_local_box();

    domainKK->image_flip(flipxy,flipxz,flipyz);

    domainKK->remap_all();

    domainKK->x2lamda(atom->nlocal);
    atomKK->sync(Host,ALL_MASK);
    irregular->migrate_atoms();
    atomKK->modified(Host,ALL_MASK);
    domainKK->lamda2x(atom->nlocal);
  }
}

namespace LAMMPS_NS {
template class FixNHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNHKokkos<LMPHostType>;
#endif
}
