// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_FIX_NH_KOKKOS_H
#define LMP_FIX_NH_KOKKOS_H

#include "fix_nh.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {
namespace DpaFastStep { struct NhcParams; }

template<int TRICLINIC_FLAG>
struct TagFixNH_nh_v_press{};

template<int RMASS>
struct TagFixNH_nve_v{};

struct TagFixNH_nve_x{};

struct TagFixNH_nh_v_temp{};

template<class DeviceType>
class FixNHKokkos : public FixNH {
 public:
  typedef DeviceType device_type;

  FixNHKokkos(class LAMMPS *, int, char **);

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void pre_exchange() override;

  template<int TRICLINIC_FLAG>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_press<TRICLINIC_FLAG>, const int&) const;

  template<int RMASS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nve_v<RMASS>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nve_x, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_temp, const int&) const;

 protected:
  void remap() override;

  void nve_x() override;            // may be overwritten by child classes
  void nve_v() override;
  void nh_v_press() override;
  void nh_v_temp() override;

  void remap_box();

  bool fast_step_allowed() const;
  bool fast_npt_allowed() const;
  bool fast_nve_v_temp();
  void fused_nve_vx(bool scale, double scale_factor);
  void fused_nvt_initial();
  void fused_scale_temp();
  void fused_npt_initial();
  double v_update_temp(bool scale, bool nve, bool press, bool read = true);

  // DPA_LMP_FAST_STEP bit 16: NVT chain state lives on the device during a
  // run. "run ... pre no" skips setup() but not post_run(), so post_run()
  // copies the state to the host and the next run uploads it again.
  bool nhc_device = false;
  bool nhc_upload_pending = false;
  bool nhc_device_allowed() const;
  void nhc_upload();
  void nhc_download();
  void nhc_device_initial();
  void nhc_device_final();
  DpaFastStep::NhcParams nhc_params();
  void post_run() override;
  void press_factors();

  // set while initial_integrate() defers the thermostat scaling into the
  // fused velocity/position update
  bool defer_temp_scale = false;
  bool temp_scale_pending = false;
  double pending_factor_eta = 1.0;

  F_FLOAT factor[3];

  class DomainKokkos *domainKK;

  typename ArrayTypes<DeviceType>::t_x_array x;
  typename ArrayTypes<DeviceType>::t_v_array v;
  typename ArrayTypes<DeviceType>::t_f_array_const f;
  typename ArrayTypes<DeviceType>::t_float_1d rmass;
  typename ArrayTypes<DeviceType>::t_float_1d mass;
  typename ArrayTypes<DeviceType>::t_int_1d type;
  typename ArrayTypes<DeviceType>::t_int_1d mask;
};

}

#endif
