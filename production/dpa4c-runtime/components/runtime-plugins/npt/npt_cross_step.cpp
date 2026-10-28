#include "fix_npt_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "dpa_fast_step_kokkos.h"
#include "force.h"
#include "kspace.h"
#include "lammpsplugin.h"
#include "update.h"
#include "utils.h"
#include "version.h"

#include <cstdlib>

using namespace LAMMPS_NS;

namespace {

struct NPT_CROSS_STEPBoxMap {
  double lo[3];
  double m[6];
};

// Preserve the native FP64 operation order while folding the pending
// thermostat multiply into the already fused NPT velocity/remap pass.
template <int RMASS, class DeviceType, class X, class V, class F, class M,
          class R, class T>
void npt_cross_step_npt_vx(int n, X xx, V vv, F ff, M mm, R ms, T type,
                 bool two_scales, double temp_scale0, double temp_scale1,
                 double p0, double p1, double p2, double dtf, double dtv,
                 int gb, NPT_CROSS_STEPBoxMap ai, NPT_CROSS_STEPBoxMap b, NPT_CROSS_STEPBoxMap bi,
                 NPT_CROSS_STEPBoxMap c) {
  Kokkos::parallel_for(
      "FixNHKokkos::npt_cross_step_npt_vx", Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int atom_mask = mm[i];
        if (atom_mask & gb) {
          if (two_scales) {
            vv(i, 0) *= temp_scale0;
            vv(i, 1) *= temp_scale0;
            vv(i, 2) *= temp_scale0;
          }
          vv(i, 0) *= temp_scale1;
          vv(i, 1) *= temp_scale1;
          vv(i, 2) *= temp_scale1;
          vv(i, 0) *= p0;
          vv(i, 1) *= p1;
          vv(i, 2) *= p2;
          vv(i, 0) *= p0;
          vv(i, 1) *= p1;
          vv(i, 2) *= p2;
          const F_FLOAT dtfm = dtf / (RMASS ? ms[i] : ms[type[i]]);
          vv(i, 0) += dtfm * ff(i, 0);
          vv(i, 1) += dtfm * ff(i, 1);
          vv(i, 2) += dtfm * ff(i, 2);
        }
        {
          F_FLOAT delta[3];
          delta[0] = xx(i, 0) - ai.lo[0];
          delta[1] = xx(i, 1) - ai.lo[1];
          delta[2] = xx(i, 2) - ai.lo[2];
          xx(i, 0) =
              ai.m[0] * delta[0] + ai.m[5] * delta[1] + ai.m[4] * delta[2];
          xx(i, 1) = ai.m[1] * delta[1] + ai.m[3] * delta[2];
          xx(i, 2) = ai.m[2] * delta[2];
          const double xi1 = xx(i, 1);
          const double xi2 = xx(i, 2);
          xx(i, 0) = b.m[0] * xx(i, 0) + b.m[5] * xi1 + b.m[4] * xi2 +
                     b.lo[0];
          xx(i, 1) = b.m[1] * xi1 + b.m[3] * xi2 + b.lo[1];
          xx(i, 2) = b.m[2] * xi2 + b.lo[2];
        }
        if (atom_mask & gb) {
          xx(i, 0) += dtv * vv(i, 0);
          xx(i, 1) += dtv * vv(i, 1);
          xx(i, 2) += dtv * vv(i, 2);
        }
        {
          F_FLOAT delta[3];
          delta[0] = xx(i, 0) - bi.lo[0];
          delta[1] = xx(i, 1) - bi.lo[1];
          delta[2] = xx(i, 2) - bi.lo[2];
          xx(i, 0) =
              bi.m[0] * delta[0] + bi.m[5] * delta[1] + bi.m[4] * delta[2];
          xx(i, 1) = bi.m[1] * delta[1] + bi.m[3] * delta[2];
          xx(i, 2) = bi.m[2] * delta[2];
          const double xi1 = xx(i, 1);
          const double xi2 = xx(i, 2);
          xx(i, 0) = c.m[0] * xx(i, 0) + c.m[5] * xi1 + c.m[4] * xi2 +
                     c.lo[0];
          xx(i, 1) = c.m[1] * xi1 + c.m[3] * xi2 + c.lo[1];
          xx(i, 2) = c.m[2] * xi2 + c.lo[2];
        }
      });
}

class FixNPTCrossStepKokkos : public FixNPTKokkos<LMPDeviceType> {
 public:
  using Base = FixNPTKokkos<LMPDeviceType>;

  FixNPTCrossStepKokkos(LAMMPS *lmp, int narg, char **arg)
      : Base(lmp, narg, arg) {}

  void init() override {
    Base::init();
    const char *setting = std::getenv("DPA_NPT_CROSS_STEP_NPT_CROSS_STEP");
    const bool requested = setting && std::atoi(setting) != 0;
    eligible_ = requested && atom->natoms <= 4096 &&
                DpaFastStep::enabled(DpaFastStep::kInitial) &&
                fast_step_allowed() && fast_npt_allowed() &&
                nc_tchain == 1 && tstat_flag && pstat_flag && !nhc_device;
    // The verification path intentionally replays the unfused operations;
    // do not install a second verifier inside this plugin.
    if (DpaFastStep::verify()) eligible_ = false;
  }

  void setup(int vflag) override {
    Base::setup(vflag);
    fuse_integrate_flag = eligible_ ? 1 : 0;
  }

  void initial_integrate(int vflag) override {
    if (!eligible_) {
      Base::initial_integrate(vflag);
      return;
    }

    static bool reported = false;
    if (!reported && comm->me == 0) {
      reported = true;
      utils::logmesg(lmp,
          "NPT_CROSS_STEP_NPT_CROSS_STEP active=1 atoms={} policy=initial_scale_scalar\n",
          atom->natoms);
    }

    initial_with_pending_scales(false, 1.0);
  }

  void fused_integrate(int vflag) override {
    if (!eligible_) {
      Base::final_integrate();
      Base::initial_integrate(vflag);
      return;
    }

    static bool reported = false;
    if (!reported && comm->me == 0) {
      reported = true;
      utils::logmesg(lmp,
          "NPT_CROSS_STEP_NPT_CROSS_STEP fused=1 atoms={} policy=two_thermostat_scales\n",
          atom->natoms);
    }

    // Defer only the final thermostat velocity multiply.  The final kick,
    // pressure scaling, temperature reduction and all host scalar recurrences
    // retain their native order.
    defer_temp_scale = true;
    Base::final_integrate();
    defer_temp_scale = false;
    const double final_scale = pending_factor_eta;
    initial_with_pending_scales(true, final_scale);
  }

 private:
  bool eligible_ = false;

  void initial_with_pending_scales(bool two_scales, double final_scale) {

    // Original FixNH order.  nhc_temp_integrate() already advances
    // t_current by factor_eta^2.  Apply the identical per-atom velocity
    // scale, then publish that scalar to the pressure compute instead of
    // reducing the just-scaled velocities back to the host a second time.
    if (mpchain) nhc_press_integrate();
    compute_temp_target();
    defer_temp_scale = true;
    nhc_temp_integrate();
    // Deferral applies only to this initial half-step.  The final half-step
    // must use the native thermostat kernel, matching FixNHKokkos.
    defer_temp_scale = false;

    temperature->invoked_scalar = update->ntimestep;
    temperature->scalar = t_current;

    atomKK->sync(pressure->execution_space, pressure->datamask_read);
    pressure->compute_scalar();
    atomKK->modified(temperature->execution_space,
                     temperature->datamask_modify);
    atomKK->modified(pressure->execution_space, pressure->datamask_modify);
    atomKK->sync(execution_space, temperature->datamask_modify);
    atomKK->sync(execution_space, pressure->datamask_modify);
    couple();
    pressure->addstep(update->ntimestep + 1);

    compute_press_target();
    nh_omega_dot();
    const double initial_scale = pending_factor_eta;
    temp_scale_pending = false;
    fused_npt_initial_with_scales(two_scales, final_scale, initial_scale);
    if (kspace_flag) force->kspace->setup();
  }

  void fused_npt_initial_with_scales(bool two_scales, double temp_scale0,
                                     double temp_scale1) {
    atomKK->sync(execution_space,
                 X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK |
                     TYPE_MASK);
    auto xk = atomKK->k_x.view<LMPDeviceType>();
    auto vk = atomKK->k_v.view<LMPDeviceType>();

    press_factors();
    NPT_CROSS_STEPBoxMap a_inv, b, b_inv, c;
    auto fill = [&](NPT_CROSS_STEPBoxMap &map, const double *matrix) {
      for (int k = 0; k < 3; ++k) map.lo[k] = domain->boxlo[k];
      for (int k = 0; k < 6; ++k) map.m[k] = matrix[k];
    };
    fill(a_inv, domain->h_inv);
    remap_box();
    fill(b, domain->h);
    fill(b_inv, domain->h_inv);
    remap_box();
    fill(c, domain->h);

    x = xk;
    v = vk;
    f = atomKK->k_f.view<LMPDeviceType>();
    rmass = atomKK->k_rmass.view<LMPDeviceType>();
    mass = atomKK->k_mass.view<LMPDeviceType>();
    type = atomKK->k_type.view<LMPDeviceType>();
    mask = atomKK->k_mask.view<LMPDeviceType>();
    const int nlocal = atomKK->nlocal;

    copymode = 1;
    if (rmass.data())
      npt_cross_step_npt_vx<1, LMPDeviceType>(
          nlocal, x, v, f, mask, rmass, type, two_scales, temp_scale0,
          temp_scale1, factor[0], factor[1], factor[2], dtf, dtv, groupbit,
          a_inv, b, b_inv, c);
    else
      npt_cross_step_npt_vx<0, LMPDeviceType>(
          nlocal, x, v, f, mask, mass, type, two_scales, temp_scale0,
          temp_scale1, factor[0], factor[1], factor[2], dtf, dtv, groupbit,
          a_inv, b, b_inv, c);
    copymode = 0;
    atomKK->modified(execution_space, X_MASK | V_MASK);
  }
};

Fix *make_fix(LAMMPS *lmp, int narg, char **arg) {
  return new FixNPTCrossStepKokkos(lmp, narg, arg);
}

}  // namespace

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "npt/kk";
  plugin.info = "NPT_CROSS_STEP small NPT cross-step thermostat scale fusion";
  plugin.author = "DPA optimization";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&make_fix);
  plugin.handle = handle;
  register_plugin(&plugin, lmp);
  plugin.name = "npt/kk/device";
  register_plugin(&plugin, lmp);
}
