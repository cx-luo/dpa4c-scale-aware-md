#include "small_epoch_abi.h"
#include "modify.h"
#include "modify_kokkos.h"
#include "neighbor.h"
#include "output.h"
#include "timer.h"

#include <cstring>
#include "fix_npt_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "compute_temp_kokkos.h"
#include "domain.h"
#include "dpa_fast_step_kokkos.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "lammpsplugin.h"
#include "update.h"
#include "utils.h"
#include "version.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>

using namespace LAMMPS_NS;

extern "C" {
void *dpa_small_epoch_owner = nullptr;
}

namespace {

constexpr int kChain = 3;

struct BoxMap {
  double lo[3];
  double m[6];
};

struct CoroutinePayload {
  double temp_final;
  double temp_initial;
  double press[3];
  BoxMap old_inv;
  BoxMap middle;
  BoxMap middle_inv;
  BoxMap final_box;
};

struct alignas(64) CoroutineBridge {
  volatile int phase;
  int pad0[15];
  double kinetic_sum;
  CoroutinePayload payload;
};

template <int RMASS>
__global__ void __launch_bounds__(DpaFastStep::kBlock)
small_npt_coroutine_kernel(
    int n, int ni, DpaFastStep::Arr3<double> x,
    DpaFastStep::Arr3<double> v, DpaFastStep::Arr3<const double> f,
    const int *mask, const double *rmass, const double *mass,
    const int *type, double dtf, double dtv, int groupbit, int tgroupbit,
    double final_p0, double final_p1, double final_p2, double *partial,
    unsigned *ticket, CoroutineBridge *bridge) {
  __shared__ double reduction[32];
  __shared__ CoroutinePayload payload;
  __shared__ bool last_block;

  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    const int atom_mask = mask[i];
    const double massone = RMASS ? rmass[i] : mass[type[i]];
    if (atom_mask & groupbit) {
      const double dtfm = dtf / massone;
      v(i, 0) += dtfm * f(i, 0);
      v(i, 1) += dtfm * f(i, 1);
      v(i, 2) += dtfm * f(i, 2);
      v(i, 0) *= final_p0;
      v(i, 1) *= final_p1;
      v(i, 2) *= final_p2;
      v(i, 0) *= final_p0;
      v(i, 1) *= final_p1;
      v(i, 2) *= final_p2;
    }
    if (atom_mask & tgroupbit)
      acc += (v(i, 0) * v(i, 0) + v(i, 1) * v(i, 1) +
              v(i, 2) * v(i, 2)) * massone;
  }

  const double block_total = DpaFastStep::block_sum(acc, reduction);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = block_total;
    __threadfence();
    last_block = atomicAdd(ticket, 1u) == gridDim.x - 1;
  }
  __syncthreads();

  // The block that publishes the last partial becomes the reducer.  This
  // keeps the fixed-tree result ordering while avoiding a cooperative launch
  // and its grid-wide barrier.  Small-N eligibility keeps the whole grid
  // resident while the blocks wait for the host response below.
  if (last_block) {
    double total = 0.0;
    for (int k = threadIdx.x; k < static_cast<int>(gridDim.x);
         k += blockDim.x)
      total += partial[k];
    total = DpaFastStep::block_sum(total, reduction);
    if (threadIdx.x == 0) {
      *ticket = 0u;
      bridge->kinetic_sum = total;
      __threadfence_system();
      bridge->phase = 1;
      __threadfence_system();
    }
  }

  if (threadIdx.x == 0) {
    volatile int *phase = &bridge->phase;
    while (*phase != 2) { }
    __threadfence_system();
    payload = bridge->payload;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < ni; i += gridDim.x * blockDim.x) {
    const int atom_mask = mask[i];
    if (atom_mask & groupbit) {
      v(i, 0) *= payload.temp_final;
      v(i, 1) *= payload.temp_final;
      v(i, 2) *= payload.temp_final;
      v(i, 0) *= payload.temp_initial;
      v(i, 1) *= payload.temp_initial;
      v(i, 2) *= payload.temp_initial;
      v(i, 0) *= payload.press[0];
      v(i, 1) *= payload.press[1];
      v(i, 2) *= payload.press[2];
      v(i, 0) *= payload.press[0];
      v(i, 1) *= payload.press[1];
      v(i, 2) *= payload.press[2];
      const double dtfm = dtf / (RMASS ? rmass[i] : mass[type[i]]);
      v(i, 0) += dtfm * f(i, 0);
      v(i, 1) += dtfm * f(i, 1);
      v(i, 2) += dtfm * f(i, 2);
    }

    double dx = x(i, 0) - payload.old_inv.lo[0];
    double dy = x(i, 1) - payload.old_inv.lo[1];
    double dz = x(i, 2) - payload.old_inv.lo[2];
    x(i, 0) = payload.old_inv.m[0] * dx;
    x(i, 1) = payload.old_inv.m[1] * dy;
    x(i, 2) = payload.old_inv.m[2] * dz;
    x(i, 0) = payload.middle.m[0] * x(i, 0) + payload.middle.lo[0];
    x(i, 1) = payload.middle.m[1] * x(i, 1) + payload.middle.lo[1];
    x(i, 2) = payload.middle.m[2] * x(i, 2) + payload.middle.lo[2];
    if (atom_mask & groupbit) {
      x(i, 0) += dtv * v(i, 0);
      x(i, 1) += dtv * v(i, 1);
      x(i, 2) += dtv * v(i, 2);
    }
    dx = x(i, 0) - payload.middle_inv.lo[0];
    dy = x(i, 1) - payload.middle_inv.lo[1];
    dz = x(i, 2) - payload.middle_inv.lo[2];
    x(i, 0) = payload.middle_inv.m[0] * dx;
    x(i, 1) = payload.middle_inv.m[1] * dy;
    x(i, 2) = payload.middle_inv.m[2] * dz;
    x(i, 0) = payload.final_box.m[0] * x(i, 0) + payload.final_box.lo[0];
    x(i, 1) = payload.final_box.m[1] * x(i, 1) + payload.final_box.lo[1];
    x(i, 2) = payload.final_box.m[2] * x(i, 2) + payload.final_box.lo[2];
  }
}
template<int RMASS>
__global__ void small_epoch_initial_kernel(int ni,DpaFastStep::Arr3<double> x,
    DpaFastStep::Arr3<double> v,DpaFastStep::Arr3<const double> f,
    const int *mask,const double *rmass,const double *mass,const int *type,
    double dtf,double dtv,int groupbit,CoroutinePayload payload) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < ni; i += gridDim.x * blockDim.x) {
    const int atom_mask = mask[i];
    if (atom_mask & groupbit) {
      v(i, 0) *= payload.temp_final;
      v(i, 1) *= payload.temp_final;
      v(i, 2) *= payload.temp_final;
      v(i, 0) *= payload.temp_initial;
      v(i, 1) *= payload.temp_initial;
      v(i, 2) *= payload.temp_initial;
      v(i, 0) *= payload.press[0];
      v(i, 1) *= payload.press[1];
      v(i, 2) *= payload.press[2];
      v(i, 0) *= payload.press[0];
      v(i, 1) *= payload.press[1];
      v(i, 2) *= payload.press[2];
      const double dtfm = dtf / (RMASS ? rmass[i] : mass[type[i]]);
      v(i, 0) += dtfm * f(i, 0);
      v(i, 1) += dtfm * f(i, 1);
      v(i, 2) += dtfm * f(i, 2);
    }

    double dx = x(i, 0) - payload.old_inv.lo[0];
    double dy = x(i, 1) - payload.old_inv.lo[1];
    double dz = x(i, 2) - payload.old_inv.lo[2];
    x(i, 0) = payload.old_inv.m[0] * dx;
    x(i, 1) = payload.old_inv.m[1] * dy;
    x(i, 2) = payload.old_inv.m[2] * dz;
    x(i, 0) = payload.middle.m[0] * x(i, 0) + payload.middle.lo[0];
    x(i, 1) = payload.middle.m[1] * x(i, 1) + payload.middle.lo[1];
    x(i, 2) = payload.middle.m[2] * x(i, 2) + payload.middle.lo[2];
    if (atom_mask & groupbit) {
      x(i, 0) += dtv * v(i, 0);
      x(i, 1) += dtv * v(i, 1);
      x(i, 2) += dtv * v(i, 2);
    }
    dx = x(i, 0) - payload.middle_inv.lo[0];
    dy = x(i, 1) - payload.middle_inv.lo[1];
    dz = x(i, 2) - payload.middle_inv.lo[2];
    x(i, 0) = payload.middle_inv.m[0] * dx;
    x(i, 1) = payload.middle_inv.m[1] * dy;
    x(i, 2) = payload.middle_inv.m[2] * dz;
    x(i, 0) = payload.final_box.m[0] * x(i, 0) + payload.final_box.lo[0];
    x(i, 1) = payload.final_box.m[1] * x(i, 1) + payload.final_box.lo[1];
    x(i, 2) = payload.final_box.m[2] * x(i, 2) + payload.final_box.lo[2];
  }
}


template <int RMASS>
bool launch_coroutine(
    int n, int ni, DpaFastStep::Arr3<double> x,
    DpaFastStep::Arr3<double> v, DpaFastStep::Arr3<const double> f,
    const int *mask, const double *rmass, const double *mass,
    const int *type, double dtf, double dtv, int groupbit, int tgroupbit,
    const double final_press[3], double *partial, unsigned *ticket,
    CoroutineBridge *bridge) {
  const int blocks = DpaFastStep::grid_for(n);
  static bool reported = false;
  if (!reported) {
    reported = true;
    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, small_npt_coroutine_kernel<RMASS>);
    std::fprintf(stderr,
                 "DPA_SMALL_NPT_COROUTINE_RESOURCE rmass=%d blocks=%d "
                 "regs=%d local=%zu shared=%zu\n",
                 RMASS, blocks, attr.numRegs,
                 attr.localSizeBytes, attr.sharedSizeBytes);
  }
  double p0 = final_press[0], p1 = final_press[1], p2 = final_press[2];
  small_npt_coroutine_kernel<RMASS>
      <<<blocks, DpaFastStep::kBlock, 0, DpaFastStep::stream()>>>(
          n, ni, x, v, f, mask, rmass, mass, type, dtf, dtv, groupbit,
          tgroupbit, p0, p1, p2, partial, ticket, bridge);
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    std::fprintf(stderr, "DPA_SMALL_NPT_COROUTINE_LAUNCH_FAILED %s\n",
                 cudaGetErrorString(status));
    cudaGetLastError();
    return false;
  }
  return true;
}

template <int RMASS, class DeviceType, class X, class V, class F, class M,
          class R, class T>
void fallback_npt_vx(int n, X xx, V vv, F ff, M mm, R ms, T type,
                     bool two_scales, double ts0, double ts1,
                     double p0, double p1, double p2, double dtf, double dtv,
                     int gb, BoxMap ai, BoxMap b, BoxMap bi, BoxMap c) {
  Kokkos::parallel_for(
      "FixNHKokkos::small_npt_fallback_npt_vx",
      Kokkos::RangePolicy<DeviceType>(0, n),
      KOKKOS_LAMBDA(const int i) {
        const int am = mm[i];
        if (am & gb) {
          if (two_scales) {
            vv(i, 0) *= ts0; vv(i, 1) *= ts0; vv(i, 2) *= ts0;
          }
          vv(i, 0) *= ts1; vv(i, 1) *= ts1; vv(i, 2) *= ts1;
          vv(i, 0) *= p0; vv(i, 1) *= p1; vv(i, 2) *= p2;
          vv(i, 0) *= p0; vv(i, 1) *= p1; vv(i, 2) *= p2;
          const double dm = dtf / (RMASS ? ms[i] : ms[type[i]]);
          vv(i, 0) += dm * ff(i, 0);
          vv(i, 1) += dm * ff(i, 1);
          vv(i, 2) += dm * ff(i, 2);
        }
        double dx = xx(i, 0) - ai.lo[0];
        double dy = xx(i, 1) - ai.lo[1];
        double dz = xx(i, 2) - ai.lo[2];
        xx(i, 0) = ai.m[0] * dx; xx(i, 1) = ai.m[1] * dy; xx(i, 2) = ai.m[2] * dz;
        xx(i, 0) = b.m[0] * xx(i, 0) + b.lo[0];
        xx(i, 1) = b.m[1] * xx(i, 1) + b.lo[1];
        xx(i, 2) = b.m[2] * xx(i, 2) + b.lo[2];
        if (am & gb) {
          xx(i, 0) += dtv * vv(i, 0); xx(i, 1) += dtv * vv(i, 1); xx(i, 2) += dtv * vv(i, 2);
        }
        dx = xx(i, 0) - bi.lo[0]; dy = xx(i, 1) - bi.lo[1]; dz = xx(i, 2) - bi.lo[2];
        xx(i, 0) = bi.m[0] * dx; xx(i, 1) = bi.m[1] * dy; xx(i, 2) = bi.m[2] * dz;
        xx(i, 0) = c.m[0] * xx(i, 0) + c.lo[0];
        xx(i, 1) = c.m[1] * xx(i, 1) + c.lo[1];
        xx(i, 2) = c.m[2] * xx(i, 2) + c.lo[2];
      });
}

class FixNPTCoroutineKokkos : public FixNPTKokkos<LMPDeviceType> {
 public:
  using Base = FixNPTKokkos<LMPDeviceType>;

  FixNPTCoroutineKokkos(LAMMPS *lmp, int narg, char **arg)
      : Base(lmp, narg, arg) {}

  ~FixNPTCoroutineKokkos() override {
    if(epoch_host_)std::fprintf(stderr,"DPA_SMALL_EPOCH_PHASES produced=%lu consumed=%lu pending=%d\n",epoch_produced_,epoch_consumed_,epoch_pending_);
    if(epoch_event_)cudaEventDestroy(epoch_event_);
    if(epoch_device_)cudaFree(epoch_device_);
    if(epoch_host_)cudaFreeHost(epoch_host_);
    if (epoch_owner_ == this) {
      epoch_owner_ = nullptr;
      dpa_small_epoch_owner = nullptr;
    }
    if (host_bridge_) cudaFreeHost(host_bridge_);
  }

  void init() override {
    Base::init();
    const char *setting = std::getenv("DPA_SMALL_NPT_COROUTINE");
    const bool requested = setting && std::atoi(setting) != 0;
    const char *epoch_setting = std::getenv("DPA_SMALL_EPOCH_PHASES");
    const bool epoch_requested =
        epoch_setting && std::atoi(epoch_setting) != 0;
    long epoch_max_atoms = 4096;
    if (const char *max_setting =
            std::getenv("DPA_NPT_FINITE_EPOCH_MAX_ATOMS")) {
      const long parsed = std::atol(max_setting);
      if (parsed > 0) epoch_max_atoms = parsed;
    }
    auto *tkk = dynamic_cast<ComputeTempKokkos<LMPDeviceType> *>(temperature);
    const bool common_eligible =
        requested && comm->nprocs == 1 && fast_step_allowed() &&
        fast_npt_allowed() && DpaFastStep::enabled(DpaFastStep::kInitial) &&
        DpaFastStep::enabled(DpaFastStep::kFinal) && tstat_flag &&
        pstat_flag && !nhc_device && nc_tchain == 1 && nc_pchain == 1 &&
        mtchain == kChain && mpchain == kChain && t_start == t_stop &&
        p_start[0] == p_stop[0] && p_flag[0] && p_flag[1] && p_flag[2] &&
        !p_flag[3] && !p_flag[4] && !p_flag[5] && !deviatoric_flag &&
        !kspace_flag && tkk && !tkk->dynamic_dof() &&
        temperature->dof > 0.0;
    eligible_ = common_eligible && atom->natoms <= 4096;
    epoch_eligible_ = common_eligible && epoch_requested &&
                      atom->natoms <= epoch_max_atoms;
    if (DpaFastStep::verify()) {
      eligible_ = false;
      epoch_eligible_ = false;
    }
    if (eligible_) {
      cudaHostAlloc(&host_bridge_, sizeof(CoroutineBridge),
                    cudaHostAllocMapped | cudaHostAllocPortable);
      cudaHostGetDevicePointer(&device_bridge_, host_bridge_, 0);
      host_bridge_->phase = 0;
    }
    if (epoch_eligible_) {
      cudaHostAlloc(&epoch_host_, sizeof(DpaSmallEpoch::Ev),
                    cudaHostAllocPortable);
      cudaMalloc(&epoch_device_, sizeof(DpaSmallEpoch::Ev));
      cudaEventCreateWithFlags(&epoch_event_, cudaEventDisableTiming);
      epoch_owner_ = this;
      dpa_small_epoch_owner = lmp;
      if (comm->me == 0)
        utils::logmesg(
            lmp,
            "DPA_NPT_FINITE_EPOCH eligible=1 atoms={} mapped_coroutine={} "
            "max_atoms={}\n",
            atom->natoms, eligible_ ? 1 : 0, epoch_max_atoms);
    }
  }

  void setup(int vflag) override {
    Base::setup(vflag);
    fuse_integrate_flag = (eligible_ || epoch_eligible_) ? 1 : 0;
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
          "DPA_SMALL_NPT_COROUTINE active=1 atoms={} mtchain={} mpchain={} policy=atomic_ticket_mapped_coroutine\n",
          atom->natoms, mtchain, mpchain);
    }
    initial_with_pending_scales(false, 1.0);
  }

  void fused_integrate(int vflag) override {
    if (epoch_pending_) {
      consume_epoch();
      return;
    }
    if (!eligible_) {
      Base::final_integrate();
      Base::initial_integrate(vflag);
      return;
    }
    static bool reported = false;
    if (!reported && comm->me == 0) {
      reported = true;
      utils::logmesg(lmp,
          "DPA_SMALL_NPT_COROUTINE fused=1 atoms={} policy=atomic_ticket_mapped_coroutine\n",
          atom->natoms);
    }
    if (coroutine_step()) return;

    defer_temp_scale = true;
    Base::final_integrate();
    defer_temp_scale = false;
    const double final_scale = pending_factor_eta;
    initial_with_pending_scales(true, final_scale);
  }


 public:
  inline static FixNPTCoroutineKokkos *epoch_owner_=nullptr;
  bool prepare_epoch(void *owner,double *virial,double *energy,double scale,
      bool do_e,bool do_v,DpaSmallEpoch::Args *a,DpaSmallEpoch::Ev **out) {
    if(owner!=lmp || !epoch_eligible_ || !epoch_host_ || epoch_pending_ || update->setupflag ||
       update->ntimestep>=update->endstep || update->ntimestep==output->next ||
       modify->n_end_of_step || modify->n_pre_reverse || modify->n_post_integrate ||
       !do_v || neighbor->ago==0 || DpaFastStep::verify()) return false;
    if(lmp->timer->has_timeout() || !static_cast<ModifyKokkos*>(modify)->check_fuse_integrate()) return false;
    // A post-force observer consumes pre-final-kick velocities. Never move
    // the kick across a selected capture or an unknown fix dependency.
    for(auto *fix:modify->get_fix_list()) {
      if(fix==this)continue;
      if(std::strcmp(fix->style,"dpa/force-observer") ||
         (fix->datamask_read & (V_MASK|ENERGY_MASK|VIRIAL_MASK)) ||
         (fix->datamask_modify & F_MASK)) return false;
    }
    auto &check=DpaFastStep::scratch();
    if(check.stamp<=0 || !check.flag_ready || !check.host_flag ||
       cudaEventQuery(check.flag_ready)!=cudaSuccess || *check.host_flag==check.stamp) return false;
    // All ownership/view synchronization happens BEFORE producer submission.
    atomKK->sync(execution_space,V_MASK|MASK_MASK|RMASS_MASK|TYPE_MASK);
    atomKK->k_mass.sync<LMPDeviceType>();
    press_factors();
    auto vk=atomKK->k_v.view<LMPDeviceType>();
    *a={}; a->velocity=DpaFastStep::arr3(vk);
    a->mask=atomKK->k_mask.view<LMPDeviceType>().data();
    a->type=atomKK->k_type.view<LMPDeviceType>().data();
    a->rmass=atomKK->k_rmass.view<LMPDeviceType>().data();
    a->mass=atomKK->k_mass.view<LMPDeviceType>().data();
    a->dtf=dtf; a->groupbit=groupbit; a->tgroupbit=temperature->groupbit;
    for(int k=0;k<3;++k)a->press[k]=factor[k];
    *out=epoch_device_; epoch_virial_=virial; epoch_energy_=energy;
    epoch_scale_=scale; epoch_do_e_=do_e; epoch_do_v_=do_v;
    return true;
  }
  void publish_epoch(cudaStream_t stream) {
    cudaMemcpyAsync(epoch_host_,epoch_device_,sizeof(DpaSmallEpoch::Ev),cudaMemcpyDeviceToHost,stream);
    cudaEventRecord(epoch_event_,stream); epoch_pending_=true; ++epoch_produced_;
  }
 private:
  DpaSmallEpoch::Ev *epoch_host_=nullptr,*epoch_device_=nullptr;
  cudaEvent_t epoch_event_=nullptr;
  bool epoch_pending_=false,epoch_do_e_=false,epoch_do_v_=false;
  double *epoch_virial_=nullptr,*epoch_energy_=nullptr,epoch_scale_=1;
  unsigned long epoch_produced_=0,epoch_consumed_=0;
  void consume_epoch() {
    // This event depends only on finite kernels/copies, never a host waiter.
    cudaEventSynchronize(epoch_event_);
    if(epoch_do_e_)*epoch_energy_+=epoch_scale_*epoch_host_->e;
    if(epoch_do_v_)for(int k=0;k<6;++k)epoch_virial_[k]+=epoch_scale_*epoch_host_->v[k];
    CoroutinePayload payload{};host_resume(epoch_host_->kinetic,payload);
    auto xx=atomKK->k_x.view<LMPDeviceType>();auto vv=atomKK->k_v.view<LMPDeviceType>();
    auto ff=atomKK->k_f.view<LMPDeviceType>();
    DpaFastStep::Arr3<const double> fv{ff.data(),static_cast<long>(ff.stride_0()),static_cast<long>(ff.stride_1())};
    const int blocks=DpaFastStep::grid_for(atomKK->nlocal);
    auto rm=atomKK->k_rmass.view<LMPDeviceType>();auto ms=atomKK->k_mass.view<LMPDeviceType>();
    auto tp=atomKK->k_type.view<LMPDeviceType>();auto ma=atomKK->k_mask.view<LMPDeviceType>();
    if(rm.data())small_epoch_initial_kernel<1><<<blocks,DpaFastStep::kBlock,0,DpaFastStep::stream()>>>(
      atomKK->nlocal,DpaFastStep::arr3(xx),DpaFastStep::arr3(vv),fv,ma.data(),rm.data(),ms.data(),tp.data(),dtf,dtv,groupbit,payload);
    else small_epoch_initial_kernel<0><<<blocks,DpaFastStep::kBlock,0,DpaFastStep::stream()>>>(
      atomKK->nlocal,DpaFastStep::arr3(xx),DpaFastStep::arr3(vv),fv,ma.data(),rm.data(),ms.data(),tp.data(),dtf,dtv,groupbit,payload);
    if(cudaPeekAtLastError()!=cudaSuccess)error->all(FLERR,"small epoch initial launch failed");
    epoch_pending_=false;++epoch_consumed_;
    if(epoch_consumed_==1)std::fprintf(stderr,"DPA_SMALL_EPOCH_PHASES active=1 policy=finite_producer_native_npt_consumer\n");
    atomKK->modified(execution_space,X_MASK|V_MASK);
  }
  bool eligible_ = false;
  bool epoch_eligible_ = false;
  CoroutineBridge *host_bridge_ = nullptr;
  CoroutineBridge *device_bridge_ = nullptr;

  void host_resume(double local_sum, CoroutinePayload &payload) {
    auto *tkk = static_cast<ComputeTempKokkos<LMPDeviceType> *>(temperature);
    t_current = tkk->scalar_from_local_sum(local_sum);
    tdof = temperature->dof;
    temperature->invoked_scalar = update->ntimestep;
    temperature->scalar = t_current;
    pressure->compute_scalar();
    couple();
    pressure->addstep(update->ntimestep + 1);
    nh_omega_dot();

    defer_temp_scale = true;
    nhc_temp_integrate();
    defer_temp_scale = false;
    payload.temp_final = pending_factor_eta;
    if (mpchain) nhc_press_integrate();

    if (mpchain) nhc_press_integrate();
    compute_temp_target();
    defer_temp_scale = true;
    nhc_temp_integrate();
    defer_temp_scale = false;
    payload.temp_initial = pending_factor_eta;
    temp_scale_pending = false;

    temperature->invoked_scalar = update->ntimestep;
    temperature->scalar = t_current;
    pressure->compute_scalar();
    couple();
    pressure->addstep(update->ntimestep + 1);
    compute_press_target();
    nh_omega_dot();
    press_factors();
    payload.press[0] = factor[0];
    payload.press[1] = factor[1];
    payload.press[2] = factor[2];

    auto fill = [&](BoxMap &m, const double *matrix) {
      for (int k = 0; k < 3; ++k) m.lo[k] = domain->boxlo[k];
      for (int k = 0; k < 6; ++k) m.m[k] = matrix[k];
    };
    fill(payload.old_inv, domain->h_inv);
    remap_box();
    fill(payload.middle, domain->h);
    fill(payload.middle_inv, domain->h_inv);
    remap_box();
    fill(payload.final_box, domain->h);
  }

  bool coroutine_step() {
    atomKK->sync(execution_space,
        X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
    atomKK->k_mass.sync<LMPDeviceType>();
    x = atomKK->k_x.view<LMPDeviceType>();
    v = atomKK->k_v.view<LMPDeviceType>();
    f = atomKK->k_f.view<LMPDeviceType>();
    rmass = atomKK->k_rmass.view<LMPDeviceType>();
    mass = atomKK->k_mass.view<LMPDeviceType>();
    type = atomKK->k_type.view<LMPDeviceType>();
    mask = atomKK->k_mask.view<LMPDeviceType>();
    const int nlocal = atomKK->nlocal;

    press_factors();
    const double final_press[3] = {factor[0], factor[1], factor[2]};
    const DpaFastStep::Arr3<const double> force_view{
        f.data(), static_cast<long>(f.stride_0()),
        static_cast<long>(f.stride_1())};
    __atomic_store_n(&host_bridge_->phase, 0, __ATOMIC_RELEASE);
    bool launched;
    if (rmass.data())
      launched = launch_coroutine<1>(
          nlocal, nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v),
          force_view, mask.data(), rmass.data(), mass.data(), type.data(),
          dtf, dtv, groupbit, temperature->groupbit, final_press,
          DpaFastStep::scratch().partial, DpaFastStep::scratch().ticket,
          device_bridge_);
    else
      launched = launch_coroutine<0>(
          nlocal, nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v),
          force_view, mask.data(), rmass.data(), mass.data(), type.data(),
          dtf, dtv, groupbit, temperature->groupbit, final_press,
          DpaFastStep::scratch().partial, DpaFastStep::scratch().ticket,
          device_bridge_);
    if (!launched) return false;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(1);
    while (__atomic_load_n(&host_bridge_->phase, __ATOMIC_ACQUIRE) != 1) {
      if (std::chrono::steady_clock::now() > deadline) {
        __atomic_store_n(&host_bridge_->phase, 2, __ATOMIC_RELEASE);
        cudaStreamSynchronize(DpaFastStep::stream());
        error->all(FLERR, "small NPT mapped coroutine handshake timeout");
      }
    }
    host_resume(host_bridge_->kinetic_sum, host_bridge_->payload);
    __atomic_store_n(&host_bridge_->phase, 2, __ATOMIC_RELEASE);
    // The next distance-check/communication work is enqueued on the same
    // stream.  Returning here preserves stream order and lets its existing
    // synchronization cover completion of this kernel.
    atomKK->modified(execution_space, X_MASK | V_MASK);
    return true;
  }

  void initial_with_pending_scales(bool two_scales, double final_scale) {
    if (mpchain) nhc_press_integrate();
    compute_temp_target();
    defer_temp_scale = true;
    nhc_temp_integrate();
    defer_temp_scale = false;
    temperature->invoked_scalar = update->ntimestep;
    temperature->scalar = t_current;
    atomKK->sync(pressure->execution_space, pressure->datamask_read);
    pressure->compute_scalar();
    atomKK->modified(temperature->execution_space, temperature->datamask_modify);
    atomKK->modified(pressure->execution_space, pressure->datamask_modify);
    atomKK->sync(execution_space, temperature->datamask_modify);
    atomKK->sync(execution_space, pressure->datamask_modify);
    couple();
    pressure->addstep(update->ntimestep + 1);
    compute_press_target();
    nh_omega_dot();
    const double initial_scale = pending_factor_eta;
    temp_scale_pending = false;
    fallback_initial(two_scales, final_scale, initial_scale);
    if (kspace_flag) force->kspace->setup();
  }

  void fallback_initial(bool two_scales, double final_scale,
                        double initial_scale) {
    atomKK->sync(execution_space,
        X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
    auto xk = atomKK->k_x.view<LMPDeviceType>();
    auto vk = atomKK->k_v.view<LMPDeviceType>();
    press_factors();
    BoxMap a_inv{}, b{}, b_inv{}, c{};
    auto fill = [&](BoxMap &m, const double *matrix) {
      for (int k = 0; k < 3; ++k) m.lo[k] = domain->boxlo[k];
      for (int k = 0; k < 6; ++k) m.m[k] = matrix[k];
    };
    fill(a_inv, domain->h_inv);
    remap_box();
    fill(b, domain->h);
    fill(b_inv, domain->h_inv);
    remap_box();
    fill(c, domain->h);
    x = xk; v = vk;
    f = atomKK->k_f.view<LMPDeviceType>();
    rmass = atomKK->k_rmass.view<LMPDeviceType>();
    mass = atomKK->k_mass.view<LMPDeviceType>();
    type = atomKK->k_type.view<LMPDeviceType>();
    mask = atomKK->k_mask.view<LMPDeviceType>();
    const int nlocal = atomKK->nlocal;
    copymode = 1;
    if (rmass.data())
      fallback_npt_vx<1, LMPDeviceType>(
          nlocal, x, v, f, mask, rmass, type, two_scales, final_scale,
          initial_scale, factor[0], factor[1], factor[2], dtf, dtv,
          groupbit, a_inv, b, b_inv, c);
    else
      fallback_npt_vx<0, LMPDeviceType>(
          nlocal, x, v, f, mask, mass, type, two_scales, final_scale,
          initial_scale, factor[0], factor[1], factor[2], dtf, dtv,
          groupbit, a_inv, b, b_inv, c);
    copymode = 0;
    atomKK->modified(execution_space, X_MASK | V_MASK);
  }
};

Fix *make_fix(LAMMPS *lmp, int narg, char **arg) {
  return new FixNPTCoroutineKokkos(lmp, narg, arg);
}

}  // namespace


extern "C" bool dpa_small_epoch_prepare(void *owner,double *virial,double *energy,double scale,
    bool do_e,bool do_v,DpaSmallEpoch::Args *args,DpaSmallEpoch::Ev **out) {
  auto *p=FixNPTCoroutineKokkos::epoch_owner_;
  return p && p->prepare_epoch(owner,virial,energy,scale,do_e,do_v,args,out);
}
extern "C" void dpa_small_epoch_publish(cudaStream_t stream) {
  auto *p=FixNPTCoroutineKokkos::epoch_owner_;if(p)p->publish_epoch(stream);
}
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "npt/kk";
  plugin.info = "Guarded small ISO-NPT atomic-ticket host/device coroutine";
  plugin.author = "DPA optimization";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&make_fix);
  plugin.handle = handle;
  register_plugin(&plugin, lmp);
  plugin.name = "npt/kk/device";
  register_plugin(&plugin, lmp);
}
