#include "fix_nvt_fused_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "comm_kokkos.h"
#include "compute.h"
#include "compute_temp_kokkos.h"
#include "fix.h"
#include "modify.h"
#include "output.h"
#include "update.h"
#include "dpa_fast_step_kokkos.h"
#include "neighbor_kokkos.h"
#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

using namespace LAMMPS_NS;

namespace {

using PACKED_NVTGetPackedFn = int (*)(unsigned long long**, int**, long*);
using PACKED_NVTArmFn = int (*)(int);
using PACKED_NVTMaterializeFn = int (*)(void*);
using PACKED_NVTObserverNeedsFn = int (*)(long long);

class PACKED_NVTModifyAccess : public Modify {
 public:
  int post_count() const { return n_post_force; }
  int post_group_count() const { return n_post_force_group; }
  int* post_list() const { return list_post_force; }
};

bool packed_nvt_requested() {
  static const bool value = [] {
    const char* setting = std::getenv("DPA_SMALL_NVT_PACKED_FORCE");
    return setting && std::atoi(setting) != 0;
  }();
  return value;
}

int packed_nvt_arm(int enable) {
  static auto arm = reinterpret_cast<PACKED_NVTArmFn>(
      dlsym(RTLD_DEFAULT, "dpa_chain_small_nvt_arm_small_nvt"));
  if (!packed_nvt_requested() || !arm) return 0;
  return arm(enable);
}

int packed_nvt_materialize(cudaStream_t stream) {
  static auto materialize = reinterpret_cast<PACKED_NVTMaterializeFn>(
      dlsym(RTLD_DEFAULT, "dpa_chain_small_nvt_materialize_force"));
  if (!packed_nvt_requested() || !materialize) return 0;
  return materialize(reinterpret_cast<void*>(stream));
}

bool packed_nvt_get_packed(unsigned long long*& packed, int*& skip_force,
                     long& nodes) {
  static auto get = reinterpret_cast<PACKED_NVTGetPackedFn>(
      dlsym(RTLD_DEFAULT, "dpa_chain_small_nvt_get_packed_force"));
  if (!packed_nvt_requested() || !get) return false;
  return get(&packed, &skip_force, &nodes) != 0;
}

bool packed_nvt_observer_needs_force(long long step) {
  static auto needs = reinterpret_cast<PACKED_NVTObserverNeedsFn>(
      dlsym(RTLD_DEFAULT, "dpa_force_observer_needs_force"));
  // A missing query makes the optimization ineligible.  This keeps an old
  // observer library correct instead of guessing its capture schedule.
  return !needs || needs(step) != 0;
}

bool packed_nvt_post_force_safe(Modify* base, long long step) {
  auto* modify = static_cast<PACKED_NVTModifyAccess*>(base);
  if (modify->post_group_count() != 0) return false;
  if (modify->post_count() == 0) return true;
  if (modify->post_count() != 1) return false;
  Fix* fix = modify->fix[modify->post_list()[0]];
  const bool observer = fix && fix->id && fix->style &&
      std::strcmp(fix->id, "organizer_force_observer") == 0 &&
      std::strcmp(fix->style, "dpa/force-observer") == 0;
  return observer && !packed_nvt_observer_needs_force(step);
}

struct PACKED_NVTForce3 { double x, y, z; };

__device__ __forceinline__ PACKED_NVTForce3 packed_nvt_decode_force(
    const unsigned long long* packed, int atom) {
  constexpr double inv = 1.0 / 16777216.0;
  const unsigned long long xy = packed[2 * atom];
  const long long qx = static_cast<int>(static_cast<unsigned>(xy));
  const long long qy = (static_cast<long long>(xy) - qx) >> 32;
  const long long qz = static_cast<long long>(packed[2 * atom + 1]);
  return {
      static_cast<double>(static_cast<float>(double(qx) * inv)),
      static_cast<double>(static_cast<float>(double(qy) * inv)),
      static_cast<double>(static_cast<float>(double(qz) * inv))};
}

template <int RMASS, int M>
__global__ void __launch_bounds__(DpaFastStep::kBlock)
dpa_nhc_fused_final_initial_kernel(
    int n, DpaFastStep::Arr3<double> x, DpaFastStep::Arr3<double> v,
    DpaFastStep::Arr3<const double> f, const int *mask,
    const double *rmass, const double *mass, const int *type,
    double dtf, double dtv, int groupbit, const double *sum,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p)
{
  __shared__ double factors[2];
  if (threadIdx.x == 0) {
    DpaFastStep::NhcState middle;
    factors[0] = DpaFastStep::nhc_chain<M>(
        in, &middle, *sum * p.tfactor, p);
    factors[1] = DpaFastStep::nhc_chain<M>(
        &middle, blockIdx.x == 0 ? out : nullptr, -1.0, p);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    if (mask[i] & groupbit) {
      // Keep the two thermostat multiplies separate to preserve the same
      // rounding order as final_integrate followed by initial_integrate.
      v(i, 0) *= factors[0];
      v(i, 1) *= factors[0];
      v(i, 2) *= factors[0];
      v(i, 0) *= factors[1];
      v(i, 1) *= factors[1];
      v(i, 2) *= factors[1];
      const double dtfm = dtf / (RMASS ? rmass[i] : mass[type[i]]);
      v(i, 0) += dtfm * f(i, 0);
      v(i, 1) += dtfm * f(i, 1);
      v(i, 2) += dtfm * f(i, 2);
      x(i, 0) += dtv * v(i, 0);
      x(i, 1) += dtv * v(i, 1);
      x(i, 2) += dtv * v(i, 2);
    }
  }
}



template <int RMASS, int M>
__global__ void __launch_bounds__(DpaFastStep::kBlock)
dpa_cooperative_nhc_coop_final_initial_kernel(
    int n, int ni, DpaFastStep::Arr3<double> x,
    DpaFastStep::Arr3<double> v, DpaFastStep::Arr3<const double> f,
    const int *mask, const double *rmass, const double *mass,
    const int *type, double dtf, double dtv, int groupbit,
    int tgroupbit, double *partial, double *result,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p, DpaFastStep::Arr3<const double> xhold,
    double deltasq, int ncheck, int check_stamp, int *check_flag,
    int fused_check, const int *pack_map, long pack_s0, long pack_s1,
    int pack_count, double pack_xprd, double pack_yprd, double pack_zprd,
    double pack_xy, double pack_xz, double pack_yz, int pack_triclinic,
    int fused_pack, const unsigned long long* packed_force, int use_packed,
    int* next_skip_flag, int next_skip)
{
  cg::grid_group whole_grid = cg::this_grid();
  __shared__ double reduction[32];
  __shared__ double factors[2];

  // This is the existing v_update_temp<RMASS,0,1,0> arithmetic.  Global
  // stores are intentionally retained before the grid barrier so the
  // numerical path matches the former first kernel, including FP64 rounding.
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    const int m = mask[i];
    const double massone = RMASS ? rmass[i] : mass[type[i]];
    if (m & groupbit) {
      const double dtfm = dtf / massone;
      if (use_packed) {
        const PACKED_NVTForce3 pf = packed_nvt_decode_force(packed_force, i);
        v(i, 0) += dtfm * pf.x;
        v(i, 1) += dtfm * pf.y;
        v(i, 2) += dtfm * pf.z;
      } else {
        v(i, 0) += dtfm * f(i, 0);
        v(i, 1) += dtfm * f(i, 1);
        v(i, 2) += dtfm * f(i, 2);
      }
    }
    if (m & tgroupbit)
      acc += (v(i, 0) * v(i, 0) + v(i, 1) * v(i, 1) +
              v(i, 2) * v(i, 2)) * massone;
  }

  // Preserve the old fixed-grid reduction tree: one partial per logical
  // 256-thread block, followed by thread k consuming partial[k].
  const double block_total = DpaFastStep::block_sum(acc, reduction);
  if (threadIdx.x == 0) partial[blockIdx.x] = block_total;
  whole_grid.sync();

  double total = 0.0;
  for (int k = threadIdx.x; k < static_cast<int>(gridDim.x);
       k += blockDim.x)
    total += partial[k];
  total = DpaFastStep::block_sum(total, reduction);

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) *result = total;
    DpaFastStep::NhcState middle;
    factors[0] = DpaFastStep::nhc_chain<M>(
        in, &middle, total * p.tfactor, p);
    factors[1] = DpaFastStep::nhc_chain<M>(
        &middle, blockIdx.x == 0 ? out : nullptr, -1.0, p);
  }
  __syncthreads();

  // Existing dpa_nhc_fused_final_initial_kernel arithmetic.  Keep the two
  // thermostat multiplies and the next half kick as separate operations.
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < ni; i += gridDim.x * blockDim.x) {
    if (mask[i] & groupbit) {
      v(i, 0) *= factors[0];
      v(i, 1) *= factors[0];
      v(i, 2) *= factors[0];
      v(i, 0) *= factors[1];
      v(i, 1) *= factors[1];
      v(i, 2) *= factors[1];
      const double dtfm = dtf / (RMASS ? rmass[i] : mass[type[i]]);
      if (use_packed) {
        const PACKED_NVTForce3 pf = packed_nvt_decode_force(packed_force, i);
        v(i, 0) += dtfm * pf.x;
        v(i, 1) += dtfm * pf.y;
        v(i, 2) += dtfm * pf.z;
      } else {
        v(i, 0) += dtfm * f(i, 0);
        v(i, 1) += dtfm * f(i, 1);
        v(i, 2) += dtfm * f(i, 2);
      }
      x(i, 0) += dtv * v(i, 0);
      x(i, 1) += dtv * v(i, 1);
      x(i, 2) += dtv * v(i, 2);
      // The displacement belongs to the same atom just updated above, so it
      // needs neither another pass nor a grid barrier.  Standard NVT uses the
      // all group; the host arms this path only when ni == n.
      if (fused_check && i < ncheck) {
        const double delx = x(i, 0) - xhold(i, 0);
        const double dely = x(i, 1) - xhold(i, 1);
        const double delz = x(i, 2) - xhold(i, 2);
        if (delx * delx + dely * dely + delz * delz > deltasq)
          *check_flag = check_stamp;
      }
    }
  }

  if (fused_pack) {
    // All owned positions must be globally visible before ghosts read them.
    whole_grid.sync();
    for (int ii = blockIdx.x * blockDim.x + threadIdx.x;
         ii < pack_count; ii += gridDim.x * blockDim.x) {
      const int *row = pack_map + ii * pack_s0;
      const int source = row[0 * pack_s1];
      const int destination = row[1 * pack_s1];
      double dx = 0.0, dy = 0.0, dz = 0.0;
      if (row[2 * pack_s1]) {
        dx = row[3 * pack_s1] * pack_xprd;
        dy = row[4 * pack_s1] * pack_yprd;
        dz = row[5 * pack_s1] * pack_zprd;
        if (pack_triclinic) {
          dx += row[8 * pack_s1] * pack_xy +
                row[7 * pack_s1] * pack_xz;
          dy += row[6 * pack_s1] * pack_yz;
        }
      }
      x(destination, 0) = x(source, 0) + dx;
      x(destination, 1) = x(source, 1) + dy;
      x(destination, 2) = x(source, 2) + dz;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && next_skip_flag)
    *next_skip_flag = next_skip;
}

template <int RMASS, int M>
bool launch_cooperative_nhc_cooperative(
    int n, int ni, DpaFastStep::Arr3<double> x,
    DpaFastStep::Arr3<double> v, DpaFastStep::Arr3<const double> f,
    const int *mask, const double *rmass, const double *mass,
    const int *type, double dtf, double dtv, int groupbit,
    int tgroupbit, double *partial, double *result,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p, DpaFastStep::Arr3<const double> xhold,
    double deltasq, int ncheck, int check_stamp, int *check_flag,
    int fused_check, const int *pack_map, long pack_s0, long pack_s1,
    int pack_count, double pack_xprd, double pack_yprd, double pack_zprd,
    double pack_xy, double pack_xz, double pack_yz, int pack_triclinic,
    int fused_pack, const unsigned long long* packed_force, int use_packed,
    int* next_skip_flag, int next_skip)
{
  const int blocks = DpaFastStep::grid_for(n);
  static const int capacity = [] {
    int device = 0, supported = 0, sms = 0, resident = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch,
                               device) != cudaSuccess || !supported)
      return 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess)
      return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, dpa_cooperative_nhc_coop_final_initial_kernel<RMASS, M>,
            DpaFastStep::kBlock, 0) != cudaSuccess)
      return 0;
    return sms * resident;
  }();
  static bool reported = false;
  if (!reported) {
    reported = true;
    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(
        &attributes, dpa_cooperative_nhc_coop_final_initial_kernel<RMASS, M>);
    std::fprintf(stderr,
                 "COOPERATIVE_NHC_COOP_RESOURCE rmass=%d chain=%d blocks=%d capacity=%d "
                 "regs=%d local=%zu shared=%zu\n",
                 RMASS, M, blocks, capacity, attributes.numRegs,
                 attributes.localSizeBytes, attributes.sharedSizeBytes);
  }
  if (capacity <= 0 || blocks > capacity) return false;

  void *arguments[] = {
      &n, &ni, &x, &v, &f, &mask, &rmass, &mass, &type,
      &dtf, &dtv, &groupbit, &tgroupbit, &partial, &result,
      &in, &out, &p, &xhold, &deltasq, &ncheck, &check_stamp,
      &check_flag, &fused_check, &pack_map, &pack_s0, &pack_s1,
      &pack_count, &pack_xprd, &pack_yprd, &pack_zprd,
      &pack_xy, &pack_xz, &pack_yz, &pack_triclinic, &fused_pack,
      &packed_force, &use_packed, &next_skip_flag, &next_skip};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(dpa_cooperative_nhc_coop_final_initial_kernel<RMASS, M>),
      dim3(blocks), dim3(DpaFastStep::kBlock), arguments, 0,
      DpaFastStep::stream());
  if (status != cudaSuccess) {
    std::fprintf(stderr, "COOPERATIVE_NHC_COOP_LAUNCH_FAILED status=%s\n",
                 cudaGetErrorString(status));
    cudaGetLastError();
    return false;
  }
  return true;
}


template <int RMASS, int M>
__global__ void __launch_bounds__(DpaFastStep::kBlock)
dpa_cooperative_nhc_output_final_kernel(
    int n, DpaFastStep::Arr3<double> v,
    DpaFastStep::Arr3<const double> f, const int *mask,
    const double *rmass, const double *mass, const int *type,
    double dtf, int groupbit, int tgroupbit,
    double *partial, double *result,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p)
{
  cg::grid_group whole_grid = cg::this_grid();
  __shared__ double reduction[32];
  __shared__ double factor;

  // Complete only the current timestep.  Output must observe x(t) and the
  // fully updated v(t), so this route deliberately does not execute the next
  // initial half-kick or position update.
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    const int m = mask[i];
    const double massone = RMASS ? rmass[i] : mass[type[i]];
    if (m & groupbit) {
      const double dtfm = dtf / massone;
      v(i, 0) += dtfm * f(i, 0);
      v(i, 1) += dtfm * f(i, 1);
      v(i, 2) += dtfm * f(i, 2);
    }
    if (m & tgroupbit)
      acc += (v(i, 0) * v(i, 0) + v(i, 1) * v(i, 1) +
              v(i, 2) * v(i, 2)) * massone;
  }

  const double block_total = DpaFastStep::block_sum(acc, reduction);
  if (threadIdx.x == 0) partial[blockIdx.x] = block_total;
  whole_grid.sync();

  double total = 0.0;
  for (int k = threadIdx.x; k < static_cast<int>(gridDim.x);
       k += blockDim.x)
    total += partial[k];
  total = DpaFastStep::block_sum(total, reduction);

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) *result = total;
    factor = DpaFastStep::nhc_chain<M>(
        in, blockIdx.x == 0 ? out : nullptr, total * p.tfactor, p);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    if (mask[i] & groupbit) {
      v(i, 0) *= factor;
      v(i, 1) *= factor;
      v(i, 2) *= factor;
    }
  }
}

template <int RMASS, int M>
bool launch_cooperative_nhc_output_final(
    int n, DpaFastStep::Arr3<double> v,
    DpaFastStep::Arr3<const double> f, const int *mask,
    const double *rmass, const double *mass, const int *type,
    double dtf, int groupbit, int tgroupbit,
    double *partial, double *result,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p)
{
  const int blocks = DpaFastStep::grid_for(n);
  static const int capacity = [] {
    int device = 0, supported = 0, sms = 0, resident = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch,
                               device) != cudaSuccess || !supported)
      return 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess)
      return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, dpa_cooperative_nhc_output_final_kernel<RMASS, M>,
            DpaFastStep::kBlock, 0) != cudaSuccess)
      return 0;
    return sms * resident;
  }();
  static bool reported = false;
  if (!reported) {
    reported = true;
    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(
        &attributes, dpa_cooperative_nhc_output_final_kernel<RMASS, M>);
    std::fprintf(stderr,
        "SMALL_ROUTEGRAPH_OUTPUT_FINAL_RESOURCE rmass=%d chain=%d "
        "blocks=%d capacity=%d regs=%d local=%zu shared=%zu\n",
        RMASS, M, blocks, capacity, attributes.numRegs,
        attributes.localSizeBytes, attributes.sharedSizeBytes);
  }
  if (capacity <= 0 || blocks > capacity) return false;

  void *arguments[] = {
      &n, &v, &f, &mask, &rmass, &mass, &type, &dtf,
      &groupbit, &tgroupbit, &partial, &result, &in, &out, &p};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(
          dpa_cooperative_nhc_output_final_kernel<RMASS, M>),
      dim3(blocks), dim3(DpaFastStep::kBlock), arguments, 0,
      DpaFastStep::stream());
  if (status != cudaSuccess) {
    std::fprintf(stderr,
        "SMALL_ROUTEGRAPH_OUTPUT_FINAL_LAUNCH_FAILED status=%s\n",
        cudaGetErrorString(status));
    cudaGetLastError();
    return false;
  }
  return true;
}


template <int RMASS, int M>
__global__ void __launch_bounds__(DpaFastStep::kBlock)
dpa_cooperative_nhc_output_initial_kernel(
    int n, DpaFastStep::Arr3<double> x, DpaFastStep::Arr3<double> v,
    DpaFastStep::Arr3<const double> f, const int *mask,
    const double *rmass, const double *mass, const int *type,
    double dtf, double dtv, int groupbit,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p, DpaFastStep::Arr3<const double> xhold,
    double deltasq, int ncheck, int check_stamp, int *check_flag,
    int fused_check, const int *pack_map, long pack_s0, long pack_s1,
    int pack_count, double pack_xprd, double pack_yprd, double pack_zprd,
    double pack_xy, double pack_xz, double pack_yz, int pack_triclinic,
    int fused_pack, int *next_skip_flag, int next_skip)
{
  cg::grid_group whole_grid = cg::this_grid();
  __shared__ double factor;
  if (threadIdx.x == 0)
    factor = DpaFastStep::nhc_chain<M>(
        in, blockIdx.x == 0 ? out : nullptr, -1.0, p);
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    if (mask[i] & groupbit) {
      v(i, 0) *= factor;
      v(i, 1) *= factor;
      v(i, 2) *= factor;
      const double dtfm = dtf / (RMASS ? rmass[i] : mass[type[i]]);
      v(i, 0) += dtfm * f(i, 0);
      v(i, 1) += dtfm * f(i, 1);
      v(i, 2) += dtfm * f(i, 2);
      x(i, 0) += dtv * v(i, 0);
      x(i, 1) += dtv * v(i, 1);
      x(i, 2) += dtv * v(i, 2);
      if (fused_check && i < ncheck) {
        const double delx = x(i, 0) - xhold(i, 0);
        const double dely = x(i, 1) - xhold(i, 1);
        const double delz = x(i, 2) - xhold(i, 2);
        if (delx * delx + dely * dely + delz * delz > deltasq)
          *check_flag = check_stamp;
      }
    }
  }

  if (fused_pack) {
    whole_grid.sync();
    for (int ii = blockIdx.x * blockDim.x + threadIdx.x;
         ii < pack_count; ii += gridDim.x * blockDim.x) {
      const int *row = pack_map + ii * pack_s0;
      const int source_atom = row[0 * pack_s1];
      const int destination = row[1 * pack_s1];
      double dx = 0.0, dy = 0.0, dz = 0.0;
      if (row[2 * pack_s1]) {
        dx = row[3 * pack_s1] * pack_xprd;
        dy = row[4 * pack_s1] * pack_yprd;
        dz = row[5 * pack_s1] * pack_zprd;
        if (pack_triclinic) {
          dx += row[8 * pack_s1] * pack_xy + row[7 * pack_s1] * pack_xz;
          dy += row[6 * pack_s1] * pack_yz;
        }
      }
      x(destination, 0) = x(source_atom, 0) + dx;
      x(destination, 1) = x(source_atom, 1) + dy;
      x(destination, 2) = x(source_atom, 2) + dz;
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && next_skip_flag)
    *next_skip_flag = next_skip;
}

template <int RMASS, int M>
bool launch_cooperative_nhc_output_initial(
    int n, DpaFastStep::Arr3<double> x, DpaFastStep::Arr3<double> v,
    DpaFastStep::Arr3<const double> f, const int *mask,
    const double *rmass, const double *mass, const int *type,
    double dtf, double dtv, int groupbit,
    const DpaFastStep::NhcState *in, DpaFastStep::NhcState *out,
    DpaFastStep::NhcParams p, DpaFastStep::Arr3<const double> xhold,
    double deltasq, int ncheck, int check_stamp, int *check_flag,
    int fused_check, const int *pack_map, long pack_s0, long pack_s1,
    int pack_count, double pack_xprd, double pack_yprd, double pack_zprd,
    double pack_xy, double pack_xz, double pack_yz, int pack_triclinic,
    int fused_pack, int *next_skip_flag, int next_skip)
{
  const int blocks = DpaFastStep::grid_for(n);
  static const int capacity = [] {
    int device = 0, supported = 0, sms = 0, resident = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch,
                               device) != cudaSuccess || !supported)
      return 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess)
      return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, dpa_cooperative_nhc_output_initial_kernel<RMASS, M>,
            DpaFastStep::kBlock, 0) != cudaSuccess)
      return 0;
    return sms * resident;
  }();
  static bool reported = false;
  if (!reported) {
    reported = true;
    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(
        &attributes, dpa_cooperative_nhc_output_initial_kernel<RMASS, M>);
    std::fprintf(stderr,
        "SMALL_ROUTEGRAPH_OUTPUT_INITIAL_RESOURCE rmass=%d chain=%d "
        "blocks=%d capacity=%d regs=%d local=%zu shared=%zu\n",
        RMASS, M, blocks, capacity, attributes.numRegs,
        attributes.localSizeBytes, attributes.sharedSizeBytes);
  }
  if (capacity <= 0 || blocks > capacity) return false;
  void *arguments[] = {
      &n, &x, &v, &f, &mask, &rmass, &mass, &type,
      &dtf, &dtv, &groupbit, &in, &out, &p, &xhold, &deltasq,
      &ncheck, &check_stamp, &check_flag, &fused_check,
      &pack_map, &pack_s0, &pack_s1, &pack_count,
      &pack_xprd, &pack_yprd, &pack_zprd, &pack_xy, &pack_xz,
      &pack_yz, &pack_triclinic, &fused_pack, &next_skip_flag, &next_skip};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(
          dpa_cooperative_nhc_output_initial_kernel<RMASS, M>),
      dim3(blocks), dim3(DpaFastStep::kBlock), arguments, 0,
      DpaFastStep::stream());
  if (status != cudaSuccess) {
    std::fprintf(stderr,
        "SMALL_ROUTEGRAPH_OUTPUT_INITIAL_LAUNCH_FAILED status=%s\n",
        cudaGetErrorString(status));
    cudaGetLastError();
    return false;
  }
  return true;
}

}  // namespace

FixNVTFusedKokkos::FixNVTFusedKokkos(LAMMPS *lmp, int narg, char **arg)
    : FixNVTKokkos<LMPDeviceType>(lmp, narg, arg)
{
  const bool requested = packed_nvt_requested();
  const bool fixed_temperature = t_start == t_stop;
  const bool small_system = atom->natoms <= 4096;
  const bool all_group = igroup == 0;
  const int arm_status = requested && fixed_temperature && small_system && all_group &&
      comm->nprocs == 1 ? packed_nvt_arm(1) : 0;
  packed_nvt_armed = arm_status > 0;
  std::fprintf(stderr,
      "PACKED_NVT_ARM requested=%d fixed=%d small=%d all=%d ranks=%d status=%d atoms=%lld\n",
      requested ? 1 : 0, fixed_temperature ? 1 : 0, small_system ? 1 : 0,
      all_group ? 1 : 0, comm->nprocs, arm_status,
      static_cast<long long>(atom->natoms));
}

void FixNVTFusedKokkos::init()
{
  FixNVTKokkos<LMPDeviceType>::init();
  dpa_fuse_enabled = false;
  cooperative_nhc_coop_enabled = false;
  packed_nvt_packed_current = false;
  fuse_integrate_flag = 0;
  bool consumers_safe = true;
  for (int i = 0; i < modify->nfix; ++i) {
    if (modify->fmask[i] & FixConst::END_OF_STEP) consumers_safe = false;
    if (modify->fmask[i] & FixConst::POST_FORCE) {
      Fix* fix = modify->fix[i];
      const bool observer = fix && fix->id && fix->style &&
          std::strcmp(fix->id, "organizer_force_observer") == 0 &&
          std::strcmp(fix->style, "dpa/force-observer") == 0;
      if (!observer) consumers_safe = false;
    }
  }
  int cooperative_supported = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess)
    cudaDeviceGetAttribute(&cooperative_supported,
                           cudaDevAttrCooperativeLaunch, device);
#endif
  const bool valid = packed_nvt_armed && t_start == t_stop &&
      atom->natoms <= 4096 && igroup == 0 && comm->nprocs == 1 &&
      cooperative_supported != 0 && consumers_safe;
  if (!valid && packed_nvt_armed) {
    if (packed_nvt_arm(0) < 0) {
      std::fprintf(stderr, "PACKED_NVT could not disarm after graph capture\n");
      std::abort();
    }
    packed_nvt_armed = false;
  }
}

void FixNVTFusedKokkos::setup(int vflag)
{
  routegraph_output_initial_pending = false;
  if (packed_nvt_armed && packed_nvt_materialize(DpaFastStep::stream()) < 0) {
    std::fprintf(stderr, "PACKED_NVT initial-force materialization failed\n");
    std::abort();
  }
  FixNVTKokkos<LMPDeviceType>::setup(vflag);
  // setup() can execute a fresh force evaluation (notably organizer run 0
  // after fresh_state).  The packed-force producer deliberately leaves the
  // conventional LAMMPS force view untouched, so publish it at this
  // observation boundary after the base setup has produced the new force.
  const int post_setup_materialized = packed_nvt_armed ?
      packed_nvt_materialize(DpaFastStep::stream()) : 0;
  if (post_setup_materialized < 0) {
    std::fprintf(stderr, "PACKED_NVT post-setup force materialization failed\n");
    std::abort();
  }
  // The materializer writes the raw device pointer outside Kokkos dispatch.
  // Invalidate a host copy that FixNVT::setup may have synchronized so the
  // enclosing VerletKokkos::setup copies the newly materialized force.
  if (post_setup_materialized > 0)
    atomKK->modified(execution_space, F_MASK);
  static bool integrated_neighbor_check_plugin_reported = false;
  if (!integrated_neighbor_check_plugin_reported && comm->me == 0) {
    integrated_neighbor_check_plugin_reported = true;
    const char *integrated_neighbor_check = std::getenv("DPA_INTEGRATED_NEIGHBOR_CHECK_INTEGRATE_CHECK");
    utils::logmesg(lmp, "DPA_INTEGRATED_NEIGHBOR_CHECK_NVT_PLUGIN selected={}\n",
                   integrated_neighbor_check && std::atoi(integrated_neighbor_check) != 0 ? 1 : 0);
    const char *flat_self_pack = std::getenv("DPA_FLAT_SELF_PACK_FLAT_SELF_PACK");
    utils::logmesg(lmp, "DPA_FLAT_SELF_PACK_FLAT_SELF_PACK selected={}\n",
                   flat_self_pack && std::atoi(flat_self_pack) != 0 ? 1 : 0);
  }
  dpa_nhc_host_current = true;
  const char *setting = std::getenv("DPA_NHC_FUSION");
  const bool requested = setting && std::atoi(setting) != 0;
  const bool fixed_temperature = t_start == t_stop;
  const bool device_nhc_system =
      DpaFastStep::within_device_nhc_limit(atom->natoms);
  dpa_fuse_enabled = requested && nhc_device && fixed_temperature &&
      device_nhc_system;
  fuse_integrate_flag = dpa_fuse_enabled ? 1 : 0;

  const char *cooperative_nhc_setting = std::getenv("DPA_COOPERATIVE_NHC_COOP_NHC");
  const bool cooperative_nhc_requested = cooperative_nhc_setting && std::atoi(cooperative_nhc_setting) != 0;
  int cooperative_supported = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess)
    cudaDeviceGetAttribute(&cooperative_supported,
                           cudaDevAttrCooperativeLaunch, device);
#endif
  cooperative_nhc_coop_enabled = cooperative_nhc_requested && dpa_fuse_enabled &&
                      comm->nprocs == 1 && cooperative_supported != 0;
  if (cooperative_nhc_requested && comm->me == 0)
    utils::logmesg(lmp,
        "DPA_COOPERATIVE_NHC_COOP_NHC active={} cooperative={} single_rank={} atoms={}\n",
        cooperative_nhc_coop_enabled ? 1 : 0, cooperative_supported,
        comm->nprocs == 1 ? 1 : 0, atom->natoms);

  if (requested && comm->me == 0) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      utils::logmesg(lmp,
          "DPA_NHC_FUSED active={} nhc_device={} fixed_target={} eligible={} atoms={}\n",
          dpa_fuse_enabled ? 1 : 0, nhc_device ? 1 : 0,
          fixed_temperature ? 1 : 0, device_nhc_system ? 1 : 0,
          atom->natoms);
    }
  }
}

void FixNVTFusedKokkos::fused_integrate(int vflag)
{
  dpa_nhc_host_current = false;
  if (!dpa_fuse_enabled) {
    if (packed_nvt_armed && packed_nvt_materialize(DpaFastStep::stream()) < 0) {
      std::fprintf(stderr, "PACKED_NVT fallback force materialization failed\n");
      std::abort();
    }
    FixNVTKokkos<LMPDeviceType>::final_integrate();
    FixNVTKokkos<LMPDeviceType>::initial_integrate(vflag);
    return;
  }

#if defined(KOKKOS_ENABLE_CUDA)
  if (cooperative_nhc_coop_enabled && cooperative_nhc_integrate()) return;
  tdof = temperature->dof;

  // Current-step final half-kick plus the temperature reduction.  The
  // following kernel consumes the device result in stream order.
  v_update_temp(false, true, false, false);

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

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;
  const DpaFastStep::NhcParams p = nhc_params();
  auto &buffers = DpaFastStep::nhc_buffers();
  const auto *in = buffers.state + buffers.current;
  auto *out = buffers.state + (buffers.current ^ 1);
  buffers.current ^= 1;

  const DpaFastStep::Arr3<const double> force_view{
      f.data(), static_cast<long>(f.stride_0()), static_cast<long>(f.stride_1())};
  const int grid = DpaFastStep::grid_for(nlocal);
  const double *sum = DpaFastStep::scratch().result;
  auto launch = [&](auto chain) {
    constexpr int M = decltype(chain)::value;
    if (rmass.data())
      dpa_nhc_fused_final_initial_kernel<1, M>
          <<<grid, DpaFastStep::kBlock, 0, DpaFastStep::stream()>>>(
              nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v), force_view,
              mask.data(), rmass.data(), mass.data(), type.data(), dtf, dtv,
              groupbit, sum, in, out, p);
    else
      dpa_nhc_fused_final_initial_kernel<0, M>
          <<<grid, DpaFastStep::kBlock, 0, DpaFastStep::stream()>>>(
              nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v), force_view,
              mask.data(), rmass.data(), mass.data(), type.data(), dtf, dtv,
              groupbit, sum, in, out, p);
  };
  switch (mtchain) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    default: launch(std::integral_constant<int, 5>{}); break;
  }
  atomKK->modified(execution_space, X_MASK | V_MASK);
#else
  FixNVTKokkos<LMPDeviceType>::final_integrate();
  FixNVTKokkos<LMPDeviceType>::initial_integrate(vflag);
#endif
}



bool FixNVTFusedKokkos::cooperative_nhc_output_initial()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if (!routegraph_output_initial_pending || !dpa_fuse_enabled ||
      !cooperative_nhc_coop_enabled || !packed_nvt_armed ||
      igroup != 0 || packed_nvt_packed_current)
    return false;
  routegraph_output_initial_pending = false;
  compute_temp_target();
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
  const DpaFastStep::NhcParams p = nhc_params();
  auto &buffers = DpaFastStep::nhc_buffers();
  const int parity = buffers.current;
  const auto *in = buffers.state + parity;
  auto *out = buffers.state + (parity ^ 1);
  const DpaFastStep::Arr3<const double> force_view{
      f.data(), static_cast<long>(f.stride_0()),
      static_cast<long>(f.stride_1())};

  DpaFusedCheckArgs check_args{};
  DpaFlatSelfPackArgs pack_args{};
  auto *neighbor_kk = static_cast<NeighborKokkos *>(neighbor);
  auto *comm_kk = static_cast<CommKokkos *>(comm);
  const bool fused_check = neighbor_kk->dpa_prepare_fused_check(check_args);
  const DpaFastStep::Arr3<const double> hold_view{
      check_args.xhold, check_args.xhold_s0, check_args.xhold_s1};
  const bool fused_pack = comm_kk->dpa_prepare_flat_self_pack(pack_args);
  // This path publishes both the displacement check and self-pack as a
  // single phase.  If either preparation is unavailable, abandon the route
  // before changing x/v and let the original initial_integrate run.
  if (!fused_check || !fused_pack) {
    if (fused_check) neighbor_kk->dpa_cancel_fused_check();
    if (fused_pack) comm_kk->dpa_cancel_flat_self_pack();
    return false;
  }

  unsigned long long *packed = nullptr;
  int *skip_flag = nullptr;
  long packed_nodes = 0;
  const bool packed_available = packed_nvt_get_packed(
      packed, skip_flag, packed_nodes) && packed_nodes == nlocal;
  const bigint step = update->ntimestep;
  const bool next_skip = packed_available && step < update->laststep &&
      output && output->next > step && modify->n_end_of_step == 0 &&
      packed_nvt_post_force_safe(modify, step);

  bool launched = false;
  auto launch = [&](auto chain) {
    constexpr int M = decltype(chain)::value;
    if (rmass.data())
      launched = launch_cooperative_nhc_output_initial<1, M>(
          nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v), force_view,
          mask.data(), rmass.data(), mass.data(), type.data(), dtf, dtv,
          groupbit, in, out, p, hold_view, check_args.deltasq,
          check_args.ncheck, check_args.stamp, check_args.flag,
          fused_check ? 1 : 0, pack_args.map, pack_args.map_s0,
          pack_args.map_s1, pack_args.count, pack_args.xprd,
          pack_args.yprd, pack_args.zprd, pack_args.xy, pack_args.xz,
          pack_args.yz, pack_args.triclinic, fused_pack ? 1 : 0,
          skip_flag, next_skip ? 1 : 0);
    else
      launched = launch_cooperative_nhc_output_initial<0, M>(
          nlocal, DpaFastStep::arr3(x), DpaFastStep::arr3(v), force_view,
          mask.data(), rmass.data(), mass.data(), type.data(), dtf, dtv,
          groupbit, in, out, p, hold_view, check_args.deltasq,
          check_args.ncheck, check_args.stamp, check_args.flag,
          fused_check ? 1 : 0, pack_args.map, pack_args.map_s0,
          pack_args.map_s1, pack_args.count, pack_args.xprd,
          pack_args.yprd, pack_args.zprd, pack_args.xy, pack_args.xz,
          pack_args.yz, pack_args.triclinic, fused_pack ? 1 : 0,
          skip_flag, next_skip ? 1 : 0);
  };
  switch (mtchain) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    default: launch(std::integral_constant<int, 5>{}); break;
  }
  if (!launched) {
    if (fused_check) neighbor_kk->dpa_cancel_fused_check();
    if (fused_pack) comm_kk->dpa_cancel_flat_self_pack();
    return false;
  }
  if (fused_check) neighbor_kk->dpa_publish_fused_check();
  if (fused_pack) comm_kk->dpa_publish_flat_self_pack();
  buffers.current ^= 1;
  packed_nvt_packed_current = next_skip;
  atomKK->modified(execution_space, X_MASK | V_MASK);
  dpa_nhc_host_current = false;
  return true;
#else
  routegraph_output_initial_pending = false;
  return false;
#endif
}

bool FixNVTFusedKokkos::cooperative_nhc_output_final()
{
#if defined(KOKKOS_ENABLE_CUDA)
  static const bool requested = [] {
    const char *setting = std::getenv("DPA_SMALL_NVT_OUTPUT_FINAL_COOP");
    return setting && std::atoi(setting) != 0;
  }();
  const bigint step = update->ntimestep;
  const bool thermo_only = output && output->next == step &&
      output->next_thermo == step &&
      (!output->ndump || output->next_dump_any != step) &&
      (!output->restart_flag || output->next_restart != step);
  if (!requested || !dpa_fuse_enabled || !cooperative_nhc_coop_enabled ||
      !packed_nvt_armed || igroup != 0 || !thermo_only ||
      modify->n_end_of_step != 0 || packed_nvt_packed_current)
    return false;

  tdof = temperature->dof;
  atomKK->sync(execution_space,
      V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
  atomKK->k_mass.sync<LMPDeviceType>();
  v = atomKK->k_v.view<LMPDeviceType>();
  f = atomKK->k_f.view<LMPDeviceType>();
  rmass = atomKK->k_rmass.view<LMPDeviceType>();
  mass = atomKK->k_mass.view<LMPDeviceType>();
  type = atomKK->k_type.view<LMPDeviceType>();
  mask = atomKK->k_mask.view<LMPDeviceType>();

  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;
  const DpaFastStep::NhcParams p = nhc_params();
  auto &buffers = DpaFastStep::nhc_buffers();
  const int parity = buffers.current;
  const auto *in = buffers.state + parity;
  auto *out = buffers.state + (parity ^ 1);
  auto &scratch = DpaFastStep::scratch();
  const DpaFastStep::Arr3<const double> force_view{
      f.data(), static_cast<long>(f.stride_0()),
      static_cast<long>(f.stride_1())};

  bool launched = false;
  auto launch = [&](auto chain) {
    constexpr int M = decltype(chain)::value;
    if (rmass.data())
      launched = launch_cooperative_nhc_output_final<1, M>(
          nlocal, DpaFastStep::arr3(v), force_view, mask.data(),
          rmass.data(), mass.data(), type.data(), dtf, groupbit,
          temperature->groupbit, scratch.partial, scratch.result,
          in, out, p);
    else
      launched = launch_cooperative_nhc_output_final<0, M>(
          nlocal, DpaFastStep::arr3(v), force_view, mask.data(),
          rmass.data(), mass.data(), type.data(), dtf, groupbit,
          temperature->groupbit, scratch.partial, scratch.result,
          in, out, p);
  };
  switch (mtchain) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    default: launch(std::integral_constant<int, 5>{}); break;
  }
  if (!launched) return false;

  buffers.current ^= 1;
  auto *tkk = static_cast<ComputeTempKokkos<LMPDeviceType> *>(temperature);
  t_current = tkk->scalar_from_local_sum(
      DpaFastStep::read_result(scratch));
  atomKK->modified(execution_space, V_MASK);
  dpa_nhc_host_current = false;
  routegraph_output_initial_pending = true;
  static bool reported = false;
  if (!reported && comm->me == 0) {
    reported = true;
    utils::logmesg(lmp,
        "DPA_SMALL_NVT_OUTPUT_FINAL_COOP active=1 thermo_only=1\n");
  }
  return true;
#else
  return false;
#endif
}

bool FixNVTFusedKokkos::cooperative_nhc_integrate()
{
#if defined(KOKKOS_ENABLE_CUDA)
  tdof = temperature->dof;
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
  unsigned long long* packed_nvt_packed = nullptr;
  int* packed_nvt_skip_flag = nullptr;
  long packed_nvt_nodes = 0;
  const bool packed_nvt_available = packed_nvt_get_packed(
      packed_nvt_packed, packed_nvt_skip_flag, packed_nvt_nodes) && packed_nvt_nodes == nlocal;
  const bool packed_nvt_use_packed = packed_nvt_packed_current && packed_nvt_available;
  if (packed_nvt_packed_current && !packed_nvt_use_packed) {
    std::fprintf(stderr, "PACKED_NVT packed force disappeared while armed\n");
    std::abort();
  }
  int packed_nvt_materialize_status = 2;
  if (packed_nvt_available && !packed_nvt_use_packed) {
    packed_nvt_materialize_status = packed_nvt_materialize(DpaFastStep::stream());
    if (packed_nvt_materialize_status < 0) {
      std::fprintf(stderr, "PACKED_NVT cold-path force materialization failed\n");
      std::abort();
    }
  }
  const bigint packed_nvt_next_step = update->ntimestep + 1;
  const bool packed_nvt_next_skip = packed_nvt_available && igroup == 0 &&
      packed_nvt_next_step < update->laststep && output &&
      output->next > packed_nvt_next_step && modify->n_end_of_step == 0 &&
      packed_nvt_post_force_safe(modify, packed_nvt_next_step);
  int nintegrate = nlocal;
  if (igroup == atomKK->firstgroup) nintegrate = atomKK->nfirst;
  const DpaFastStep::NhcParams p = nhc_params();
  auto &buffers = DpaFastStep::nhc_buffers();
  const int parity = buffers.current;
  const auto *in = buffers.state + parity;
  auto *out = buffers.state + (parity ^ 1);
  auto &scratch = DpaFastStep::scratch();
  const DpaFastStep::Arr3<const double> force_view{
      f.data(), static_cast<long>(f.stride_0()),
      static_cast<long>(f.stride_1())};
  DpaFusedCheckArgs check_args{};
  DpaFlatSelfPackArgs pack_args{};
  auto *neighbor_kk = static_cast<NeighborKokkos *>(neighbor);
  auto *comm_kk = static_cast<CommKokkos *>(comm);
  // The integrated displacement check is complete only when this fix owns
  // every local atom.  A non-first subgroup can still leave nintegrate equal
  // to nlocal while the mask excludes atoms, so require LAMMPS group all.
  const bool fused_check = igroup == 0 && nintegrate == nlocal &&
      neighbor_kk->dpa_prepare_fused_check(check_args);
  const DpaFastStep::Arr3<const double> hold_view{
      check_args.xhold, check_args.xhold_s0, check_args.xhold_s1};
  const bool fused_pack = igroup == 0 && nintegrate == nlocal &&
      comm_kk->dpa_prepare_flat_self_pack(pack_args);

  bool launched = false;
  auto launch = [&](auto chain) {
    constexpr int M = decltype(chain)::value;
    if (rmass.data())
      launched = launch_cooperative_nhc_cooperative<1, M>(
          nlocal, nintegrate, DpaFastStep::arr3(x),
          DpaFastStep::arr3(v), force_view, mask.data(), rmass.data(),
          mass.data(), type.data(), dtf, dtv, groupbit,
          temperature->groupbit, scratch.partial, scratch.result,
          in, out, p, hold_view, check_args.deltasq, check_args.ncheck,
          check_args.stamp, check_args.flag, fused_check ? 1 : 0,
          pack_args.map, pack_args.map_s0, pack_args.map_s1, pack_args.count,
          pack_args.xprd, pack_args.yprd, pack_args.zprd,
          pack_args.xy, pack_args.xz, pack_args.yz, pack_args.triclinic,
          fused_pack ? 1 : 0, packed_nvt_packed, packed_nvt_use_packed ? 1 : 0,
          packed_nvt_skip_flag, packed_nvt_next_skip ? 1 : 0);
    else
      launched = launch_cooperative_nhc_cooperative<0, M>(
          nlocal, nintegrate, DpaFastStep::arr3(x),
          DpaFastStep::arr3(v), force_view, mask.data(), rmass.data(),
          mass.data(), type.data(), dtf, dtv, groupbit,
          temperature->groupbit, scratch.partial, scratch.result,
          in, out, p, hold_view, check_args.deltasq, check_args.ncheck,
          check_args.stamp, check_args.flag, fused_check ? 1 : 0,
          pack_args.map, pack_args.map_s0, pack_args.map_s1, pack_args.count,
          pack_args.xprd, pack_args.yprd, pack_args.zprd,
          pack_args.xy, pack_args.xz, pack_args.yz, pack_args.triclinic,
          fused_pack ? 1 : 0, packed_nvt_packed, packed_nvt_use_packed ? 1 : 0,
          packed_nvt_skip_flag, packed_nvt_next_skip ? 1 : 0);
  };
  switch (mtchain) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    default: launch(std::integral_constant<int, 5>{}); break;
  }
  if (!launched) {
    if (packed_nvt_use_packed) {
      std::fprintf(stderr, "PACKED_NVT cooperative launch failed while packed force armed\n");
      std::abort();
    }
    packed_nvt_packed_current = false;
    if (fused_check) neighbor_kk->dpa_cancel_fused_check();
    if (fused_pack) comm_kk->dpa_cancel_flat_self_pack();
    cooperative_nhc_coop_enabled = false;
    if (comm->me == 0)
      utils::logmesg(lmp,
          "DPA_COOPERATIVE_NHC_COOP_NHC disabled after launch/capacity failure\\n");
    return false;
  }
  if (fused_check) neighbor_kk->dpa_publish_fused_check();
  if (fused_pack) comm_kk->dpa_publish_flat_self_pack();
  buffers.current ^= 1;
  packed_nvt_packed_current = packed_nvt_next_skip;
  atomKK->modified(execution_space, X_MASK | V_MASK);
  return true;
#else
  return false;
#endif
}

// The NHC recurrence runs on the PPU, while FixNH observation and native
// restart serialization read host eta/eta_dot. Mirror the current phase
// lazily on an actual host read; ordinary integration steps stay on device.
void FixNVTFusedKokkos::sync_nhc_host_read()
{
  if (nhc_device && !nhc_upload_pending && !dpa_nhc_host_current) {
    nhc_download();
    dpa_nhc_host_current = true;
  }
}

void FixNVTFusedKokkos::initial_integrate(int vflag)
{
  if (cooperative_nhc_output_initial()) return;
  FixNVTKokkos<LMPDeviceType>::initial_integrate(vflag);
  dpa_nhc_host_current = !nhc_device;
}

void FixNVTFusedKokkos::final_integrate()
{
  if (packed_nvt_armed && packed_nvt_materialize(DpaFastStep::stream()) < 0) {
    std::fprintf(stderr, "PACKED_NVT final force materialization failed\n");
    std::abort();
  }
  if (cooperative_nhc_output_final()) return;
  FixNVTKokkos<LMPDeviceType>::final_integrate();
  dpa_nhc_host_current = !nhc_device;
}

void FixNVTFusedKokkos::post_run()
{
  routegraph_output_initial_pending = false;
  FixNVTKokkos<LMPDeviceType>::post_run();
  dpa_nhc_host_current = true;
}

void FixNVTFusedKokkos::restart(char *buffer)
{
  routegraph_output_initial_pending = false;
  FixNH::restart(buffer);
  // If an existing fix is restored, the restored host chain wins over the
  // old device state and is uploaded by the next initial integration.
  nhc_upload_pending = nhc_device;
  dpa_nhc_host_current = true;
}

double FixNVTFusedKokkos::compute_scalar()
{
  sync_nhc_host_read();
  return FixNH::compute_scalar();
}

double FixNVTFusedKokkos::compute_vector(int index)
{
  sync_nhc_host_read();
  return FixNH::compute_vector(index);
}

int FixNVTFusedKokkos::pack_restart_data(double *buffer)
{
  sync_nhc_host_read();
  return FixNH::pack_restart_data(buffer);
}
