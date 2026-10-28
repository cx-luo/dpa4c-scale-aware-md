/* -*- c++ -*- ----------------------------------------------------------
   Short per-step device paths for the Kokkos Verlet control chain.

   Kokkos parallel_reduce into a host scalar costs a multi-stage reduction
   kernel plus two stream fences on the PPU. The helpers here replace it with
   one deterministic single-pass reduction (fixed grid, fixed per-thread
   order, last block sums block partials in index order) and one D2H copy.
   DPA_LMP_FAST_STEP is a bit mask (default 7): 1 neighbor distance check,
   2 fused FixNH initial_integrate, 4 fused FixNH nve_v + temperature,
   8 asynchronous distance check with a speculative force pass, 16 NVT
   Nose-Hoover chain on the device (no temperature read back per step).
   DPA_LMP_FAST_STEP=0 restores the original Kokkos code paths.
------------------------------------------------------------------------- */

#ifndef LMP_DPA_FAST_STEP_KOKKOS_H
#define LMP_DPA_FAST_STEP_KOKKOS_H

#include "kokkos_type.h"

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace LAMMPS_NS {
namespace DpaFastStep {

enum : int { kCheck = 1, kInitial = 2, kFinal = 4, kAsync = 8, kDeviceNhc = 16 };

inline bool enabled(int bit)
{
  static const int mask = [] {
    const char *s = std::getenv("DPA_LMP_FAST_STEP");
    return (s && s[0]) ? std::atoi(s) : 7;
  }();
  return (mask & bit) != 0;
}

// Bits 8 and 16 help the three small cases and slow Si-medium. When
// DPA_LMP_FAST_MAX_ATOMS is a positive atom count, those bits stay off
// above that count. Unset or 0 keeps the mask for every system size.
inline bool within_atom_limit(long long natoms)
{
  static const long long limit = [] {
    const char *s = std::getenv("DPA_LMP_FAST_MAX_ATOMS");
    return (s && s[0]) ? std::atoll(s) : 0;
  }();
  return limit <= 0 || natoms <= limit;
}

// Keep the broad fast-step limit unchanged while allowing a separately
// guarded NVT-only admission route.  Zero/unset preserves production.
inline bool within_device_nhc_limit(long long natoms)
{
  if (within_atom_limit(natoms)) return true;
  static const long long limit = [] {
    const char *s = std::getenv("DPA_NVT_DEVICE_MAX_ATOMS");
    return (s && s[0]) ? std::atoll(s) : 0;
  }();
  return limit > 0 && natoms <= limit;
}

// DPA_LMP_FAST_STEP_VERIFY=1: run the original path after every fast call,
// report the difference and continue with the original result.
inline bool verify()
{
  static const bool on = [] {
    const char *s = std::getenv("DPA_LMP_FAST_STEP_VERIFY");
    return s && s[0] == '1';
  }();
  return on;
}

struct VerifyStats {
  const char *what;
  long calls = 0, mismatches = 0;
  double max_a = 0.0, max_b = 0.0;
  explicit VerifyStats(const char *w) : what(w) {}
  void add(bool mismatch, double a, double b)
  {
    ++calls;
    if (mismatch) ++mismatches;
    if (a > max_a) max_a = a;
    if (b > max_b) max_b = b;
    if (calls % 500 == 0 || (mismatch && mismatches <= 5))
      std::fprintf(stderr, "DPA_FAST_VERIFY %s calls=%ld mismatches=%ld max_a=%.3e max_b=%.3e\n",
                   what, calls, mismatches, max_a, max_b);
  }
};

template <class DeviceType>
constexpr bool is_cuda_device()
{
#if defined(KOKKOS_ENABLE_CUDA)
  return std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value;
#else
  return false;
#endif
}

#if defined(KOKKOS_ENABLE_CUDA)

constexpr int kBlock = 256;
constexpr int kMaxBlocks = 512;

// Raw pointer plus strides: PPU host stubs cannot take View<T*[3]> kernel
// parameters (see compat kokkos-ppu-kernel-argument.patch).
template <class T> struct Arr3 {
  T *p;
  long s0, s1;
  __device__ T &operator()(int i, int k) const { return p[i * s0 + k * s1]; }
};

template <class View> Arr3<typename View::value_type> arr3(const View &v)
{
  return {v.data(), static_cast<long>(v.stride_0()), static_cast<long>(v.stride_1())};
}

struct Scratch {
  double *partial = nullptr;
  unsigned *ticket = nullptr;
  double *result = nullptr;
  int *flag = nullptr;
  double *host_result = nullptr;
  int *host_flag = nullptr;
  int stamp = 0;
  cudaEvent_t flag_ready = nullptr;
};

inline Scratch &scratch()
{
  static Scratch s;
  if (!s.partial) {
    cudaMalloc(&s.partial, kMaxBlocks * sizeof(double));
    cudaMalloc(&s.ticket, sizeof(unsigned));
    cudaMalloc(&s.result, sizeof(double));
    cudaMalloc(&s.flag, sizeof(int));
    cudaMemset(s.ticket, 0, sizeof(unsigned));
    cudaMemset(s.flag, 0, sizeof(int));
    cudaHostAlloc(&s.host_result, sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc(&s.host_flag, sizeof(int), cudaHostAllocDefault);
    cudaEventCreateWithFlags(&s.flag_ready, cudaEventDisableTiming);
    cudaDeviceSynchronize();
  }
  return s;
}

inline cudaStream_t stream() { return Kokkos::Cuda().cuda_stream(); }

inline int grid_for(int n)
{
  int g = (n + kBlock - 1) / kBlock;
  if (g < 1) g = 1;
  return g < kMaxBlocks ? g : kMaxBlocks;
}

__device__ inline double block_sum(double v, double *sh)
{
  __syncthreads();
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? sh[lane] : 0.0;
  if (warp == 0)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__device__ inline void grid_sum_finish(double acc, double *partial, unsigned *ticket, double *result)
{
  __shared__ double sh[32];
  __shared__ bool last;
  const double b = block_sum(acc, sh);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = b;
    __threadfence();
    last = atomicAdd(ticket, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  const volatile double *vp = partial;
  double t = 0.0;
  for (int k = threadIdx.x; k < (int) gridDim.x; k += blockDim.x) t += vp[k];
  t = block_sum(t, sh);
  if (threadIdx.x == 0) {
    *result = t;
    *ticket = 0u;
  }
}

struct VUpdate {
  double scale;       // FixNH::nh_v_temp factor_eta
  double dtf;         // FixNH::nve_v
  double press[3];    // FixNH::nh_v_press factor (non-triclinic)
};

// Optional nh_v_temp, nve_v and nh_v_press in FixNH order, then the
// ComputeTemp::compute_scalar per-atom sum, with identical arithmetic.
template <int RMASS, int SCALE, int NVE, int PRESS>
__global__ void __launch_bounds__(kBlock)
v_update_temp_kernel(int n, Arr3<double> v, Arr3<const double> f, const int *mask,
                     const double *rmass, const double *mass, const int *type, VUpdate u,
                     int groupbit, int tgroupbit, double *partial, unsigned *ticket, double *result)
{
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int m = mask[i];
    const double massone = RMASS ? rmass[i] : mass[type[i]];
    if (m & groupbit) {
      if (SCALE) {
        v(i, 0) *= u.scale;
        v(i, 1) *= u.scale;
        v(i, 2) *= u.scale;
      }
      if (NVE) {
        const double dtfm = u.dtf / massone;
        v(i, 0) += dtfm * f(i, 0);
        v(i, 1) += dtfm * f(i, 1);
        v(i, 2) += dtfm * f(i, 2);
      }
      if (PRESS) {
        v(i, 0) *= u.press[0];
        v(i, 1) *= u.press[1];
        v(i, 2) *= u.press[2];
        v(i, 0) *= u.press[0];
        v(i, 1) *= u.press[1];
        v(i, 2) *= u.press[2];
      }
    }
    if (m & tgroupbit)
      acc += (v(i, 0) * v(i, 0) + v(i, 1) * v(i, 1) + v(i, 2) * v(i, 2)) * massone;
  }
  grid_sum_finish(acc, partial, ticket, result);
}

template <int RMASS, int SCALE, int NVE, int PRESS>
void launch_v_update_temp(int n, Arr3<double> v, Arr3<const double> f, const int *mask,
                          const double *rmass, const double *mass, const int *type,
                          const VUpdate &u, int groupbit, int tgroupbit)
{
  Scratch &s = scratch();
  v_update_temp_kernel<RMASS, SCALE, NVE, PRESS><<<grid_for(n), kBlock, 0, stream()>>>(
      n, v, f, mask, rmass, mass, type, u, groupbit, tgroupbit, s.partial, s.ticket, s.result);
}

static __global__ void __launch_bounds__(kBlock)
check_distance_kernel(int n, Arr3<const double> x, Arr3<const double> xhold, double deltasq,
                      int stamp, int *flag)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const double delx = x(i, 0) - xhold(i, 0);
    const double dely = x(i, 1) - xhold(i, 1);
    const double delz = x(i, 2) - xhold(i, 2);
    if (delx * delx + dely * dely + delz * delz > deltasq) *flag = stamp;
  }
}

inline double read_result(const Scratch &s)
{
  cudaMemcpyAsync(s.host_result, s.result, sizeof(double), cudaMemcpyDeviceToHost, stream());
  cudaStreamSynchronize(stream());
  return *s.host_result;
}

template <class XView, class HoldView>
int check_distance(int n, const XView &x, const HoldView &xhold, double deltasq)
{
  Scratch &s = scratch();
  if (++s.stamp <= 0) {
    s.stamp = 1;
    cudaMemsetAsync(s.flag, 0, sizeof(int), stream());
  }
  if (n > 0)
    check_distance_kernel<<<grid_for(n), kBlock, 0, stream()>>>(
        n, {x.data(), (long) x.stride_0(), (long) x.stride_1()},
        {xhold.data(), (long) xhold.stride_0(), (long) xhold.stride_1()}, deltasq, s.stamp, s.flag);
  cudaMemcpyAsync(s.host_flag, s.flag, sizeof(int), cudaMemcpyDeviceToHost, stream());
  cudaStreamSynchronize(stream());
  return *s.host_flag == s.stamp;
}

// Split form of check_distance(): the launch returns at once and the wait
// blocks only until the flag copy has landed, not until the stream drains.
template <class XView, class HoldView>
void check_distance_launch(int n, const XView &x, const HoldView &xhold, double deltasq)
{
  Scratch &s = scratch();
  if (++s.stamp <= 0) {
    s.stamp = 1;
    cudaMemsetAsync(s.flag, 0, sizeof(int), stream());
  }
  if (n > 0)
    check_distance_kernel<<<grid_for(n), kBlock, 0, stream()>>>(
        n, {x.data(), (long) x.stride_0(), (long) x.stride_1()},
        {xhold.data(), (long) xhold.stride_0(), (long) xhold.stride_1()}, deltasq, s.stamp, s.flag);
  cudaMemcpyAsync(s.host_flag, s.flag, sizeof(int), cudaMemcpyDeviceToHost, stream());
  cudaEventRecord(s.flag_ready, stream());
}

inline int check_distance_wait()
{
  Scratch &s = scratch();
  cudaEventSynchronize(s.flag_ready);
  return *s.host_flag == s.stamp;
}

/* Nose-Hoover chain on the device (NVT, nc_tchain == 1).
   FixNH::nhc_temp_integrate() is a scalar recurrence driven by the current
   temperature. Thread 0 of every block of the velocity kernels evaluates it
   from the same input state and shares the scale factor with its block, so
   no separate launch is needed; block 0 writes the new state into the other
   buffer.
   FP64 exp and division each cost ~0.3 us of latency on the PPU, so the
   chain masses and their reciprocals come from the host and exp of the small
   chain arguments is a Taylor polynomial; results differ from the host
   recurrence at the ulp level only. */

constexpr int kMaxChain = 5;

struct NhcState {
  double eta[kMaxChain];
  double eta_dot[kMaxChain + 1];
  double eta_dotdot[kMaxChain];
  double eta_mass[kMaxChain];
  double t_current;
  double factor_eta;
};

struct NhcParams {
  double tdof_boltz, ke_target, boltz_t_target, dt4, dt8, dthalf, tdrag_factor, tfactor;
  double eta_mass[kMaxChain], inv_eta_mass[kMaxChain];
};

__device__ __forceinline__ double nhc_exp(double x)
{
  if (fabs(x) >= 1e-3) return exp(x);
  return 1.0 + x * (1.0 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24 + x * (1.0 / 120 + x * (1.0 / 720 +
         x * (1.0 / 5040)))))));
}

// FixNH::nhc_temp_integrate() with nc_tchain == 1 and mtchain == M, same
// expression order. t_current < 0 keeps the value stored in the state.
template <int M>
__device__ inline double nhc_chain(const NhcState *in, NhcState *out, double t_current, const NhcParams &p)
{
  double eta[M], eta_dot[M + 1], eta_dotdot[M];
#pragma unroll
  for (int i = 0; i < M; i++) {
    eta[i] = in->eta[i];
    eta_dotdot[i] = in->eta_dotdot[i];
  }
#pragma unroll
  for (int i = 0; i <= M; i++) eta_dot[i] = in->eta_dot[i];
  if (t_current < 0.0) t_current = in->t_current;

  double expfac;
  double kecurrent = p.tdof_boltz * t_current;
  eta_dotdot[0] = p.eta_mass[0] > 0.0 ? (kecurrent - p.ke_target) * p.inv_eta_mass[0] : 0.0;
#pragma unroll
  for (int ich = M - 1; ich > 0; ich--) {
    expfac = nhc_exp(-p.dt8 * eta_dot[ich + 1]);
    eta_dot[ich] *= expfac;
    eta_dot[ich] += eta_dotdot[ich] * p.dt4;
    eta_dot[ich] *= p.tdrag_factor;
    eta_dot[ich] *= expfac;
  }
  expfac = nhc_exp(-p.dt8 * eta_dot[1]);
  eta_dot[0] *= expfac;
  eta_dot[0] += eta_dotdot[0] * p.dt4;
  eta_dot[0] *= p.tdrag_factor;
  eta_dot[0] *= expfac;
  const double factor_eta = nhc_exp(-p.dthalf * eta_dot[0]);
  t_current *= factor_eta * factor_eta;
  kecurrent = p.tdof_boltz * t_current;
  eta_dotdot[0] = p.eta_mass[0] > 0.0 ? (kecurrent - p.ke_target) * p.inv_eta_mass[0] : 0.0;
#pragma unroll
  for (int ich = 0; ich < M; ich++) eta[ich] += p.dthalf * eta_dot[ich];
  eta_dot[0] *= expfac;
  eta_dot[0] += eta_dotdot[0] * p.dt4;
  eta_dot[0] *= expfac;
#pragma unroll
  for (int ich = 1; ich < M; ich++) {
    expfac = nhc_exp(-p.dt8 * eta_dot[ich + 1]);
    eta_dot[ich] *= expfac;
    eta_dotdot[ich] = (p.eta_mass[ich - 1] * eta_dot[ich - 1] * eta_dot[ich - 1] - p.boltz_t_target) *
        p.inv_eta_mass[ich];
    eta_dot[ich] += eta_dotdot[ich] * p.dt4;
    eta_dot[ich] *= expfac;
  }

  if (out) {
#pragma unroll
    for (int i = 0; i < M; i++) {
      out->eta[i] = eta[i];
      out->eta_dotdot[i] = eta_dotdot[i];
      out->eta_mass[i] = p.eta_mass[i];
    }
#pragma unroll
    for (int i = 0; i <= M; i++) out->eta_dot[i] = eta_dot[i];
    out->t_current = t_current;
    out->factor_eta = factor_eta;
  }
  return factor_eta;
}

// End of final_integrate: t_current from the reduction of v_update_temp,
// chain update, then nh_v_temp with the new factor.
template <int M>
__global__ void __launch_bounds__(kBlock)
nhc_final_kernel(int n, Arr3<double> v, const int *mask, int groupbit, const double *sum,
                 const NhcState *in, NhcState *out, NhcParams p)
{
  __shared__ double shared_factor;
  if (threadIdx.x == 0)
    shared_factor = nhc_chain<M>(in, blockIdx.x == 0 ? out : nullptr, *sum * p.tfactor, p);
  __syncthreads();
  const double factor = shared_factor;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (mask[i] & groupbit) {
      v(i, 0) *= factor;
      v(i, 1) *= factor;
      v(i, 2) *= factor;
    }
  }
}

// NVT initial_integrate: chain update, then nh_v_temp + nve_v + nve_x with
// the arithmetic of nh_small_vx.
template <int RMASS, int M>
__global__ void __launch_bounds__(kBlock)
nhc_initial_kernel(int n, Arr3<double> x, Arr3<double> v, Arr3<const double> f, const int *mask,
                   const double *rmass, const double *mass, const int *type, double dtf, double dtv,
                   int groupbit, const NhcState *in, NhcState *out, NhcParams p)
{
  __shared__ double shared_factor;
  if (threadIdx.x == 0) shared_factor = nhc_chain<M>(in, blockIdx.x == 0 ? out : nullptr, -1.0, p);
  __syncthreads();
  const double fe = shared_factor;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (mask[i] & groupbit) {
      v(i, 0) *= fe;
      v(i, 1) *= fe;
      v(i, 2) *= fe;
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

struct NhcBuffers {
  NhcState *state = nullptr;    // two device states, used alternately
  int current = 0;
};

inline NhcBuffers &nhc_buffers()
{
  static NhcBuffers b;
  if (!b.state) cudaMalloc(&b.state, 2 * sizeof(NhcState));
  return b;
}

#endif    // KOKKOS_ENABLE_CUDA

}    // namespace DpaFastStep
}    // namespace LAMMPS_NS

#endif
