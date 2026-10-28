// SPDX-License-Identifier: LGPL-3.0-or-later
// 70-96-96-96 silu fitting, energy and input gradient, on FP16 tensor cores.
//
// Every FP32 operand is split into power-of-two scaled FP16 halves and each
// product is taken as lo*lo + lo*hi + hi*lo + hi*hi.  FITTING_IMAGE restores the
// low-by-low term omitted by K3N so the Tensor Core path more closely follows
// the full FP32 fitting network. The PPU MMA truncates products below one ulp of its
// running accumulator, so the accumulator restarts every kPeriod k-blocks and
// the partials are summed in FP32.
//
// A warp owns 16 nodes. Both weight directions are stored pre-arranged in
// fragment order (one 128-bit shared load per lane per fragment), which fills
// 208 KB of shared memory, so sigma' of layers 0/1 lives in a per-warp global
// scratch that stays L2 resident. The C fragment of one layer is the A
// fragment of the next after permuting k inside each 16-block of layers 1/2:
//   A[m][k]: x[0..3] = row g, k {2t,2t+1,2t+8,2t+9}; x[4..7] = row g+8
//   B[k][n]: x[0..3] = col g, same k;                x[4..7] = col g+8
//   C[m][n]: x[0..3] = row g, n {t,t+4,t+8,t+12};    x[4..7] = row g+8
// (g = lane >> 2, t = lane & 3, measured on ZW810E.)
#ifndef K3N_DEVICE_ONLY
#if __has_include("dpa_l2_persist.h")
#include "l2_persist.h"
#else
#include "l2_persist.h"
#endif
#endif
namespace k3n {
using namespace nvcuda;

constexpr int kH = 96;
constexpr int kIn = 70;
constexpr float kScaleX = 8192.f, kScaleH = 1024.f, kScaleT = 1024.f;
#ifndef FITTING_IMAGE_KPERIOD
#define FITTING_IMAGE_KPERIOD 6
#endif
#ifndef FITTING_IMAGE_PRODUCTS
#define FITTING_IMAGE_PRODUCTS 4
#endif
static_assert(FITTING_IMAGE_PRODUCTS == 3 || FITTING_IMAGE_PRODUCTS == 4, "FITTING_IMAGE_PRODUCTS must be 3 or 4");
constexpr int kPeriod = FITTING_IMAGE_KPERIOD;
constexpr int kProducts = FITTING_IMAGE_PRODUCTS;
#ifndef SPLIT_FITTING_MAX_WARPS
#define SPLIT_FITTING_MAX_WARPS 16
#endif
constexpr int kMaxWarps = SPLIT_FITTING_MAX_WARPS;
constexpr int kMinWarps = 8;
constexpr int kFwdFrags = (5 + 6 + 6) * 6;        // k-blocks x 6 n-blocks
constexpr int kBwdFrags = 6 * 6 + 6 * 6 + 6 * 5;  // layers 2, 1, 0
constexpr int kFragBytes = 512;
constexpr int kFragArea = (kFwdFrags + kBwdFrags) * 2 * kFragBytes;
// After the fragments: B[3][96]; HD[100] = head, unused, weight scales of
// layers 0..2; E in C-fragment order ([nb][t][j] <-> neuron nb*16 + t + 4j):
// half biases of the three layers, head, head * kScaleT / 2.
constexpr int kEpiFloats = 5 * kH;
constexpr int kImageBytes = kFragArea + (3 * kH + 100 + kEpiFloats) * 4;
constexpr int kScratchPerWarp = 2 * 48 * 32;
static_assert(kImageBytes % 16 == 0, "image is copied in 16-byte words");

__host__ __device__ constexpr int pi16(int s) {
  return ((s >> 1) & 3) + 4 * (s & 1) + 8 * (s >> 3);
}
__host__ __device__ constexpr int epi_neuron(int i) {
  return (i / 16) * 16 + (i % 4) * 4 + (i % 16) / 4;
}
__host__ __device__ constexpr int fwd_kb_base(int l) {
  return l == 0 ? 0 : (l == 1 ? 5 : 11);
}
__host__ __device__ constexpr int bwd_base(int l) {
  return l == 2 ? 0 : (l == 1 ? 36 : 72);
}

struct Params {
  const float* w[3];
  const float* b[3];
  const float* head;
};

__device__ __forceinline__ float* image_hd(unsigned char* img) {
  return reinterpret_cast<float*>(img + kFragArea) + 3 * kH;
}

#ifndef K3N_DEVICE_ONLY
// Power-of-two scale that maps max|W| of each layer just below 2^14.
__global__ void weight_scale_kernel(Params p, unsigned char* img) {
  __shared__ float part[256];
  const int l = blockIdx.x;
  const int count = (l == 0 ? kIn : kH) * kH;
  float m = 0.f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) m = fmaxf(m, fabsf(p.w[l][i]));
  part[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) part[threadIdx.x] = fmaxf(part[threadIdx.x], part[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float top = part[0];
    image_hd(img)[97 + l] = top > 0.f ? ldexpf(1.f, ilogbf(__fdiv_rn(16384.f, top))) : 1.f;
  }
}

__global__ void build_image_kernel(Params p, unsigned char* img) {
  __half* frag = reinterpret_cast<__half*>(img);
  float* B = reinterpret_cast<float*>(img + kFragArea);
  float* HD = B + 3 * kH;
  float* E = HD + 100;
  const int total = (kFwdFrags + kBwdFrags) * 256;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const int f = i / 256, lane = (i / 8) % 32, e = i % 8;
    const int gg = lane >> 2, t = lane & 3, m = e & 3;
    const int col = gg + 8 * (e >> 2);
    const int s = 2 * t + (m & 1) + 8 * (m >> 1);
    int l, k, n;
    if (f < kFwdFrags) {
      const int kbg = f / 6, nb = f % 6;
      l = kbg < 5 ? 0 : (kbg < 11 ? 1 : 2);
      const int kb = kbg - fwd_kb_base(l);
      n = nb * 16 + col;
      k = kb * 16 + (l == 0 ? s : pi16(s));
    } else {
      const int fb = f - kFwdFrags;
      l = fb < 36 ? 2 : (fb < 72 ? 1 : 0);
      const int NK = l == 0 ? 5 : 6, r = fb - bwd_base(l), kb = r / NK, nk = r % NK;
      n = kb * 16 + pi16(s);
      k = nk * 16 + col;
    }
    const int rows = l == 0 ? kIn : kH;
    const float v = k < rows ? p.w[l][k * kH + n] * HD[97 + l] : 0.f;
    const __half hi = __float2half_rn(v);
    frag[(2 * f) * 256 + lane * 8 + e] = hi;
    frag[(2 * f + 1) * 256 + lane * 8 + e] = __float2half_rn(v - __half2float(hi));
  }
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < 3 * kH; i += blockDim.x) {
      const float* b = p.b[i / kH];
      B[i] = b ? b[i % kH] : 0.f;
      E[i] = b ? 0.5f * b[epi_neuron(i % kH)] : 0.f;
    }
    for (int i = threadIdx.x; i < kH; i += blockDim.x) {
      HD[i] = p.head[i];
      E[3 * kH + i] = p.head[epi_neuron(i)];
      E[4 * kH + i] = p.head[epi_neuron(i)] * (0.5f * kScaleT);
    }
  }
}
#endif

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

struct Ctx {
  const unsigned char* img;
  const float* HD;
  const float* E;
  float* sd;
  int frag0;
  int lane, gr, tq;
};

__device__ __forceinline__ void split_frag(const float (&v)[8], FragA& hi, FragA& lo) {
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const __half2 h2 = __floats2half2_rn(v[i], v[i + 1]);
    const float2 b = __half22float2(h2);
    *reinterpret_cast<__half2*>(&hi.x[i]) = h2;
    *reinterpret_cast<__half2*>(&lo.x[i]) = __floats2half2_rn(v[i] - b.x, v[i + 1] - b.y);
  }
}

__device__ __forceinline__ void load_fragb(FragB& hi, FragB& lo, const unsigned char* img, int f, int lane) {
#if defined(SPLIT_FITTING_LOAD64) && SPLIT_FITTING_LOAD64
  auto* hi_words = reinterpret_cast<unsigned long long*>(&hi.x[0]);
  auto* lo_words = reinterpret_cast<unsigned long long*>(&lo.x[0]);
  const auto* high_source = reinterpret_cast<const unsigned long long*>(
      img + (2 * f) * kFragBytes + lane * 16);
  const auto* low_source = reinterpret_cast<const unsigned long long*>(
      img + (2 * f + 1) * kFragBytes + lane * 16);
  hi_words[0] = high_source[0];
  hi_words[1] = high_source[1];
  lo_words[0] = low_source[0];
  lo_words[1] = low_source[1];
#else
  *reinterpret_cast<uint4*>(&hi.x[0]) = *reinterpret_cast<const uint4*>(img + (2 * f) * kFragBytes + lane * 16);
  *reinterpret_cast<uint4*>(&lo.x[0]) = *reinterpret_cast<const uint4*>(img + (2 * f + 1) * kFragBytes + lane * 16);
#endif
}

template <int G>
__device__ __forceinline__ void mma3(FragC* acc, const FragA& ahi, const FragA& alo, const FragB* bh, const FragB* bl) {
#pragma unroll
#if FITTING_IMAGE_PRODUCTS == 4
  for (int i = 0; i < G; ++i) wmma::mma_sync(acc[i], alo, bl[i], acc[i]);
#pragma unroll
#endif
  for (int i = 0; i < G; ++i) wmma::mma_sync(acc[i], alo, bh[i], acc[i]);
#pragma unroll
  for (int i = 0; i < G; ++i) wmma::mma_sync(acc[i], ahi, bl[i], acc[i]);
#pragma unroll
  for (int i = 0; i < G; ++i) wmma::mma_sync(acc[i], ahi, bh[i], acc[i]);
}

__device__ __forceinline__ float4 epi4(const Ctx& c, int off, int nb) {
  return *reinterpret_cast<const float4*>(c.E + off + nb * 16 + c.tq * 4);
}
__device__ __forceinline__ float f4get(const float4& v, int j) {
  return j == 0 ? v.x : (j == 1 ? v.y : (j == 2 ? v.z : v.w));
}
// sigma' scratch: [layer][nb][half][lane][4].
__device__ __forceinline__ float4* sd4(const Ctx& c, int L, int nb, int half) {
  return reinterpret_cast<float4*>(c.sd) + (L * 12 + nb * 2 + half) * 32 + c.lane;
}

// Without the pins the scheduler hoists later k-blocks' independent MMA chains
// above the FP32 partial sums and spills ~1 KB per thread.
__device__ __forceinline__ void pin8(float* v) {
  asm volatile("" : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3]),
                    "+f"(v[4]), "+f"(v[5]), "+f"(v[6]), "+f"(v[7]));
}

template <int NB>
__device__ __forceinline__ void restart(int kb, FragC* acc) {
  if (kb % kPeriod == 0 && kb > 0) {
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) wmma::fill_fragment(acc[nb], 0.f);
  }
}
template <int NB>
__device__ __forceinline__ void retire(int kb, int KB, FragC* acc, float (*accf)[8]) {
  if (kb % kPeriod == kPeriod - 1 || kb == KB - 1) {
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
      for (int e = 0; e < 8; ++e) accf[nb][e] += acc[nb].x[e];
      pin8(accf[nb]);
    }
  }
}

// All x loads of a task are issued together: one global latency per task.
__device__ __forceinline__ void load_x(float2 (&xq)[5][4], const float* x, long r0, long r1, long n, int tq) {
#pragma unroll
  for (int kb = 0; kb < 5; ++kb) {
    const int k0 = kb * 16 + 2 * tq, k1 = k0 + 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) xq[kb][i] = make_float2(0.f, 0.f);
    if (r0 < n && k0 < kIn) xq[kb][0] = *reinterpret_cast<const float2*>(x + r0 * kIn + k0);
    if (r0 < n && k1 < kIn) xq[kb][1] = *reinterpret_cast<const float2*>(x + r0 * kIn + k1);
    if (r1 < n && k0 < kIn) xq[kb][2] = *reinterpret_cast<const float2*>(x + r1 * kIn + k0);
    if (r1 < n && k1 < kIn) xq[kb][3] = *reinterpret_cast<const float2*>(x + r1 * kIn + k1);
  }
}

// silu on hz = z/2 with t = tanh.approx(hz): silu = hz + hz*t and
// silu' = (1 + t + hz*(1 - t^2)) / 2. The stored derivative carries the
// backward operand scale kScaleT (and the head weight for layer 2).
template <int L, int L2Half = -1, bool SharedLayer2 = false>
__device__ __forceinline__ void forward(const Ctx& c, float (&h)[6][8], float (&d2)[6][8], float (&p)[2],
                                        const float2 (&xq)[5][4]) {
  constexpr float sa = L == 0 ? kScaleX : kScaleH;
  constexpr int KB = L == 0 ? 5 : 6;
  const float s = sa * c.HD[97 + L], inv = 1.f / s;
  FragC acc[6];
  float accf[6][8];
#pragma unroll
  for (int nb = 0; nb < 6; ++nb) {
    wmma::fill_fragment(acc[nb], 0.f);
#pragma unroll
    for (int e = 0; e < 8; ++e) accf[nb][e] = 0.f;
  }
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) {
    float v[8];
    if constexpr (L == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[2 * i] = xq[kb][i].x * sa;
        v[2 * i + 1] = xq[kb][i].y * sa;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = h[kb][i] * sa;
    }
    pin8(v);
    restart<6>(kb, acc);
    FragA ahi, alo;
    split_frag(v, ahi, alo);
    if constexpr (L == 2 && L2Half >= 0) {
      constexpr int q = L2Half;
      FragB bh[3], bl[3];
#pragma unroll
      for (int i = 0; i < 3; ++i)
        load_fragb(bh[i], bl[i], c.img,
                   (fwd_kb_base(L) + kb) * 6 + q * 3 + i - c.frag0,
                   c.lane);
      mma3<3>(acc + q * 3, ahi, alo, bh, bl);
    } else {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        FragB bh[3], bl[3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
          load_fragb(bh[i], bl[i], c.img,
                     (fwd_kb_base(L) + kb) * 6 + q * 3 + i - c.frag0,
                     c.lane);
        mma3<3>(acc + q * 3, ahi, alo, bh, bl);
      }
    }
    retire<6>(kb, KB, acc, accf);
  }
#pragma unroll
  for (int nb = 0; nb < 6; ++nb) {
    const bool col2_skip = (L == 2 && L2Half >= 0 && (nb < L2Half * 3 || nb >= L2Half * 3 + 3));
    if (!col2_skip) {
    const float4 bh4 = epi4(c, L * kH, nb);
    float4 hp4, hq4;
    if constexpr (L == 2) {
      hp4 = epi4(c, 3 * kH, nb);
      hq4 = epi4(c, 4 * kH, nb);
    }
    float sdv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int j = i & 3;
      const float hz = fmaf(accf[nb][i], 0.5f * inv, f4get(bh4, j));
      float t;
      asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(hz));
      const float av = fmaf(hz, t, hz);
      const float w = fmaf(hz, fmaf(-t, t, 1.f), t);
      float cst = 0.5f * kScaleT;
      if constexpr (L == 2) cst = f4get(hq4, j);
      const float dzs = fmaf(w, cst, cst);
      if constexpr (L == 2) {
        p[i >> 2] = fmaf(h[nb][i] + av, f4get(hp4, j), p[i >> 2]);
        if constexpr(SharedLayer2) sdv[i] = dzs;
        else d2[nb][i] = dzs;
      } else {
        sdv[i] = dzs;
        if constexpr (L == 0) h[nb][i] = av;
        else h[nb][i] += av;
      }
    }
    if constexpr (L < 2 || SharedLayer2) {
      *sd4(c, L, nb, 0) = make_float4(sdv[0], sdv[1], sdv[2], sdv[3]);
      *sd4(c, L, nb, 1) = make_float4(sdv[4], sdv[5], sdv[6], sdv[7]);
    }
    }
  }
}

template <int L, bool SharedLayer2 = false>
__device__ __forceinline__ void backward(const Ctx& c, float (&g)[6][8], const float (&d2)[6][8]) {
  constexpr int NK = L == 0 ? 5 : 6;
  const float s = kScaleT * c.HD[97 + L], inv = 1.f / s;
  FragC acc[6];
  float accf[6][8];
#pragma unroll
  for (int nb = 0; nb < NK; ++nb) {
    wmma::fill_fragment(acc[nb], 0.f);
#pragma unroll
    for (int e = 0; e < 8; ++e) accf[nb][e] = 0.f;
  }
#pragma unroll
  for (int kb = 0; kb < 6; ++kb) {
    float4 sdc[2];
    if constexpr (L < 2 || SharedLayer2) {
      sdc[0] = *sd4(c, L, kb, 0);
      sdc[1] = *sd4(c, L, kb, 1);
    }
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (L == 2) {
        if constexpr(SharedLayer2) v[i] = f4get(sdc[i >> 2], i & 3);
        else v[i] = d2[kb][i];
      } else v[i] = g[kb][i] * f4get(sdc[i >> 2], i & 3);
    }
    pin8(v);
    restart<NK>(kb, acc);
    FragA ahi, alo;
    split_frag(v, ahi, alo);
#pragma unroll
    for (int q = 0; q < (NK + 2) / 3; ++q) {
      FragB bh[3], bl[3];
#pragma unroll
      for (int i = 0; i < 3; ++i)
        if (q * 3 + i < NK)
          load_fragb(bh[i], bl[i], c.img,
                     kFwdFrags + bwd_base(L) + kb * NK + q * 3 + i - c.frag0,
                     c.lane);
      if (q * 3 + 3 <= NK) mma3<3>(acc + q * 3, ahi, alo, bh, bl);
      else mma3<NK % 3>(acc + q * 3, ahi, alo, bh, bl);
    }
    retire<NK>(kb, 6, acc, accf);
  }
#pragma unroll
  for (int nb = 0; nb < NK; ++nb) {
    float4 hp4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (L == 2) hp4 = epi4(c, 3 * kH, nb);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (L == 0) g[nb][i] = accf[nb][i] * inv;
      else if constexpr (L == 2) g[nb][i] = fmaf(accf[nb][i], inv, f4get(hp4, i & 3));
      else g[nb][i] = fmaf(accf[nb][i], inv, g[nb][i]);
    }
  }
}


#ifndef K3N_DEVICE_ONLY
__global__ __launch_bounds__(kMaxWarps * 32, 1)
void fitting_col2_seq_kernel(long n, float* x, const long* __restrict__ types, const double* __restrict__ seed,
                             const double* __restrict__ atom_bias, const float* __restrict__ head_bias,
                             const uint4* __restrict__ img, float* __restrict__ scratch,
                             double* __restrict__ energy) {
  extern __shared__ __align__(16) unsigned char smraw[];
  {
    uint4* d = reinterpret_cast<uint4*>(smraw);
    for (int i = threadIdx.x; i < kImageBytes / 16; i += blockDim.x) d[i] = __ldg(img + i);
    __syncthreads();
  }
  const float* HD = reinterpret_cast<const float*>(smraw + kFragArea) + 3 * kH;
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  Ctx c{smraw, HD, HD + 100, scratch + (long(blockIdx.x) * warps + warp) * kScratchPerWarp,
        0, lane, lane >> 2, lane & 3};
  const long stride = long(gridDim.x) * warps;
  const float hb = head_bias ? *head_bias : 0.f;
  for (long task = long(blockIdx.x) * warps + warp; task * 16 < n; task += stride) {
    const long r0 = task * 16 + c.gr, r1 = r0 + 8;
    float2 xq[5][4];
    float h[6][8];
    float d2[6][8];
    float p[2] = {0.f, 0.f};
    load_x(xq, x, r0, r1, n, c.tq);
    forward<0>(c, h, d2, p, xq);
    forward<1>(c, h, d2, p, xq);
    forward<2, 0>(c, h, d2, p, xq);
    forward<2, 1>(c, h, d2, p, xq);
    float p0 = p[0], p1 = p[1];
    p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
    if (c.tq == 0) {
      if (r0 < n) energy[r0] = double(p0 + hb) + atom_bias[types[r0]];
      if (r1 < n) energy[r1] = double(p1 + hb) + atom_bias[types[r1]];
    }
    backward<2>(c, h, d2);
    backward<1>(c, h, d2);
    backward<0>(c, h, d2);
    const float s0 = r0 < n ? float(seed[r0]) : 0.f;
    const float s1 = r1 < n ? float(seed[r1]) : 0.f;
#pragma unroll
    for (int nb = 0; nb < 5; ++nb)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = nb * 16 + c.tq + 4 * j;
        if (k < kIn) {
          if (r0 < n) x[r0 * kIn + k] = h[nb][j] * s0;
          if (r1 < n) x[r1 * kIn + k] = h[nb][4 + j] * s1;
        }
      }
  }
}
#include "small_weight_resident.cuh"

#endif


static __device__ __noinline__ void fit_warp_col2(const unsigned char* img, float* x, const long* types, const double* seed,
                              const double* atom_bias, const float* head_bias, float* scratch, double* energy,
                              long n, long row0) {
  const float* HD = reinterpret_cast<const float*>(img + kFragArea) + 3 * kH;
  const int lane = threadIdx.x & 31;
  Ctx c{img, HD, HD + 100, scratch, 0, lane, lane >> 2, lane & 3};
  const float hb = head_bias ? *head_bias : 0.f;
  const long r0 = row0 + c.gr, r1 = r0 + 8;
  float2 xq[5][4];
  float h[6][8];
  float d2[6][8];
  float p[2] = {0.f, 0.f};
  load_x(xq, x, r0, r1, n, c.tq);
  forward<0>(c, h, d2, p, xq);
  forward<1>(c, h, d2, p, xq);
  forward<2, 0>(c, h, d2, p, xq);
  forward<2, 1>(c, h, d2, p, xq);
  float p0 = p[0], p1 = p[1];
  p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
  p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
  p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
  p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
  if (c.tq == 0) {
    if (r0 < n) energy[r0] = double(p0 + hb) + atom_bias[types[r0]];
    if (r1 < n) energy[r1] = double(p1 + hb) + atom_bias[types[r1]];
  }
  backward<2>(c, h, d2);
  backward<1>(c, h, d2);
  backward<0>(c, h, d2);
  const float s0 = r0 < n ? float(seed[r0]) : 0.f;
  const float s1 = r1 < n ? float(seed[r1]) : 0.f;
#pragma unroll
  for (int nb = 0; nb < 5; ++nb)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int k = nb * 16 + c.tq + 4 * j;
      if (k < kIn) {
        if (r0 < n) x[r0 * kIn + k] = h[nb][j] * s0;
        if (r1 < n) x[r1 * kIn + k] = h[nb][4 + j] * s1;
      }
    }
}

#ifndef K3N_DEVICE_ONLY
inline bool use_col2_env() {
  static const bool v = [] {
    const char* e = std::getenv("DPA_K3N_COL2");
    return e && e[0] == '1';
  }();
  return v;
}

// The cotangent replaces x in place: a warp reads its 16 rows before its
// forward and writes them after its backward, and no other warp touches them.
__global__ __launch_bounds__(kMaxWarps * 32, 1)
void fitting_kernel(long n, float* x, const long* __restrict__ types, const double* __restrict__ seed,
                    const double* __restrict__ atom_bias, const float* __restrict__ head_bias,
                    const uint4* __restrict__ img,
                    float* __restrict__ scratch, double* __restrict__ energy) {
  extern __shared__ __align__(16) unsigned char smraw[];
  {
    uint4* d = reinterpret_cast<uint4*>(smraw);
    for (int i = threadIdx.x; i < kImageBytes / 16; i += blockDim.x) d[i] = __ldg(img + i);
    __syncthreads();
  }
  const float* HD = reinterpret_cast<const float*>(smraw + kFragArea) + 3 * kH;
  const int warps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  Ctx c{smraw, HD, HD + 100, scratch + (long(blockIdx.x) * warps + warp) * kScratchPerWarp,
        0, lane, lane >> 2, lane & 3};
  const long stride = long(gridDim.x) * warps;
  const float hb = head_bias ? *head_bias : 0.f;
  for (long task = long(blockIdx.x) * warps + warp; task * 16 < n; task += stride) {
    const long r0 = task * 16 + c.gr, r1 = r0 + 8;
    float2 xq[5][4];
    float h[6][8];
    float d2[6][8];
    float p[2] = {0.f, 0.f};
    load_x(xq, x, r0, r1, n, c.tq);
    forward<0>(c, h, d2, p, xq);
    forward<1>(c, h, d2, p, xq);
    forward<2>(c, h, d2, p, xq);
    float p0 = p[0], p1 = p[1];
    p0 += __shfl_xor_sync(0xffffffffu, p0, 1);
    p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 1);
    p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
    if (c.tq == 0) {
      if (r0 < n) energy[r0] = double(p0 + hb) + atom_bias[types[r0]];
      if (r1 < n) energy[r1] = double(p1 + hb) + atom_bias[types[r1]];
    }
    backward<2>(c, h, d2);
    backward<1>(c, h, d2);
    backward<0>(c, h, d2);
    const float s0 = r0 < n ? float(seed[r0]) : 0.f;
    const float s1 = r1 < n ? float(seed[r1]) : 0.f;
#pragma unroll
    for (int nb = 0; nb < 5; ++nb)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = nb * 16 + c.tq + 4 * j;
        if (k < kIn) {
          if (r0 < n) x[r0 * kIn + k] = h[nb][j] * s0;
          if (r1 < n) x[r1 * kIn + k] = h[nb][4 + j] * s1;
        }
      }
  }
}

// The image is a layout copy of the potential's weights, rebuilt whenever a
// source storage or its mutation version changes. The exported model hands
// in fresh tensor objects over the same storage every step, so the key is the
// data pointer; owning the sources keeps that storage from being reused under
// it. The head bias is converted afresh on every call, so the kernel reads it
// directly instead of from the image. Captured CUDA graphs hold the image and scratch pointers, so replaced
// buffers are retired rather than freed. Launches sharing the scratch must not
// overlap in time.
struct Cache {
  std::array<torch::Tensor, 7> sources;
  std::array<uint32_t, 7> versions{};
  torch::Tensor image;
  torch::Tensor scratch;
  std::vector<torch::Tensor> retired;
  int sms = 0;
};

inline uint32_t version_of(const torch::Tensor& t) {
  if (!t.defined()) return 0;
  const auto& counter = t.unsafeGetTensorImpl()->version_counter();
  return counter.enabled() ? counter.current_version() : 0;
}

inline bool usable(const torch::Tensor& t, long numel) {
  return t.defined() && t.is_cuda() && t.scalar_type() == torch::kFloat32 &&
         t.is_contiguous() && t.numel() == numel;
}

inline const char* unsupported(long input_width, const std::vector<torch::Tensor>& ws,
                               const std::vector<torch::Tensor>& bs, const std::vector<int64_t>& resnets,
                               const torch::Tensor& w_head, const torch::Tensor& b_head, int64_t act) {
  if (act != 1) return "activation";
  if (input_width != kIn || ws.size() != 3 || bs.size() != 3 || resnets.size() != 3) return "depth";
  // A residual needs equal widths, so the 70-wide input layer never has one.
  if (resnets[1] == 0 || resnets[2] == 0) return "residual";
  if (!usable(ws[0], kIn * kH) || ws[0].size(0) != kIn || !usable(ws[1], kH * kH) ||
      !usable(ws[2], kH * kH) || !usable(w_head, kH))
    return "weights";
  for (const auto& b : bs)
    if (b.numel() != 0 && !usable(b, kH)) return "bias";
  if (b_head.numel() != 0 && !usable(b_head, 1)) return "head bias";
  return nullptr;
}

// Returns false (and leaves every output untouched) outside the supported
// network: 70-96-96-96 silu, identity residuals on layers 1 and 2, 96-wide head.
bool energy_gradient(cudaStream_t stream, float* x, long input_width, const long* atype,
                     const std::vector<torch::Tensor>& ws, const std::vector<torch::Tensor>& bs,
                     const std::vector<int64_t>& resnets, const torch::Tensor& w_head,
                     const torch::Tensor& b_head, const torch::Tensor& bias_atom_e, int64_t act,
                     const double* seed, long n, double* e) {
#ifndef FITTING_IMAGE_FORCE_K3N4
  const char* setting = std::getenv("DPA_GENERIC_K3N_FITTING");
  if (!setting || !setting[0] || setting[0] == '0' || n <= 0) return false;
#else
  if (n <= 0) return false;
#endif
  if (const char* reason = unsupported(input_width, ws, bs, resnets, w_head, b_head, act)) {
    static bool announced = false;
    if (!announced) {
      std::fprintf(stderr, "K3N_FITTING unsupported %s\n", reason);
      announced = true;
    }
    return false;
  }

  thread_local Cache cache;
  const std::array<const torch::Tensor*, 7> src = {&ws[0], &ws[1], &ws[2], &bs[0], &bs[1], &bs[2], &w_head};
  bool hit = cache.image.defined() && cache.image.device() == ws[0].device();
  int changed = -1;
  for (int i = 0; hit && i < 7; ++i) {
    const torch::Tensor& t = *src[i];
    hit = cache.sources[i].defined() == t.defined() &&
          (!t.defined() || (cache.sources[i].data_ptr() == t.data_ptr() &&
                            cache.sources[i].numel() == t.numel() &&
                            cache.versions[i] == version_of(t)));
    if (!hit) changed = i;
  }
  if (!hit) {
    static bool announced = false;
    if (cache.image.defined() && !announced) {
      std::fprintf(stderr, "K3N_FITTING image rebuilt source=%d\n", changed);
      announced = true;
    }
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, ws[0].get_device()));
    int device = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaFuncSetAttribute(fitting_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kImageBytes));
    auto opts = ws[0].options();
    if (cache.image.defined()) cache.retired.push_back(cache.image);
    cache.image = torch::empty({kImageBytes / 4}, opts);
    if (!cache.scratch.defined() || cache.scratch.device() != ws[0].device() || cache.sms != sms) {
      if (cache.scratch.defined()) cache.retired.push_back(cache.scratch);
      cache.scratch = torch::empty({long(sms) * kMaxWarps * kScratchPerWarp}, opts);
    }
    cache.sms = sms;
    Params p{{ws[0].data_ptr<float>(), ws[1].data_ptr<float>(), ws[2].data_ptr<float>()},
             {bs[0].numel() ? bs[0].data_ptr<float>() : nullptr,
              bs[1].numel() ? bs[1].data_ptr<float>() : nullptr,
              bs[2].numel() ? bs[2].data_ptr<float>() : nullptr},
             w_head.data_ptr<float>()};
    auto* img = reinterpret_cast<unsigned char*>(cache.image.data_ptr<float>());
    weight_scale_kernel<<<3, 256, 0, stream>>>(p, img);
    build_image_kernel<<<64, 256, 0, stream>>>(p, img);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int i = 0; i < 7; ++i) {
      cache.sources[i] = *src[i];
      cache.versions[i] = version_of(*src[i]);
    }
  }
  // Every block first copies the 212 KB image into shared memory, which is
  // latency-bound below ~8 warps per block, so small systems use fewer SMs
  // rather than shallower blocks.
  const long tasks = (n + 15) / 16;
  int warps = int(std::min<long>(kMaxWarps, std::max<long>(kMinWarps, (tasks + cache.sms - 1) / cache.sms)));
  if (const char* w = std::getenv("DPA_K3N_WARPS")) {
    const int forced = std::atoi(w);
    if (forced >= 1 && forced <= kMaxWarps) warps = forced;
  }
  int blocks = int(std::min<long>(cache.sms, (tasks + warps - 1) / warps));
  if (const char* cap = std::getenv("DPA_K3N_BLOCKS")) {
    const int forced = std::atoi(cap);
    if (forced >= 1 && forced < blocks) blocks = forced;
  }
  const uint4* imgp = reinterpret_cast<const uint4*>(cache.image.data_ptr<float>());
  const char* resident_fitting_env = std::getenv("DPA_RESIDENT_FITTING_SMALL_WEIGHT_RESIDENT");
  if (n <= 4096 && use_col2_env() && resident_fitting_env && resident_fitting_env[0] == '1') {
    int tile_warps = 1;
    if (const char* setting = std::getenv("DPA_RESIDENT_FITTING_TILE_WARPS")) {
      const int chosen = std::atoi(setting);
      if (chosen == 1 || chosen == 2 || chosen == 4) tile_warps = chosen;
    }
    const int tile_blocks = int((tasks + tile_warps - 1) / tile_warps);
    if (long(tile_blocks) * tile_warps <= long(cache.sms) * kMaxWarps) {
      dpa_l2_pin(stream, cache.image.data_ptr<float>(), size_t(kImageBytes), "k3n-img");
      if (deepmd_dpa4c::chain_fit_node::queue_fit({n,x,atype,seed,bias_atom_e.data_ptr<double>(),
          b_head.numel()?b_head.data_ptr<float>():nullptr,imgp,cache.scratch.data_ptr<float>(),e},stream)) return true;
      chain_resident_fitting::fitting_weight_resident_kernel<<<tile_blocks, tile_warps * 32, 0, stream>>>(
          n, x, atype, seed, bias_atom_e.data_ptr<double>(),
          b_head.numel() ? b_head.data_ptr<float>() : nullptr,
          imgp, cache.scratch.data_ptr<float>(), e);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      static bool shown = false;
      if (!shown) {
        cudaFuncAttributes attr{}; int resident = 0;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, chain_resident_fitting::fitting_weight_resident_kernel));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,
            chain_resident_fitting::fitting_weight_resident_kernel, tile_warps*32, 0));
        std::fprintf(stderr, "RESIDENT_FITTING_WEIGHT_RESIDENT nodes=%ld warps=%d blocks=%d regs=%d local=%zu shared=%zu resident=%d\n",
                     n,tile_warps,tile_blocks,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,resident);
        shown = true;
      }
      return true;
    }
  }
  if (use_col2_env()) {
    dpa_l2_pin(stream, cache.image.data_ptr<float>(), size_t(kImageBytes), "k3n-img");
    fitting_col2_seq_kernel<<<blocks, warps * 32, kImageBytes, stream>>>(
        n, x, atype, seed, bias_atom_e.data_ptr<double>(),
        b_head.numel() ? b_head.data_ptr<float>() : nullptr, imgp, cache.scratch.data_ptr<float>(), e);
  } else {
    dpa_l2_pin(stream, cache.image.data_ptr<float>(), size_t(kImageBytes), "k3n-img");
  fitting_kernel<<<blocks, warps * 32, kImageBytes, stream>>>(
        n, x, atype, seed, bias_atom_e.data_ptr<double>(),
        b_head.numel() ? b_head.data_ptr<float>() : nullptr, imgp, cache.scratch.data_ptr<float>(), e);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  static bool announced = false;
  if (!announced) {
    std::fprintf(stderr, "K3N_FITTING active nodes=%ld warps=%d blocks=%d\n", n, warps, blocks);
    announced = true;
  }
  return true;
}


struct FitImage {
  const unsigned char* img = nullptr;
  float* scratch = nullptr;
};

FitImage prepare_image(cudaStream_t stream, const std::vector<torch::Tensor>& ws,
                              const std::vector<torch::Tensor>& bs, const std::vector<int64_t>& resnets,
                              const torch::Tensor& w_head, const torch::Tensor& b_head,
                              const torch::Tensor& bias_atom_e, int64_t act) {
  FitImage out;
#ifndef FITTING_IMAGE_FORCE_K3N4
  const char* setting = std::getenv("DPA_GENERIC_K3N_FITTING");
  if (!setting || !setting[0] || setting[0] == '0') return out;
#endif
  if (unsupported(70, ws, bs, resnets, w_head, b_head, act)) return out;
  thread_local Cache cache;
  const std::array<const torch::Tensor*, 7> src = {&ws[0], &ws[1], &ws[2], &bs[0], &bs[1], &bs[2], &w_head};
  bool hit = cache.image.defined() && cache.image.device() == ws[0].device();
  for (int i = 0; hit && i < 7; ++i) {
    const torch::Tensor& t = *src[i];
    hit = cache.sources[i].defined() == t.defined() &&
          (!t.defined() || (cache.sources[i].data_ptr() == t.data_ptr() &&
                            cache.sources[i].numel() == t.numel() &&
                            cache.versions[i] == version_of(t)));
  }
  if (!hit) {
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, ws[0].get_device()));
    int device = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    auto opts = ws[0].options();
    if (cache.image.defined()) cache.retired.push_back(cache.image);
    cache.image = torch::empty({kImageBytes / 4}, opts);
    if (!cache.scratch.defined() || cache.scratch.device() != ws[0].device() || cache.sms != sms) {
      if (cache.scratch.defined()) cache.retired.push_back(cache.scratch);
      cache.scratch = torch::empty({long(sms) * kMaxWarps * kScratchPerWarp}, opts);
    }
    cache.sms = sms;
    Params p{{ws[0].data_ptr<float>(), ws[1].data_ptr<float>(), ws[2].data_ptr<float>()},
             {bs[0].numel() ? bs[0].data_ptr<float>() : nullptr,
              bs[1].numel() ? bs[1].data_ptr<float>() : nullptr,
              bs[2].numel() ? bs[2].data_ptr<float>() : nullptr},
             w_head.data_ptr<float>()};
    auto* built = reinterpret_cast<unsigned char*>(cache.image.data_ptr<float>());
    weight_scale_kernel<<<3, 256, 0, stream>>>(p, built);
    build_image_kernel<<<64, 256, 0, stream>>>(p, built);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int i = 0; i < 7; ++i) {
      cache.sources[i] = *src[i];
      cache.versions[i] = version_of(*src[i]);
    }
  }
  out.img = reinterpret_cast<const unsigned char*>(cache.image.data_ptr<float>());
  out.scratch = cache.scratch.data_ptr<float>();
  (void)bias_atom_e;
  return out;
}
#endif

}  // namespace k3n
