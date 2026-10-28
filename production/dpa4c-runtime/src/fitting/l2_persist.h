#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

inline bool dpa_l2_on() {
  const char* e = std::getenv("DPA_L2_PERSIST");
  return e && e[0] == '1';
}

inline void dpa_l2_setup_device() {
  static int once = 0;
  if (once++) return;
  int l2 = 0;
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
  int persist_attr = 0;
#if defined(cudaDevAttrMaxL2PersistingCacheSize)
  cudaDeviceGetAttribute(&persist_attr, cudaDevAttrMaxL2PersistingCacheSize, 0);
#endif
  size_t prev = 0, want = persist_attr > 0 ? size_t(persist_attr) : (size_t(48) << 20);
  if (want > size_t(56) << 20) want = size_t(56) << 20;
  cudaError_t g = cudaErrorNotSupported, s = cudaErrorNotSupported;
#ifdef cudaLimitPersistingL2CacheSize
  g = cudaDeviceGetLimit(&prev, cudaLimitPersistingL2CacheSize);
  s = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want);
#endif
  std::fprintf(stderr,
               "DPA_L2_PERSIST device l2=%d persist_attr=%d prev=%zu want=%zu get=%s set=%s\n",
               l2, persist_attr, prev, want, cudaGetErrorString(g), cudaGetErrorString(s));
}

inline void dpa_l2_pin(cudaStream_t stream, const void* ptr, size_t bytes, const char* tag) {
  if (!dpa_l2_on() || !ptr || bytes == 0) return;
  dpa_l2_setup_device();
#if defined(cudaStreamAttributeAccessPolicyWindow) && defined(cudaAccessPropertyPersisting)
  cudaStreamAttrValue v{};
  v.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
  v.accessPolicyWindow.num_bytes = bytes;
  v.accessPolicyWindow.hitRatio = 1.0f;
  v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaError_t st = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &v);
  static int n = 0;
  if (n < 12) {
    std::fprintf(stderr, "DPA_L2_PERSIST pin %s ptr=%p bytes=%zu err=%s\n", tag, ptr, bytes,
                 cudaGetErrorString(st));
    ++n;
  }
#else
  static int once = 0;
  if (!once++)
    std::fprintf(stderr, "DPA_L2_PERSIST no AccessPolicyWindow in SDK (first %s %zu)\n", tag, bytes);
  (void)stream;
#endif
}
