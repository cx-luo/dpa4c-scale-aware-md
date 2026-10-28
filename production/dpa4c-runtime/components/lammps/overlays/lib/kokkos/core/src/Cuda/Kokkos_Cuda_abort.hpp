//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_CUDA_ABORT_HPP
#define KOKKOS_CUDA_ABORT_HPP

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <cuda.h>

namespace Kokkos {
namespace Impl {

[[noreturn]] __device__ static void cuda_abort(const char *const message) {
  // PPU SDK has no CUDA __assertfail implementation. Preserve fail-closed
  // device behavior with the supported trap instruction.
  printf("Kokkos device abort: %s\n", message);
  asm volatile("trap;");
  __builtin_unreachable();
}

}  // namespace Impl
}  // namespace Kokkos
#else
void KOKKOS_CORE_SRC_CUDA_ABORT_PREVENT_LINK_ERROR() {}
#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #ifndef KOKKOS_CUDA_ABORT_HPP */
