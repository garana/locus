// The NEON kernels built again with the 8-bit dot product, for
// compilers whose default arm64 target lacks it (GCC on the Raspberry
// Pi 5). The registry picks these when sys::detect() reports dotprod.
// Their integer sums are exact either way, so the results equal the
// baseline build's bit for bit.
//
// The file keeps the baseline flags: only the functions of
// matvec_neon.cpp get the dot-product target (a target pragma opened
// after every header is in), so the inline copies of header and
// standard-library functions this TU emits stay baseline and cannot
// carry v8.1 / v8.2 instructions into a v8.0 build (the linker keeps
// one copy of each). armv8.2-a because GCC's arm_neon.h declares the
// dot-product intrinsics under that arch.

#include "locus/backend/variants.hpp"

#if defined(__aarch64__)

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "iq_grids.h"
#include "locus/backend/cpu_ops.hpp"

#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

#define LOCUS_NEON_DOTPROD_VARIANT 1
#include "matvec_neon.cpp"

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif  // __aarch64__
