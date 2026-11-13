// The SSE4 kernels built again with AVX, for x86-64 CPUs that have AVX
// but not AVX2 (Sandy Bridge, Ivy Bridge). The registry picks these
// when sys::detect() reports AVX. AVX re-encodes the 128-bit code (VEX:
// three operands, fewer register copies), and the batched Q8_0 row
// kernel keeps its two 4-lane float sums in one 256-bit register, the
// same adds in the same lanes. So the results equal the SSE4 build's
// bit for bit.
//
// The file keeps the SSE4 flags: only the functions of matvec_sse4.cpp
// get the AVX target (a target pragma opened after every header is
// in), so the inline copies of header and standard-library functions
// this TU emits carry no AVX instructions into code that runs on a CPU
// without it (the linker keeps one copy of each).

#include "locus/backend/variants.hpp"

#if defined(__x86_64__) && defined(__SSE4_1__)

#include <immintrin.h>
#include <smmintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx"))), \
                             apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx")
#endif

#define LOCUS_SSE4_AVX_VARIANT 1
#include "matvec_sse4.cpp"

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif  // __x86_64__ && __SSE4_1__
