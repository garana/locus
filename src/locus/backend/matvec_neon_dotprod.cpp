// The NEON kernels built again with the 8-bit dot-product extension
// (-march=armv8-a+dotprod), for compilers whose default arm64 target
// lacks it (GCC on the Raspberry Pi 5). The registry picks these when
// sys::detect() reports dotprod. Their integer sums are exact either
// way, so the results equal the baseline build's bit for bit.
#define LOCUS_NEON_DOTPROD_VARIANT 1
#include "matvec_neon.cpp"
