// Q8_0 weights x Q8_0 activations through sdot by lane on four
// interleaved weight rows, the default Q8_0 batch kernel on a CPU with
// dotprod (i#1076; matvec_neon_i8mm.cpp takes Q8_0 instead under
// LOCUS_CPU_FAST on an i8mm CPU). Four rows' blocks are transposed so
// that each 32-bit lane of a vector holds 4 bytes of one row: one
// vdotq_laneq_s32 then dots 4 rows against 4 activation bytes, and one
// convert, one multiply and one add per block serve the 4 rows, where
// the one-row kernels spend them per row.
//
// Q4_0 takes the same shape from a load-time copy only (repack.hpp:
// block_q4_0x4, its nibbles xor 0x88): b << 4 and b & 0xF0 are 16 times
// the signed quants, so the sdot lanes give 16 times the exact integer
// dot and vcvtq_n_f32_s32(., 4) converts it exactly (i#1243). Without a
// copy Q4_0 goes to the dotprod build's kernels as before.
//
// The float order is every CPU kernel's block-32 order (scalar
// vec_dot_block32_q8_0, llama.cpp's repacked one): per block an exact
// integer dot, times d_w * d_x, then added to the sum, a multiply then
// an add. So a token gets the same bits alone (the one-row kernels) or
// in a batch (matvec_batch_neon_q8k here), on a dotprod CPU as on an
// i8mm one.
//
// Like matvec_neon_dotprod.cpp, the file keeps the baseline flags and
// gives only its own functions the dotprod target (a pragma opened
// after the headers).

#include "locus/backend/variants.hpp"

#if defined(__aarch64__) && defined(LOCUS_HAS_NEON_DOTPROD_FAST_VARIANT)

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/repack.hpp"

#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

namespace locus::backend::neon_dotprod_fast {

namespace {

constexpr std::size_t kBlk = 34;  // f16 scale + 32 int8

/** Block b's f16 scales of 4 rows, through one hardware convert
 *  (exact). */
inline float32x4_t scales4(const std::byte* const rows[4], std::size_t b) {
    std::uint16_t h[4];
    for (int k = 0; k < 4; ++k) {
        std::memcpy(&h[k], rows[k] + b * kBlk, sizeof(h[k]));
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h)));
}

/**
 * The 4 x 4 transpose of 32-bit words: out[k] holds word k of a, b, c
 * and d, in that order.
 */
inline void transpose4(int8x16_t a, int8x16_t b, int8x16_t c, int8x16_t d,
                       int8x16_t out[4]) {
    const int32x4_t t0 =
        vtrn1q_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b));
    const int32x4_t t1 =
        vtrn2q_s32(vreinterpretq_s32_s8(a), vreinterpretq_s32_s8(b));
    const int32x4_t t2 =
        vtrn1q_s32(vreinterpretq_s32_s8(c), vreinterpretq_s32_s8(d));
    const int32x4_t t3 =
        vtrn2q_s32(vreinterpretq_s32_s8(c), vreinterpretq_s32_s8(d));
    const auto lo = [](int32x4_t x, int32x4_t y) {
        return vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(x),
                                               vreinterpretq_s64_s32(y)));
    };
    const auto hi = [](int32x4_t x, int32x4_t y) {
        return vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(x),
                                               vreinterpretq_s64_s32(y)));
    };
    out[0] = lo(t0, t2);
    out[1] = lo(t1, t3);
    out[2] = hi(t0, t2);
    out[3] = hi(t1, t3);
}

/** The weight blocks of rows r0 .. r0 + 3 (a row past the last repeats
 *  the last). */
void quad_rows(const Mat& w, std::uint32_t r0, std::size_t nb,
               const std::byte* rows[4]) {
    for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t r = std::min(r0 + k, w.rows - 1);
        rows[k] = w.data + std::size_t{r} * nb * kBlk;
    }
}

/** Block b's 32 quants of row k as two vectors. */
inline void row_quants(const std::byte* const rows[4], int k, std::size_t b,
                       int8x16_t& lo, int8x16_t& hi) {
    const auto* q =
        reinterpret_cast<const std::int8_t*>(rows[k] + b * kBlk + 2);
    lo = vld1q_s8(q);
    hi = vld1q_s8(q + 16);
}

/**
 * Rows r0 .. r0 + 3 of w packed for quad_tokens, in the load-time
 * layout (repack.hpp, i#1243): block b at out + b * 136, its 4 f16
 * scales, then 8 vectors (vector k holding bytes 4k .. 4k + 3 of each
 * row). The run-time pack for a matrix with no load-time copy.
 */
void pack_quad(const Mat& w, std::uint32_t r0, std::size_t nb,
               std::byte* out) {
    const std::byte* rows[4];
    quad_rows(w, r0, nb, rows);
    for (std::size_t b = 0; b < nb; ++b) {
        std::byte* o = out + b * kPackedQ8_0Block;
        int8x16_t lo[4], hi[4], wv[8];
        for (int k = 0; k < 4; ++k) {
            row_quants(rows, k, b, lo[k], hi[k]);
            std::memcpy(o + 2 * k, rows[k] + b * kBlk, 2);
        }
        transpose4(lo[0], lo[1], lo[2], lo[3], wv);
        transpose4(hi[0], hi[1], hi[2], hi[3], wv + 4);
        for (int k = 0; k < 8; ++k) {
            vst1q_s8(reinterpret_cast<std::int8_t*>(o + 8 + k * 16), wv[k]);
        }
    }
}

/**
 * One block's step of a packed quad for one token: the 4 rows' exact
 * integer dots (wv against activation bytes 4k ..
 * 4k + 3: lane k of x0 for k < 4, lane k - 4 of x1 after), times
 * d_w * d_x per row, then added to sum: a multiply, then an add (no
 * fused multiply-add), per lane.
 */
inline float32x4_t quad_step(float32x4_t sum, const int8x16_t wv[8],
                             float32x4_t ws, int8x16_t x0, int8x16_t x1,
                             float dx) {
    int32x4_t s = vdupq_n_s32(0);
    s = vdotq_laneq_s32(s, wv[0], x0, 0);
    s = vdotq_laneq_s32(s, wv[1], x0, 1);
    s = vdotq_laneq_s32(s, wv[2], x0, 2);
    s = vdotq_laneq_s32(s, wv[3], x0, 3);
    s = vdotq_laneq_s32(s, wv[4], x1, 0);
    s = vdotq_laneq_s32(s, wv[5], x1, 1);
    s = vdotq_laneq_s32(s, wv[6], x1, 2);
    s = vdotq_laneq_s32(s, wv[7], x1, 3);
    const float32x4_t sc = vmulq_n_f32(ws, dx);
    return vaddq_f32(sum, vmulq_f32(vcvtq_f32_s32(s), sc));
}

/**
 * Rows r0 .. r0 + 3 against one token, from the weight rows as stored
 * (no transpose): the decode path, where a packed quad would serve one
 * token only. Each row's block dot is two plain sdot, and three
 * pairwise adds fold the 4 rows' lanes into one vector of the 4 exact
 * integer dots; the float steps are quad_step's, so the same bits.
 */
float32x4_t quad_direct(const Mat& w, std::uint32_t r0, std::size_t nb,
                        const float* d, const std::int8_t* q) {
    const std::byte* rows[4];
    quad_rows(w, r0, nb, rows);
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (std::size_t b = 0; b < nb; ++b) {
        const int8x16_t x0 = vld1q_s8(q + b * 32);
        const int8x16_t x1 = vld1q_s8(q + b * 32 + 16);
        int32x4_t s[4];
        for (int k = 0; k < 4; ++k) {
            int8x16_t lo, hi;
            row_quants(rows, k, b, lo, hi);
            s[k] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, x0), hi, x1);
        }
        const int32x4_t isum = vpaddq_s32(vpaddq_s32(s[0], s[1]),
                                          vpaddq_s32(s[2], s[3]));
        const float32x4_t sc = vmulq_n_f32(scales4(rows, b), d[b]);
        sum = vaddq_f32(sum, vmulq_f32(vcvtq_f32_s32(isum), sc));
    }
    return sum;
}

/**
 * quad_step for a packed Q4_0 quad block: wv[k] (k 0 .. 3) holds bytes
 * 4k .. 4k + 3 of each row, xor 0x88, so wv[k] << 4 is 16 times the
 * signed quants 4k .. 4k + 3 (dotted with lane k of x0) and wv[k] &
 * 0xF0 16 times quants 16 + 4k .. (lane k of x1). The integer sum is 16
 * times the exact dot, so vcvtq_n_f32_s32(., 4) gives the dot exactly;
 * then d_w * d_x, a multiply and an add, as quad_step.
 */
inline float32x4_t quad4_step(float32x4_t sum, const int8x16_t lo[4],
                              const int8x16_t hi[4], float32x4_t ws,
                              int8x16_t x0, int8x16_t x1, float dx) {
    int32x4_t s = vdupq_n_s32(0);
    s = vdotq_laneq_s32(s, lo[0], x0, 0);
    s = vdotq_laneq_s32(s, lo[1], x0, 1);
    s = vdotq_laneq_s32(s, lo[2], x0, 2);
    s = vdotq_laneq_s32(s, lo[3], x0, 3);
    s = vdotq_laneq_s32(s, hi[0], x1, 0);
    s = vdotq_laneq_s32(s, hi[1], x1, 1);
    s = vdotq_laneq_s32(s, hi[2], x1, 2);
    s = vdotq_laneq_s32(s, hi[3], x1, 3);
    const float32x4_t sc = vmulq_n_f32(ws, dx);
    return vaddq_f32(sum, vmulq_f32(vcvtq_n_f32_s32(s, 4), sc));
}

/** A packed quad block's four f16 scales, converted exactly. */
inline float32x4_t packed_scales(const std::byte* o) {
    std::uint16_t h[4];
    std::memcpy(h, o, sizeof(h));
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h)));
}

/**
 * A packed quad (nb blocks at pq: Q8_0's 136 bytes, or Q4_0's 72 when
 * Q4) against T tokens (token t's scales at d + t * nb, its quants at q
 * + t * cols): lane r of sum[t] is row r's dot with token t. Each (row,
 * token) takes the same steps whatever T is, and whether the quad was
 * packed at load or here.
 */
template <int T, bool Q4 = false>
void quad_tokens(const std::byte* pq, std::size_t nb, std::size_t cols,
                 const float* d, const std::int8_t* q, float32x4_t sum[T]) {
    for (int t = 0; t < T; ++t) {
        sum[t] = vdupq_n_f32(0.0f);
    }
    for (std::size_t b = 0; b < nb; ++b) {
        if constexpr (Q4) {
            const std::byte* o = pq + b * kPackedQ4_0Block;
            const int8x16_t high = vdupq_n_s8(static_cast<std::int8_t>(0xF0));
            int8x16_t lo[4], hi[4];
            for (int k = 0; k < 4; ++k) {
                const int8x16_t v = vld1q_s8(
                    reinterpret_cast<const std::int8_t*>(o + 8 + k * 16));
                lo[k] = vshlq_n_s8(v, 4);
                hi[k] = vandq_s8(v, high);
            }
            const float32x4_t ws = packed_scales(o);
            for (int t = 0; t < T; ++t) {
                const std::int8_t* x = q + t * cols + b * 32;
                sum[t] = quad4_step(sum[t], lo, hi, ws, vld1q_s8(x),
                                    vld1q_s8(x + 16), d[t * nb + b]);
            }
        } else {
            const std::byte* o = pq + b * kPackedQ8_0Block;
            int8x16_t wv[8];
            for (int k = 0; k < 8; ++k) {
                wv[k] = vld1q_s8(
                    reinterpret_cast<const std::int8_t*>(o + 8 + k * 16));
            }
            // f16 to f32 is exact: the same scales as scales4.
            const float32x4_t ws = packed_scales(o);
            for (int t = 0; t < T; ++t) {
                const std::int8_t* x = q + t * cols + b * 32;
                sum[t] = quad_step(sum[t], wv, ws, vld1q_s8(x),
                                   vld1q_s8(x + 16), d[t * nb + b]);
            }
        }
    }
}

/**
 * Tokens per tile: as many quantized activation rows (cols bytes each)
 * as fit about 64 KiB, as the baseline kernels' tile, so each quad is
 * packed once per 64 tokens at most widths.
 */
std::uint32_t token_tile(std::size_t cols) {
    return static_cast<std::uint32_t>(
        std::clamp<std::size_t>((std::size_t{64} << 10) / cols, 4, 64));
}

/**
 * out[r * ostride + t] for every row of w and n tokens (scales d at
 * t * nb, quants q at t * cols). With a load-time copy (pk, i#1243)
 * each quad is read as packed; else, per token tile, each quad of rows
 * is packed once here (one token: dotted as stored, quad_direct). Each
 * quad is dotted against the tile's tokens 4 at a time, then one by
 * one; every path takes the same steps per (row, token). Q4 (Q4_0)
 * needs the copy: pk is never null there.
 */
template <bool Q4 = false>
void block32_rows(const Mat& w, const std::byte* pk, const float* d,
                  const std::int8_t* q, std::uint32_t n, float* out,
                  std::size_t ostride) {
    const std::size_t nb = w.cols / 32;
    constexpr std::size_t kBlock = Q4 ? kPackedQ4_0Block : kPackedQ8_0Block;
    thread_local std::vector<std::byte> qp;
    if (pk == nullptr) {
        qp.resize(nb * kPackedQ8_0Block);
    }
    const auto quad = [&](std::uint32_t r0) -> const std::byte* {
        if (Q4 || pk != nullptr) {
            return pk + std::size_t{r0 / 4} * nb * kBlock;
        }
        pack_quad(w, r0, nb, qp.data());
        return qp.data();
    };
    const auto store = [&](std::uint32_t r0, std::uint32_t t,
                           float32x4_t v4) {
        float v[4];
        vst1q_f32(v, v4);
        for (std::uint32_t k = 0; k < 4 && r0 + k < w.rows; ++k) {
            out[std::size_t{r0 + k} * ostride + t] = v[k];
        }
    };
    // One token reads the copy too when there is one: reading the rows
    // as stored while prefill reads the copy keeps both hot, and on the
    // Pi 5 (4 GB) decode then ran 22% slower against 2% (i#1243).
    if (!Q4 && n == 1 && pk == nullptr) {
        for (std::uint32_t r0 = 0; r0 < w.rows; r0 += 4) {
            store(r0, 0, quad_direct(w, r0, nb, d, q));
        }
        return;
    }
    // The tile keeps its tokens' activations in cache across the rows,
    // with or without a load-time copy.
    const std::uint32_t tile = token_tile(w.cols);
    for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
        const std::uint32_t t1 = std::min(n, t0 + tile);
        for (std::uint32_t r0 = 0; r0 < w.rows; r0 += 4) {
            const std::byte* pq = quad(r0);
            std::uint32_t t = t0;
            for (; t + 4 <= t1; t += 4) {
                float32x4_t sum[4];
                quad_tokens<4, Q4>(pq, nb, w.cols, d + std::size_t{t} * nb,
                                   q + std::size_t{t} * w.cols, sum);
                for (int k = 0; k < 4; ++k) {
                    store(r0, t + k, sum[k]);
                }
            }
            for (; t < t1; ++t) {
                float32x4_t sum[1];
                quad_tokens<1, Q4>(pq, nb, w.cols, d + std::size_t{t} * nb,
                                   q + std::size_t{t} * w.cols, sum);
                store(r0, t, sum[0]);
            }
        }
    }
}

// Other types go to the sdot kernels, as in matvec_neon_i8mm.cpp.
#if defined(LOCUS_HAS_NEON_DOTPROD_VARIANT)
namespace base = ::locus::backend::neon_dotprod;
#else
namespace base = ::locus::backend;
#endif

bool takes(const Mat& w) {
    return w.type == gguf::TensorType::kQ8_0 && w.cols % 32 == 0 &&
           w.cols > 0;
}

/** w's load-time Q4_0 copy, or null (a Q4_0 w without one, any other
 *  type). */
const std::byte* q4_0_copy(const Mat& w) {
    return w.type == gguf::TensorType::kQ4_0 ? find_packed(w) : nullptr;
}

/** All n tokens of x_batch quantized (scales at t * nb, quants at t *
 *  cols), then w's rows through block32_rows. */
template <bool Q4>
void block32_batch(const Mat& w, const std::byte* pk,
                   std::span<const float> x_batch, std::span<float> out,
                   std::uint32_t n) {
    const std::size_t nb = w.cols / 32;
    thread_local std::vector<float> d;
    thread_local std::vector<std::int8_t> q;
    d.resize(nb * n);
    q.resize(std::size_t{w.cols} * n);
    for (std::uint32_t t = 0; t < n; ++t) {
        quantize_activation_q8_0_neon(
            x_batch.subspan(std::size_t{t} * w.cols, w.cols),
            d.data() + t * nb, q.data() + std::size_t{t} * w.cols);
    }
    block32_rows<Q4>(w, pk, d.data(), q.data(), n, out.data(), n);
}

}  // namespace

void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    // Without a load-time copy one token takes the one-row sdot kernel,
    // the faster there (same bits); with one, the copy (see
    // block32_rows), for Q8_0 and Q4_0 alike.
    if (const std::byte* pk = q4_0_copy(w)) {
        block32_batch<true>(w, pk, x, out, 1);
        return;
    }
    const std::byte* pk = takes(w) ? find_packed(w) : nullptr;
    if (pk == nullptr) {
        base::matvec_neon_q8k(w, x, out);
        return;
    }
    block32_batch<false>(w, pk, x, out, 1);
}

void matvec_batch_neon_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n) {
    if (const std::byte* pk = q4_0_copy(w)) {
        block32_batch<true>(w, pk, x_batch, out_batch, n);
        return;
    }
    if (!takes(w)) {
        base::matvec_batch_neon_q8k(w, x_batch, out_batch, n);
        return;
    }
    block32_batch<false>(w, find_packed(w), x_batch, out_batch, n);
}

bool q8k_batch_size_neon(const Mat& w, std::uint32_t n, QuantBatch& b) {
    if (!takes(w)) {
        return base::q8k_batch_size_neon(w, n, b);
    }
    b.n = n;
    b.d.resize(std::size_t{w.cols} / 32 * n);
    b.qs.resize(std::size_t{w.cols} * n);
    b.bsums.clear();
    b.q6off.clear();
    return true;
}

void q8k_batch_quantize_neon(const Mat& w, std::span<const float> x_batch,
                             std::uint32_t t0, std::uint32_t t1,
                             QuantBatch& b) {
    if (!takes(w)) {
        base::q8k_batch_quantize_neon(w, x_batch, t0, t1, b);
        return;
    }
    const std::size_t nb = w.cols / 32;
    for (std::uint32_t t = t0; t < t1; ++t) {
        quantize_activation_q8_0_neon(
            x_batch.subspan(std::size_t{t} * w.cols, w.cols),
            b.d.data() + t * nb, b.qs.data() + std::size_t{t} * w.cols);
    }
}

void matvec_q8k_batch_pre_neon(const Mat& w, const QuantBatch& b,
                               std::span<float> out_batch, std::uint32_t n) {
    if (!takes(w)) {
        base::matvec_q8k_batch_pre_neon(w, b, out_batch, n);
        return;
    }
    // The load-time copy where the model made one (i#1243), as the whole
    // batch call: a row slice starts on a multiple of 4 (slice_row), so
    // find_packed finds its quads.
    block32_rows<false>(w, find_packed(w), b.d.data(), b.qs.data(), n,
                        out_batch.data(), n);
}

}  // namespace locus::backend::neon_dotprod_fast

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif  // __aarch64__ && LOCUS_HAS_NEON_DOTPROD_FAST_VARIANT
