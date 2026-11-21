// Q8_0 weights x Q8_0 activations through sdot by lane on four
// interleaved weight rows, picked only under LOCUS_CPU_FAST (cpu_fast)
// on a CPU with dotprod and no i8mm (where matvec_neon_i8mm.cpp takes
// Q8_0). Four rows' blocks are transposed so that each 32-bit lane of
// a vector holds 4 bytes of one row: one vdotq_laneq_s32 then dots 4
// rows against 4 activation bytes, and one convert, one multiply and
// one add per block serve the 4 rows, where the baseline kernels spend
// them per row.
//
// The float order is matvec_neon_i8mm.cpp's (llama.cpp's
// ggml_vec_dot_q8_0_q8_0): per block an exact integer dot, times
// d_w * d_x, then added to the sum, a multiply then an add. So under
// the flag a Q8_0 result is the same bits on a dotprod CPU as on an
// i8mm one, and a token gets the same bits alone (matvec_neon_q8k) or
// in a batch (matvec_batch_neon_q8k). It is not the baseline order,
// hence the opt-in.
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

/** A block's f16 scale, through the hardware convert (exact). */
float scale_of(const std::byte* blk) {
    std::uint16_t h;
    std::memcpy(&h, blk, sizeof(h));
    return vget_lane_f32(
        vget_low_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h)))), 0);
}

/**
 * The 4 x 4 transpose of 32-bit words: out[k] holds word k of a, b, c
 * and d, in that order.
 */
void transpose4(int8x16_t a, int8x16_t b, int8x16_t c, int8x16_t d,
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

/**
 * Block b of 4 rows interleaved: wv[k] holds bytes 4k .. 4k + 3 of
 * each row (k < 8), ws the 4 rows' scales.
 */
void quad_block(const std::byte* const rows[4], std::size_t b,
                int8x16_t wv[8], float ws[4]) {
    int8x16_t lo[4], hi[4];
    for (int k = 0; k < 4; ++k) {
        const std::byte* blk = rows[k] + b * kBlk;
        const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
        lo[k] = vld1q_s8(q);
        hi[k] = vld1q_s8(q + 16);
        ws[k] = scale_of(blk);
    }
    transpose4(lo[0], lo[1], lo[2], lo[3], wv);
    transpose4(hi[0], hi[1], hi[2], hi[3], wv + 4);
}

/**
 * Rows r0 .. r0 + 3 of w packed for quad_tokens: block b's 8 vectors
 * at qw + b * 128 and its 4 scales at qs + b * 4.
 */
void pack_quad(const Mat& w, std::uint32_t r0, std::size_t nb,
               std::int8_t* qw, float* qs) {
    const std::byte* rows[4];
    quad_rows(w, r0, nb, rows);
    for (std::size_t b = 0; b < nb; ++b) {
        int8x16_t wv[8];
        quad_block(rows, b, wv, qs + b * 4);
        for (int k = 0; k < 8; ++k) {
            vst1q_s8(qw + b * 128 + k * 16, wv[k]);
        }
    }
}

/**
 * One block's step for one token, shared by both kernels below: the
 * 4 rows' exact integer dots (wv against activation bytes 4k ..
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
 * Rows r0 .. r0 + 3 against one token, interleaving each block in
 * registers (no packed copy): the decode path, where a packed quad
 * would serve one token only. The same quad_step as quad_tokens, so
 * the same bits.
 */
float32x4_t quad_direct(const Mat& w, std::uint32_t r0, std::size_t nb,
                        const float* d, const std::int8_t* q) {
    const std::byte* rows[4];
    quad_rows(w, r0, nb, rows);
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (std::size_t b = 0; b < nb; ++b) {
        int8x16_t wv[8];
        float ws[4];
        quad_block(rows, b, wv, ws);
        sum = quad_step(sum, wv, vld1q_f32(ws), vld1q_s8(q + b * 32),
                        vld1q_s8(q + b * 32 + 16), d[b]);
    }
    return sum;
}

/**
 * A packed quad against T tokens (token t's scales at d + t * nb, its
 * quants at q + t * cols): lane r of sum[t] is row r's dot with token
 * t. Each (row, token) takes the same steps whatever T is.
 */
template <int T>
void quad_tokens(const std::int8_t* qw, const float* qs, std::size_t nb,
                 std::size_t cols, const float* d, const std::int8_t* q,
                 float32x4_t sum[T]) {
    for (int t = 0; t < T; ++t) {
        sum[t] = vdupq_n_f32(0.0f);
    }
    for (std::size_t b = 0; b < nb; ++b) {
        int8x16_t wv[8];
        for (int k = 0; k < 8; ++k) {
            wv[k] = vld1q_s8(qw + b * 128 + k * 16);
        }
        const float32x4_t ws = vld1q_f32(qs + b * 4);
        for (int t = 0; t < T; ++t) {
            const std::int8_t* x = q + t * cols + b * 32;
            sum[t] = quad_step(sum[t], wv, ws, vld1q_s8(x), vld1q_s8(x + 16),
                               d[t * nb + b]);
        }
    }
}

/**
 * Tokens per tile: as many quantized activation rows (cols bytes each)
 * as fit about 32 KiB, beside the packed quad, in a 64 KiB L1.
 */
std::uint32_t token_tile(std::size_t cols) {
    return static_cast<std::uint32_t>(
        std::clamp<std::size_t>((std::size_t{32} << 10) / cols, 4, 64));
}

/**
 * out[r * ostride + t] for every row of w and n tokens (scales d at
 * t * nb, quants q at t * cols): per token tile, each quad of rows is
 * packed once and dotted against the tile's tokens, 4 at a time and
 * then one by one.
 */
void q8_0_rows(const Mat& w, const float* d, const std::int8_t* q,
               std::uint32_t n, float* out, std::size_t ostride) {
    const std::size_t nb = w.cols / 32;
    thread_local std::vector<std::int8_t> qw;
    thread_local std::vector<float> qs;
    qw.resize(nb * 128);
    qs.resize(nb * 4);
    const auto store = [&](std::uint32_t r0, std::uint32_t t,
                           float32x4_t v4) {
        float v[4];
        vst1q_f32(v, v4);
        for (std::uint32_t k = 0; k < 4 && r0 + k < w.rows; ++k) {
            out[std::size_t{r0 + k} * ostride + t] = v[k];
        }
    };
    if (n == 1) {
        for (std::uint32_t r0 = 0; r0 < w.rows; r0 += 4) {
            store(r0, 0, quad_direct(w, r0, nb, d, q));
        }
        return;
    }
    const std::uint32_t tile = token_tile(w.cols);
    for (std::uint32_t t0 = 0; t0 < n; t0 += tile) {
        const std::uint32_t t1 = std::min(n, t0 + tile);
        for (std::uint32_t r0 = 0; r0 < w.rows; r0 += 4) {
            pack_quad(w, r0, nb, qw.data(), qs.data());
            std::uint32_t t = t0;
            for (; t + 4 <= t1; t += 4) {
                float32x4_t sum[4];
                quad_tokens<4>(qw.data(), qs.data(), nb, w.cols,
                               d + std::size_t{t} * nb,
                               q + std::size_t{t} * w.cols, sum);
                for (int k = 0; k < 4; ++k) {
                    store(r0, t + k, sum[k]);
                }
            }
            for (; t < t1; ++t) {
                float32x4_t sum[1];
                quad_tokens<1>(qw.data(), qs.data(), nb, w.cols,
                               d + std::size_t{t} * nb,
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

}  // namespace

void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    if (!takes(w)) {
        base::matvec_neon_q8k(w, x, out);
        return;
    }
    thread_local std::vector<float> d;
    thread_local std::vector<std::int8_t> q;
    d.resize(w.cols / 32);
    q.resize(w.cols);
    quantize_activation_q8_0_neon(x, d.data(), q.data());
    q8_0_rows(w, d.data(), q.data(), 1, out.data(), 1);
}

void matvec_batch_neon_q8k(const Mat& w, std::span<const float> x_batch,
                           std::span<float> out_batch, std::uint32_t n) {
    if (!takes(w)) {
        base::matvec_batch_neon_q8k(w, x_batch, out_batch, n);
        return;
    }
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
    q8_0_rows(w, d.data(), q.data(), n, out_batch.data(), n);
}

}  // namespace locus::backend::neon_dotprod_fast

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif  // __aarch64__ && LOCUS_HAS_NEON_DOTPROD_FAST_VARIANT
