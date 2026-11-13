// Q8_0 weights x Q8_0 activations with the 8-bit matrix multiply
// (smmla, FEAT_I8MM), picked only under LOCUS_CPU_FAST (cpu_fast) on a
// CPU that has it. Each smmla takes 8 int8 of two weight rows against
// 8 int8 of two tokens, so a pair of rows meets a pair of tokens in 4
// instructions per block, where the sdot kernels need 8.
//
// The float order is llama.cpp's ggml_vec_dot_q8_0_q8_0 (nrc 2): one
// exact integer dot per block, times d_w * d_x, added to the element's
// sum block by block. That is not the baseline kernels' order (four
// lanes per block, reduced at the end), so results differ from them in
// the low bits; hence the opt-in. Every (row, token) element follows
// the same sequence whatever it is paired with, so a token's result is
// the same alone (matvec_neon_q8k) or in a batch (matvec_batch_neon_q8k)
// under the flag.
//
// Like matvec_neon_dotprod.cpp, the file keeps the baseline flags and
// gives only its own functions the i8mm target (a pragma opened after
// the headers).

#include "locus/backend/variants.hpp"

#if defined(__aarch64__) && defined(LOCUS_HAS_NEON_I8MM_VARIANT)

#include <arm_neon.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"

#if defined(__clang__)
#pragma clang attribute push( \
    __attribute__((target("arch=armv8.2-a+dotprod+i8mm"))), \
    apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod+i8mm")
#endif

namespace locus::backend::neon_i8mm {

namespace {

constexpr std::size_t kBlk = 34;  // f16 scale + 32 int8

/** A block's f16 scale, through the hardware convert (exact). */
float scale_of(const std::byte* blk) {
    std::uint16_t h;
    std::memcpy(&h, blk, sizeof(h));
    return vget_lane_f32(
        vget_low_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h)))), 0);
}

/** The 64-bit halves of a and b interleaved: (a.lo, b.lo), (a.hi,
 *  b.hi), smmla's 2 x 8 operand layout. */
int8x16_t zip_lo(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(
        vzip1q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}
int8x16_t zip_hi(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_s64(
        vzip2q_s64(vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b)));
}

/**
 * Rows r0, r1 (weight blocks at w0, w1) against tokens t0, t1 (scales
 * d0, d1, quants q0, q1) over nb blocks. out = {r0 t0, r0 t1, r1 t0,
 * r1 t1}.
 */
float32x4_t pair_dot(const std::byte* w0, const std::byte* w1,
                     const float* d0, const float* d1,
                     const std::int8_t* q0, const std::int8_t* q1,
                     std::size_t nb) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* a = w0 + b * kBlk;
        const std::byte* c = w1 + b * kBlk;
        const auto* aq = reinterpret_cast<const std::int8_t*>(a + 2);
        const auto* cq = reinterpret_cast<const std::int8_t*>(c + 2);
        const int8x16_t a_l = vld1q_s8(aq), a_h = vld1q_s8(aq + 16);
        const int8x16_t c_l = vld1q_s8(cq), c_h = vld1q_s8(cq + 16);
        const int8x16_t x_l = vld1q_s8(q0 + b * 32);
        const int8x16_t x_h = vld1q_s8(q0 + b * 32 + 16);
        const int8x16_t y_l = vld1q_s8(q1 + b * 32);
        const int8x16_t y_h = vld1q_s8(q1 + b * 32 + 16);
        int32x4_t s = vdupq_n_s32(0);
        s = vmmlaq_s32(s, zip_lo(a_l, c_l), zip_lo(x_l, y_l));
        s = vmmlaq_s32(s, zip_hi(a_l, c_l), zip_hi(x_l, y_l));
        s = vmmlaq_s32(s, zip_lo(a_h, c_h), zip_lo(x_h, y_h));
        s = vmmlaq_s32(s, zip_hi(a_h, c_h), zip_hi(x_h, y_h));
        const float sa = scale_of(a), sc = scale_of(c);
        const float sc4[4] = {sa * d0[b], sa * d1[b], sc * d0[b],
                              sc * d1[b]};
        // A multiply, then an add (no fused multiply-add), per lane.
        acc = vaddq_f32(acc, vmulq_f32(vcvtq_f32_s32(s), vld1q_f32(sc4)));
    }
    return acc;
}

/**
 * pair_dot for rows r0, r1 against two token pairs at once (t0, t1)
 * and (t2, t3): each block's weights are loaded and interleaved once
 * for four tokens. Each output's sequence of steps is pair_dot's.
 */
void pair_dot2(const std::byte* w0, const std::byte* w1,
               const float* const d[4], const std::int8_t* const q[4],
               std::size_t nb, float32x4_t& acc01, float32x4_t& acc23) {
    acc01 = vdupq_n_f32(0.0f);
    acc23 = vdupq_n_f32(0.0f);
    for (std::size_t b = 0; b < nb; ++b) {
        const std::byte* a = w0 + b * kBlk;
        const std::byte* c = w1 + b * kBlk;
        const auto* aq = reinterpret_cast<const std::int8_t*>(a + 2);
        const auto* cq = reinterpret_cast<const std::int8_t*>(c + 2);
        const int8x16_t a_l = vld1q_s8(aq), a_h = vld1q_s8(aq + 16);
        const int8x16_t c_l = vld1q_s8(cq), c_h = vld1q_s8(cq + 16);
        const int8x16_t w_0 = zip_lo(a_l, c_l), w_1 = zip_hi(a_l, c_l);
        const int8x16_t w_2 = zip_lo(a_h, c_h), w_3 = zip_hi(a_h, c_h);
        const float sa = scale_of(a), sc = scale_of(c);
        int32x4_t s[2];
        for (int p = 0; p < 2; ++p) {
            const std::int8_t* x = q[2 * p] + b * 32;
            const std::int8_t* y = q[2 * p + 1] + b * 32;
            const int8x16_t x_l = vld1q_s8(x), x_h = vld1q_s8(x + 16);
            const int8x16_t y_l = vld1q_s8(y), y_h = vld1q_s8(y + 16);
            int32x4_t t = vdupq_n_s32(0);
            t = vmmlaq_s32(t, w_0, zip_lo(x_l, y_l));
            t = vmmlaq_s32(t, w_1, zip_hi(x_l, y_l));
            t = vmmlaq_s32(t, w_2, zip_lo(x_h, y_h));
            t = vmmlaq_s32(t, w_3, zip_hi(x_h, y_h));
            s[p] = t;
        }
        const float sc01[4] = {sa * d[0][b], sa * d[1][b], sc * d[0][b],
                               sc * d[1][b]};
        const float sc23[4] = {sa * d[2][b], sa * d[3][b], sc * d[2][b],
                               sc * d[3][b]};
        acc01 = vaddq_f32(acc01,
                          vmulq_f32(vcvtq_f32_s32(s[0]), vld1q_f32(sc01)));
        acc23 = vaddq_f32(acc23,
                          vmulq_f32(vcvtq_f32_s32(s[1]), vld1q_f32(sc23)));
    }
}

/**
 * out[r * ostride + t] for every row of w and n tokens (scales d at
 * t * nb, quants q at t * cols), four tokens per weight load. An odd
 * last row or token is paired with itself; its duplicate result is
 * dropped.
 */
void q8_0_rows(const Mat& w, const float* d, const std::int8_t* q,
               std::uint32_t n, float* out, std::size_t ostride) {
    const std::size_t nb = w.cols / 32;
    for (std::uint32_t r = 0; r < w.rows; r += 2) {
        const std::uint32_t r1 = r + 1 < w.rows ? r + 1 : r;
        const std::byte* w0 = w.data + std::size_t{r} * nb * kBlk;
        const std::byte* w1 = w.data + std::size_t{r1} * nb * kBlk;
        std::uint32_t t = 0;
        for (; t + 4 <= n; t += 4) {
            const float* dd[4];
            const std::int8_t* qq[4];
            for (std::uint32_t k = 0; k < 4; ++k) {
                dd[k] = d + std::size_t{t + k} * nb;
                qq[k] = q + std::size_t{t + k} * w.cols;
            }
            float32x4_t a01, a23;
            pair_dot2(w0, w1, dd, qq, nb, a01, a23);
            float v[8];
            vst1q_f32(v, a01);
            vst1q_f32(v + 4, a23);
            for (std::uint32_t k = 0; k < 4; ++k) {
                // Lanes: (r0 tA, r0 tB, r1 tA, r1 tB) per pair.
                const float* p = v + (k / 2) * 4;
                out[std::size_t{r} * ostride + t + k] = p[k % 2];
                if (r1 != r) {
                    out[std::size_t{r1} * ostride + t + k] = p[2 + k % 2];
                }
            }
        }
        for (; t < n; t += 2) {
            const std::uint32_t t1 = t + 1 < n ? t + 1 : t;
            float v[4];
            vst1q_f32(v, pair_dot(w0, w1, d + std::size_t{t} * nb,
                                  d + std::size_t{t1} * nb,
                                  q + std::size_t{t} * w.cols,
                                  q + std::size_t{t1} * w.cols, nb));
            out[std::size_t{r} * ostride + t] = v[0];
            if (t1 != t) {
                out[std::size_t{r} * ostride + t1] = v[1];
            }
            if (r1 != r) {
                out[std::size_t{r1} * ostride + t] = v[2];
                if (t1 != t) {
                    out[std::size_t{r1} * ostride + t1] = v[3];
                }
            }
        }
    }
}

bool takes(const Mat& w) {
    return w.type == gguf::TensorType::kQ8_0 && w.cols % 32 == 0 &&
           w.cols > 0;
}

}  // namespace

void matvec_neon_q8k(const Mat& w, std::span<const float> x,
                     std::span<float> out) {
    if (!takes(w)) {
        ::locus::backend::matvec_neon_q8k(w, x, out);
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
        ::locus::backend::matvec_batch_neon_q8k(w, x_batch, out_batch, n);
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

}  // namespace locus::backend::neon_i8mm

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif  // __aarch64__ && LOCUS_HAS_NEON_I8MM_VARIANT
