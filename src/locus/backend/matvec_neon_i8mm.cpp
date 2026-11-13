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
 * Activations packed for smmla, once per call: token pair p (tokens
 * 2p, 2p + 1; an odd last token paired with itself), block b holds the
 * four interleaved vectors at qp[(p * nb + b) * 64] and the two scales
 * at dp[(p * nb + b) * 2].
 */
struct Packed {
    std::vector<std::int8_t> qp;
    std::vector<float> dp;
};

void pack(const float* d, const std::int8_t* q, std::uint32_t n,
          std::size_t nb, std::size_t cols, Packed& pk) {
    const std::uint32_t pairs = (n + 1) / 2;
    pk.qp.resize(std::size_t{pairs} * nb * 64);
    pk.dp.resize(std::size_t{pairs} * nb * 2);
    for (std::uint32_t p = 0; p < pairs; ++p) {
        const std::uint32_t t0 = 2 * p, t1 = t0 + 1 < n ? t0 + 1 : t0;
        const std::int8_t* x = q + t0 * cols;
        const std::int8_t* y = q + t1 * cols;
        for (std::size_t b = 0; b < nb; ++b) {
            const int8x16_t x_l = vld1q_s8(x + b * 32);
            const int8x16_t x_h = vld1q_s8(x + b * 32 + 16);
            const int8x16_t y_l = vld1q_s8(y + b * 32);
            const int8x16_t y_h = vld1q_s8(y + b * 32 + 16);
            std::int8_t* o = pk.qp.data() + (p * nb + b) * 64;
            vst1q_s8(o, zip_lo(x_l, y_l));
            vst1q_s8(o + 16, zip_hi(x_l, y_l));
            vst1q_s8(o + 32, zip_lo(x_h, y_h));
            vst1q_s8(o + 48, zip_hi(x_h, y_h));
            pk.dp[(p * nb + b) * 2] = d[t0 * nb + b];
            pk.dp[(p * nb + b) * 2 + 1] = d[t1 * nb + b];
        }
    }
}

/**
 * Rows r0, r1 (weight blocks at w0, w1) against `np` (1 or 2) packed
 * token pairs starting at pair p, over nb blocks: acc[i] = {r0 tA,
 * r0 tB, r1 tA, r1 tB} for pair p + i. Each output takes the same
 * steps (an exact int32 per block, times s_w * d_x, then added)
 * whatever np is.
 */
template <int NP>
void rows_pairs(const std::byte* w0, const std::byte* w1, const Packed& pk,
                std::size_t p, std::size_t nb, float32x4_t acc[NP]) {
    for (int i = 0; i < NP; ++i) {
        acc[i] = vdupq_n_f32(0.0f);
    }
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
        for (int i = 0; i < NP; ++i) {
            const std::size_t at = (p + i) * nb + b;
            const std::int8_t* x = pk.qp.data() + at * 64;
            int32x4_t s = vdupq_n_s32(0);
            s = vmmlaq_s32(s, w_0, vld1q_s8(x));
            s = vmmlaq_s32(s, w_1, vld1q_s8(x + 16));
            s = vmmlaq_s32(s, w_2, vld1q_s8(x + 32));
            s = vmmlaq_s32(s, w_3, vld1q_s8(x + 48));
            const float dA = pk.dp[at * 2], dB = pk.dp[at * 2 + 1];
            const float sc4[4] = {sa * dA, sa * dB, sc * dA, sc * dB};
            // A multiply, then an add (no fused multiply-add), per lane.
            acc[i] = vaddq_f32(acc[i],
                               vmulq_f32(vcvtq_f32_s32(s), vld1q_f32(sc4)));
        }
    }
}

/**
 * out[r * ostride + t] for every row of w and n tokens (scales d at
 * t * nb, quants q at t * cols), two token pairs per weight load. An
 * odd last row or token is paired with itself; its duplicate result is
 * dropped.
 */
void q8_0_rows(const Mat& w, const float* d, const std::int8_t* q,
               std::uint32_t n, float* out, std::size_t ostride) {
    const std::size_t nb = w.cols / 32;
    thread_local Packed pk;
    pack(d, q, n, nb, w.cols, pk);
    const std::uint32_t pairs = (n + 1) / 2;
    // Writes pair p's four lanes (r0 tA, r0 tB, r1 tA, r1 tB).
    auto store = [&](std::uint32_t r, std::uint32_t r1, std::uint32_t p,
                     float32x4_t v4) {
        float v[4];
        vst1q_f32(v, v4);
        const std::uint32_t t = 2 * p;
        const bool two = t + 1 < n;
        out[std::size_t{r} * ostride + t] = v[0];
        if (two) {
            out[std::size_t{r} * ostride + t + 1] = v[1];
        }
        if (r1 != r) {
            out[std::size_t{r1} * ostride + t] = v[2];
            if (two) {
                out[std::size_t{r1} * ostride + t + 1] = v[3];
            }
        }
    };
    for (std::uint32_t r = 0; r < w.rows; r += 2) {
        const std::uint32_t r1 = r + 1 < w.rows ? r + 1 : r;
        const std::byte* w0 = w.data + std::size_t{r} * nb * kBlk;
        const std::byte* w1 = w.data + std::size_t{r1} * nb * kBlk;
        std::uint32_t p = 0;
        for (; p + 2 <= pairs; p += 2) {
            float32x4_t acc[2];
            rows_pairs<2>(w0, w1, pk, p, nb, acc);
            store(r, r1, p, acc[0]);
            store(r, r1, p + 1, acc[1]);
        }
        if (p < pairs) {
            float32x4_t acc[1];
            rows_pairs<1>(w0, w1, pk, p, nb, acc);
            store(r, r1, p, acc[0]);
        }
    }
}

// Other types go to the sdot kernels: every i8mm CPU has dotprod, and
// where the dot product is a variant of its own (GCC's default arm64
// target), the baseline build would be the slower one.
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

}  // namespace locus::backend::neon_i8mm

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif

#endif  // __aarch64__ && LOCUS_HAS_NEON_I8MM_VARIANT
