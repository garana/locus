#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/vulkan_seq.hpp"

/**
 * The GPU forward pieces that FLUX and Qwen-Image share: both are
 * double-stream DiTs (MMDiT), where the text and image rows keep their
 * own weights but attend jointly. The hidden rows sit in one [n][w]
 * slot, text first, then image.
 */
namespace locus::image::mmdit {

using Seq = backend::VulkanSeq;
using Rows = Seq::Rows;

/** The scratch slots of one forward; a model adds its own from kFirst. */
enum : Seq::Slot {
    kH,      ///< hidden rows, joined: text, then image [n][w]
    kNrm,    ///< modulated LayerNorm output [n][w]
    kQ,      ///< [n][w]
    kK,      ///< [n][w]
    kV,      ///< [n][w]
    kAtt,    ///< attention output [n][w]
    kO,      ///< projection output [n][w]
    kFf,     ///< FFN hidden [n][ffn]
    kMod,    ///< modulation vectors [12 w]: image stream, then text
    kSt,     ///< silu of the conditioning vector [w]
    kCos,    ///< RoPE tables [n][hd]
    kSin,
    kIn,     ///< image inputs, then text inputs
    kKh,     ///< per-head K, V^T and score blocks (attention)
    kVt,
    kFirst,  ///< the first slot a model may add
};
constexpr Seq::Slot kScores = kIn;  // reused once the inputs are embedded

/** The sizes of one forward. */
struct Dims {
    std::uint32_t w = 0;      ///< hidden width
    std::uint32_t hd = 0;     ///< head width
    std::uint32_t nh = 0;     ///< heads
    std::uint32_t f = 0;      ///< FFN width
    std::uint32_t n_img = 0;  ///< image rows
    std::uint32_t n_txt = 0;  ///< text rows
    float eps = 0.0f;
    std::uint32_t n() const { return n_txt + n_img; }
};

/**
 * Reserves the shared slots and uploads the conditioning vector st, the
 * RoPE tables and the inputs (x, then text) into kIn.
 */
inline void setup(Seq& q, const Dims& d, std::span<const float> x,
                  std::span<const float> text, std::span<const float> st,
                  std::span<const float> cosv, std::span<const float> sinv) {
    const std::size_t nw = std::size_t{d.n()} * d.w;
    for (Seq::Slot s : {kH, kNrm, kQ, kK, kV, kAtt, kO}) q.reserve(s, nw);
    q.reserve(kFf, std::size_t{d.n()} * d.f);
    q.reserve(kMod, std::size_t{12} * d.w);
    q.reserve(kSt, d.w);
    q.reserve(kCos, cosv.size());
    q.reserve(kSin, sinv.size());
    q.reserve(kIn, x.size() + text.size());
    q.write(kSt, 0, st);
    q.write(kCos, 0, cosv);
    q.write(kSin, 0, sinv);
    q.write(kIn, 0, x);
    q.write(kIn, x.size(), text);
}

/** @returns rows [row0, row0 + rows) of kH, read back to the host. */
inline std::vector<float> read_rows(Seq& q, const Dims& d, std::uint32_t row0,
                                    std::uint32_t rows) {
    std::vector<float> v(std::size_t{rows} * d.w);
    q.read(kH, std::size_t{row0} * d.w, v);
    return v;
}

/**
 * One double-stream block on kH, then a flush. Each stream (fields mod,
 * q, k, v, q_norm, k_norm, out, ff1, ff2) modulates, projects, norms
 * and ropes its own rows; one attention runs over all n rows; then each
 * stream takes its gated residual and gated FFN.
 *
 * @param lin lin(layer, in, out, rows): one layer on rows token rows.
 */
template <class Stream, class Lin>
void double_block(Seq& q, const Dims& d, const Stream& img,
                  const Stream& txt, const Lin& lin) {
    const std::uint32_t w = d.w, f = d.f;
    const auto mod = [&](const auto& l, std::uint32_t off) {
        lin(l, Rows{kSt, 0, 0}, Rows{kMod, off, 0}, 1);
    };
    // One stream's q / k / v over its rows [row0, row0 + rows) of the
    // joined sequence, normed and roped. mo: its modulation offset.
    const auto qkv = [&](const Stream& s, std::uint32_t rows,
                         std::uint32_t row0, std::uint32_t mo) {
        const std::uint32_t at = row0 * w;
        q.ln_mod({kH, at, w}, {kNrm, at, w}, rows, w, kMod, mo, mo + w,
                 d.eps);
        lin(s.q, Rows{kNrm, at, w}, Rows{kQ, at, w}, rows);
        lin(s.k, Rows{kNrm, at, w}, Rows{kK, at, w}, rows);
        lin(s.v, Rows{kNrm, at, w}, Rows{kV, at, w}, rows);
        q.head_norm_rope({kQ, at, w}, rows, d.nh, d.hd, s.q_norm, kCos, kSin,
                         row0, d.eps);
        q.head_norm_rope({kK, at, w}, rows, d.nh, d.hd, s.k_norm, kCos, kSin,
                         row0, d.eps);
    };
    // Gated residual of one stream's attention output, then its gated
    // FFN on the modulated LayerNorm.
    const auto finish = [&](const Stream& s, std::uint32_t rows,
                            std::uint32_t row0, std::uint32_t mo) {
        const std::uint32_t at = row0 * w;
        lin(s.out, Rows{kAtt, at, w}, Rows{kO, at, w}, rows);
        q.gated_add({kH, at, w}, {kO, at, w}, rows, w, kMod, mo + 2 * w);
        q.ln_mod({kH, at, w}, {kNrm, at, w}, rows, w, kMod, mo + 3 * w,
                 mo + 4 * w, d.eps);
        lin(s.ff1, Rows{kNrm, at, w}, Rows{kFf, row0 * f, f}, rows);
        q.gelu_tanh({kFf, row0 * f, f}, rows, f);
        lin(s.ff2, Rows{kFf, row0 * f, f}, Rows{kO, at, w}, rows);
        q.gated_add({kH, at, w}, {kO, at, w}, rows, w, kMod, mo + 5 * w);
    };
    mod(img.mod, 0);
    mod(txt.mod, 6 * w);
    qkv(img, d.n_img, d.n_txt, 0);
    qkv(txt, d.n_txt, 0, 6 * w);
    q.attention({kQ, 0, w}, {kK, 0, w}, {kV, 0, w}, {kAtt, 0, w}, d.n(),
                d.nh, d.hd, kKh, kVt, kScores);
    finish(txt, d.n_txt, 0, 6 * w);
    finish(img, d.n_img, d.n_txt, 0);
    q.flush();
}

/**
 * The output layer: norm_out's modulation (scale, then shift) on the
 * image rows, then proj_out to out_w channels, read back.
 */
template <class Layer, class Lin>
std::vector<float> output(Seq& q, const Dims& d, const Layer& norm_out,
                          const Layer& proj_out, std::uint32_t out_w,
                          const Lin& lin) {
    const std::uint32_t w = d.w;
    lin(norm_out, Rows{kSt, 0, 0}, Rows{kMod, 0, 0}, 1);
    q.ln_mod({kH, d.n_txt * w, w}, {kNrm, 0, w}, d.n_img, w, kMod, w, 0,
             d.eps);
    lin(proj_out, Rows{kNrm, 0, w}, Rows{kO, 0, out_w}, d.n_img);
    std::vector<float> out(std::size_t{d.n_img} * out_w);
    q.read(kO, 0, out);
    return out;
}

}  // namespace locus::image::mmdit
