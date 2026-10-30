#include <cstddef>

#include "locus/backend/vulkan_seq.hpp"
#include "locus/image/flux_dit.hpp"

namespace locus::image {

namespace {

using Seq = backend::VulkanSeq;
using Rows = Seq::Rows;

/** The scratch slots of one forward. */
enum : Seq::Slot {
    kH,      ///< hidden rows, joined: text, then image [n][w]
    kNrm,    ///< modulated LayerNorm output [n][w]
    kQ,      ///< [n][w]
    kK,      ///< [n][w]
    kV,      ///< [n][w]
    kAtt,    ///< attention output (double blocks) [n][w]
    kO,      ///< projection output [n][w]
    kFf,     ///< FFN hidden (double) [n][ffn]
    kCat,    ///< [attention, gelu(mlp)] (single) [n][w + ffn]
    kMod,    ///< modulation vectors [12 w]
    kSt,     ///< silu of the conditioning vector [w]
    kCos,    ///< RoPE tables [n][hd]
    kSin,
    kIn,     ///< packed latent, then T5 rows
    kKh,     ///< per-head K, V^T and score blocks (attention)
    kVt,
};
constexpr Seq::Slot kScores = kIn;  // reused once the inputs are embedded

}  // namespace

std::vector<float> FluxDit::forward_vulkan(std::span<const float> x,
                                           std::uint32_t n_img,
                                           std::span<const float> text,
                                           std::uint32_t n_txt,
                                           const std::vector<float>& st,
                                           const std::vector<float>& cosv,
                                           const std::vector<float>& sinv,
                                           Trace* trace) const {
    const std::uint32_t w = width_, hd = head_dim_, nh = heads_, f = ffn_;
    const std::uint32_t n = n_txt + n_img;
    const std::uint32_t ti = n_txt * w;  // first image row of a [n][w]
    Seq q;
    const std::size_t nw = std::size_t{n} * w;
    for (Seq::Slot s : {kH, kNrm, kQ, kK, kV, kAtt, kO}) q.reserve(s, nw);
    q.reserve(kFf, std::size_t{n} * f);
    q.reserve(kCat, std::size_t{n} * (w + f));
    q.reserve(kMod, std::size_t{12} * w);
    q.reserve(kSt, w);
    q.reserve(kCos, cosv.size());
    q.reserve(kSin, sinv.size());
    q.reserve(kIn, x.size() + text.size());
    q.write(kSt, 0, st);
    q.write(kCos, 0, cosv);
    q.write(kSin, 0, sinv);
    q.write(kIn, 0, x);
    q.write(kIn, x.size(), text);

    const auto lin = [&](const Linear& l, Rows in, Rows out,
                         std::uint32_t rows) {
        q.linear(l.w, l.b, in, out, rows);
    };
    const auto mod = [&](const Linear& l, std::uint32_t off) {
        lin(l, {kSt, 0, 0}, {kMod, off, 0}, 1);
    };
    lin(x_embed_, {kIn, 0, in_ch_}, {kH, ti, w}, n_img);
    lin(ctx_embed_, {kIn, static_cast<std::uint32_t>(x.size()), joint_},
        {kH, 0, w}, n_txt);
    q.reserve(kScores, 1);  // kIn's inputs are no longer needed

    // One stream's q / k / v over its rows [row0, row0 + rows) of the
    // joined sequence, normed and roped. mo: its modulation offset.
    const auto qkv = [&](const Stream& s, std::uint32_t rows,
                         std::uint32_t row0, std::uint32_t mo) {
        const std::uint32_t at = row0 * w;
        q.ln_mod({kH, at, w}, {kNrm, at, w}, rows, w, kMod, mo, mo + w,
                 eps_);
        lin(s.q, {kNrm, at, w}, {kQ, at, w}, rows);
        lin(s.k, {kNrm, at, w}, {kK, at, w}, rows);
        lin(s.v, {kNrm, at, w}, {kV, at, w}, rows);
        q.head_norm_rope({kQ, at, w}, rows, nh, hd, s.q_norm, kCos, kSin,
                         row0, eps_);
        q.head_norm_rope({kK, at, w}, rows, nh, hd, s.k_norm, kCos, kSin,
                         row0, eps_);
    };
    // Gated residual of one stream's attention output, then its gated
    // FFN on the modulated LayerNorm.
    const auto finish = [&](const Stream& s, std::uint32_t rows,
                            std::uint32_t row0, std::uint32_t mo) {
        const std::uint32_t at = row0 * w;
        lin(s.out, {kAtt, at, w}, {kO, at, w}, rows);
        q.gated_add({kH, at, w}, {kO, at, w}, rows, w, kMod, mo + 2 * w);
        q.ln_mod({kH, at, w}, {kNrm, at, w}, rows, w, kMod, mo + 3 * w,
                 mo + 4 * w, eps_);
        lin(s.ff1, {kNrm, at, w}, {kFf, row0 * f, f}, rows);
        q.gelu_tanh({kFf, row0 * f, f}, rows, f);
        lin(s.ff2, {kFf, row0 * f, f}, {kO, at, w}, rows);
        q.gated_add({kH, at, w}, {kO, at, w}, rows, w, kMod, mo + 5 * w);
    };
    const auto read_rows = [&](std::uint32_t row0, std::uint32_t rows) {
        std::vector<float> v(std::size_t{rows} * w);
        q.read(kH, std::size_t{row0} * w, v);
        return v;
    };

    for (std::size_t bi = 0; bi < doubles_.size(); ++bi) {
        const DoubleBlock& b = doubles_[bi];
        mod(b.img.mod, 0);
        mod(b.txt.mod, 6 * w);
        qkv(b.img, n_img, n_txt, 0);
        qkv(b.txt, n_txt, 0, 6 * w);
        q.attention({kQ, 0, w}, {kK, 0, w}, {kV, 0, w}, {kAtt, 0, w}, n, nh,
                    hd, kKh, kVt, kScores);
        finish(b.txt, n_txt, 0, 6 * w);
        finish(b.img, n_img, n_txt, 0);
        q.flush();
        if (trace != nullptr && bi == 0) {
            trace->double0_img = read_rows(n_txt, n_img);
            trace->double0_txt = read_rows(0, n_txt);
        }
    }

    // Single blocks on the joined rows. kCat holds [attention, gelu(mlp)]
    // per token, so proj_out reads both in one matmul.
    const std::uint32_t cw = w + f;
    for (std::size_t bi = 0; bi < singles_.size(); ++bi) {
        const SingleBlock& b = singles_[bi];
        mod(b.mod, 0);
        q.ln_mod({kH, 0, w}, {kNrm, 0, w}, n, w, kMod, 0, w, eps_);
        lin(b.mlp, {kNrm, 0, w}, {kCat, w, cw}, n);
        q.gelu_tanh({kCat, w, cw}, n, f);
        lin(b.q, {kNrm, 0, w}, {kQ, 0, w}, n);
        lin(b.k, {kNrm, 0, w}, {kK, 0, w}, n);
        lin(b.v, {kNrm, 0, w}, {kV, 0, w}, n);
        q.head_norm_rope({kQ, 0, w}, n, nh, hd, b.q_norm, kCos, kSin, 0, eps_);
        q.head_norm_rope({kK, 0, w}, n, nh, hd, b.k_norm, kCos, kSin, 0, eps_);
        q.attention({kQ, 0, w}, {kK, 0, w}, {kV, 0, w}, {kCat, 0, cw}, n, nh,
                    hd, kKh, kVt, kScores);
        lin(b.out, {kCat, 0, cw}, {kO, 0, w}, n);
        q.gated_add({kH, 0, w}, {kO, 0, w}, n, w, kMod, 2 * w);
        q.flush();
        if (trace != nullptr && bi == 0) trace->single0 = read_rows(0, n);
    }

    // norm_out (scale, then shift) on the image rows, then proj_out.
    mod(norm_out_, 0);
    q.ln_mod({kH, ti, w}, {kNrm, 0, w}, n_img, w, kMod, w, 0, eps_);
    lin(proj_out_, {kNrm, 0, w}, {kO, 0, in_ch_}, n_img);
    std::vector<float> out(std::size_t{n_img} * in_ch_);
    q.read(kO, 0, out);
    return out;
}

}  // namespace locus::image
