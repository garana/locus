#include <cstddef>

#include "locus/image/flux_dit.hpp"
#include "locus/image/mmdit_vulkan.hpp"

namespace locus::image {

namespace {

using namespace mmdit;

/** FLUX's own scratch slots, after the double-stream ones. */
enum : Seq::Slot {
    kCat = kFirst,  ///< [attention, gelu(mlp)] (single) [n][w + ffn]
    kLora,          ///< a LoRA's A x rows [n][rank]
};

}  // namespace

std::vector<float> FluxDit::forward_vulkan(std::span<const float> x,
                                           std::uint32_t n_img,
                                           std::span<const float> text,
                                           std::uint32_t n_txt,
                                           const std::vector<float>& st,
                                           const std::vector<float>& cosv,
                                           const std::vector<float>& sinv,
                                           Trace* trace) const {
    const Dims d{width_, head_dim_, heads_, ffn_, n_img, n_txt, eps_};
    const std::uint32_t w = width_, hd = head_dim_, nh = heads_, f = ffn_;
    const std::uint32_t n = d.n();
    const std::uint32_t ti = n_txt * w;  // first image row of a [n][w]
    Seq q;
    setup(q, d, x, text, st, cosv, sinv);
    q.reserve(kCat, std::size_t{n} * (w + f));

    // A layer, plus each LoRA part on it (i#372): out += strength *
    // (s B)(A x), as the CPU path's apply_lora.
    const auto lin = [&](const Linear& l, Rows in, Rows out,
                         std::uint32_t rows) {
        q.linear(l.w, l.b, in, out, rows);
        for (const LoraPart& p : l.lora) {
            q.reserve(kLora, std::size_t{rows} * p.a.rows);
            q.linear(p.a, {}, in, {kLora, 0, p.a.rows}, rows);
            q.linear_add(p.b, {kLora, 0, p.a.rows}, out, rows, p.strength);
        }
    };
    lin(x_embed_, {kIn, 0, in_ch_}, {kH, ti, w}, n_img);
    lin(ctx_embed_, {kIn, static_cast<std::uint32_t>(x.size()), joint_},
        {kH, 0, w}, n_txt);
    q.reserve(kScores, 1);  // kIn's inputs are no longer needed

    for (std::size_t bi = 0; bi < doubles_.size(); ++bi) {
        double_block(q, d, doubles_[bi].img, doubles_[bi].txt, lin);
        if (trace != nullptr && bi == 0) {
            trace->double0_img = read_rows(q, d, n_txt, n_img);
            trace->double0_txt = read_rows(q, d, 0, n_txt);
        }
    }

    // Single blocks on the joined rows. kCat holds [attention, gelu(mlp)]
    // per token, so proj_out reads both in one matmul.
    const std::uint32_t cw = w + f;
    for (std::size_t bi = 0; bi < singles_.size(); ++bi) {
        const SingleBlock& b = singles_[bi];
        lin(b.mod, {kSt, 0, 0}, {kMod, 0, 0}, 1);
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
        if (trace != nullptr && bi == 0) trace->single0 = read_rows(q, d, 0, n);
    }
    return output(q, d, norm_out_, proj_out_, in_ch_, lin);
}

}  // namespace locus::image
