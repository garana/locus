#include <algorithm>
#include <cstddef>

#include "locus/backend/vulkan_seq.hpp"
#include "locus/image/pixart_dit.hpp"

namespace locus::image {

namespace {

using Seq = backend::VulkanSeq;
using Rows = Seq::Rows;

/** The scratch slots of one forward. */
enum : Seq::Slot {
    kH,    ///< hidden rows [n][w]
    kNrm,  ///< position table, then modulated LayerNorm output [n][w]
    kQ,    ///< [n][w]
    kK,    ///< [n][w]
    kV,    ///< [n][w]
    kAtt,  ///< caption hidden, then attention output [n][w]
    kO,    ///< projection output [n][max(w, p p out_ch)]
    kFf,   ///< T5 rows, then the FFN hidden [n][ffn]
    kMod,  ///< every block's modulation [layers][6 w], then the final
           ///< shift and scale [2 w]
    kU,    ///< projected caption rows [n_text][w]
    kKc,   ///< cross-attention keys and values [n_text][w]
    kVc,
    kIn,   ///< patches [n][in_ch p p]
    kKh,   ///< per-head K, V^T and score blocks (attention)
    kVt,
};
constexpr Seq::Slot kScores = kIn;  // reused once the patches are embedded

}  // namespace

std::vector<float> PixArtDit::forward_vulkan(
    std::span<const float> patches, std::span<const float> text,
    std::uint32_t n_text, const std::vector<float>& mods) const {
    const std::uint32_t w = width_, nh = heads_, hd = head_dim_, f = ffn_;
    const std::uint32_t n = grid_ * grid_;
    const std::uint32_t pv = in_ch_ * patch_ * patch_;
    const std::uint32_t py = patch_ * patch_ * out_ch_;
    Seq q;
    const std::size_t nw = std::size_t{n} * w;
    for (Seq::Slot s : {kH, kNrm, kQ, kK, kV, kAtt}) q.reserve(s, nw);
    q.reserve(kO, std::size_t{n} * std::max(w, py));
    q.reserve(kFf, std::max(std::size_t{n} * f, text.size()));
    q.reserve(kMod, mods.size());
    for (Seq::Slot s : {kU, kKc, kVc}) q.reserve(s, std::size_t{n_text} * w);
    q.reserve(kIn, patches.size());
    q.write(kIn, 0, patches);
    q.write(kNrm, 0, pos_);
    q.write(kMod, 0, mods);
    q.write(kFf, 0, text);

    const auto lin = [&](const Linear& l, Rows in, Rows out,
                         std::uint32_t rows) {
        q.linear(l.w, l.b, in, out, rows);
    };
    // Patch embed plus the position table.
    lin(patch_embd_, {kIn, 0, pv}, {kH, 0, w}, n);
    q.add({kH, 0, w}, {kNrm, 0, w}, n, w);
    q.reserve(kScores, 1);  // kIn's patches are no longer needed
    // Caption projection over the valid T5 rows.
    lin(cap1_, {kFf, 0, cap_ch_}, {kAtt, 0, w}, n_text);
    q.gelu_tanh({kAtt, 0, w}, n_text, w);
    lin(cap2_, {kAtt, 0, w}, {kU, 0, w}, n_text);

    for (std::size_t bi = 0; bi < blocks_.size(); ++bi) {
        const Block& b = blocks_[bi];
        // Rows of this block's [6][w]: shift_msa, scale_msa, gate_msa,
        // shift_mlp, scale_mlp, gate_mlp.
        const auto mo = static_cast<std::uint32_t>(bi * 6 * w);
        q.ln_mod({kH, 0, w}, {kNrm, 0, w}, n, w, kMod, mo, mo + w, eps_);
        lin(b.q1, {kNrm, 0, w}, {kQ, 0, w}, n);
        lin(b.k1, {kNrm, 0, w}, {kK, 0, w}, n);
        lin(b.v1, {kNrm, 0, w}, {kV, 0, w}, n);
        q.attention({kQ, 0, w}, {kK, 0, w}, {kV, 0, w}, {kAtt, 0, w}, n, nh,
                    hd, kKh, kVt, kScores);
        lin(b.o1, {kAtt, 0, w}, {kO, 0, w}, n);
        q.gated_add({kH, 0, w}, {kO, 0, w}, n, w, kMod, mo + 2 * w);

        lin(b.q2, {kH, 0, w}, {kQ, 0, w}, n);
        lin(b.k2, {kU, 0, w}, {kKc, 0, w}, n_text);
        lin(b.v2, {kU, 0, w}, {kVc, 0, w}, n_text);
        q.cross_attention({kQ, 0, w}, n, {kKc, 0, w}, {kVc, 0, w}, n_text,
                          {kAtt, 0, w}, nh, hd, kKh, kVt, kScores);
        lin(b.o2, {kAtt, 0, w}, {kO, 0, w}, n);
        q.add({kH, 0, w}, {kO, 0, w}, n, w);

        q.ln_mod({kH, 0, w}, {kNrm, 0, w}, n, w, kMod, mo + 3 * w,
                 mo + 4 * w, eps_);
        lin(b.up, {kNrm, 0, w}, {kFf, 0, f}, n);
        q.gelu_tanh({kFf, 0, f}, n, f);
        lin(b.down, {kFf, 0, f}, {kO, 0, w}, n);
        q.gated_add({kH, 0, w}, {kO, 0, w}, n, w, kMod, mo + 5 * w);
        q.flush();
    }

    // Final layer: shift and scale follow the blocks' rows in kMod.
    const auto fo = static_cast<std::uint32_t>(blocks_.size() * 6 * w);
    q.ln_mod({kH, 0, w}, {kNrm, 0, w}, n, w, kMod, fo, fo + w, 1e-6f);
    lin(proj_out_, {kNrm, 0, w}, {kO, 0, py}, n);
    std::vector<float> y(std::size_t{n} * py);
    q.read(kO, 0, y);
    return y;
}

}  // namespace locus::image
