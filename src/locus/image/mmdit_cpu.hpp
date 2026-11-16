#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/image/dense.hpp"

/**
 * The CPU double block that FLUX and Qwen-Image share (the GPU one is
 * in mmdit_vulkan.hpp): the text and image rows keep their own weights
 * but attend jointly, over q, k and v rows joined text first, then
 * image.
 */
namespace locus::image::mmdit {

/** The sizes of one forward's double blocks. */
struct CpuShape {
    std::uint32_t n_txt, n_img;  ///< text and image rows
    std::uint32_t w, heads, hd;  ///< width, heads, head size
    std::uint32_t ffn;           ///< FFN hidden width
    float eps;                   ///< the norms' epsilon
};

/** One forward's scratch rows; the caller may reuse them after. */
struct CpuBufs {
    explicit CpuBufs(const CpuShape& s)
        : q(std::size_t{s.n_txt + s.n_img} * s.w), k(q.size()), v(q.size()),
          att(q.size()), nrm(q.size()), o(q.size()),
          ff(std::size_t{s.n_txt + s.n_img} * s.ffn),
          mod_i(std::size_t{6} * s.w), mod_t(mod_i.size()) {}
    std::vector<float> q, k, v, att, nrm, o, ff;
    std::vector<float> mod_i, mod_t;  ///< each stream's 6 x w modulation
};

/**
 * One double block on the CPU: each stream's modulation from st, its
 * modulated LayerNorm and q / k / v (with the q and k RMS norms), RoPE
 * and joint attention over all rows, then per stream (text first) the
 * gated attention residual and the gated GELU-tanh FFN. mod: shift,
 * scale, gate (attention), shift, scale, gate (FFN).
 *
 * @param lin Runs a layer on rows: lin(layer, in, rows, out).
 * @param lin1 Runs a layer on one vector: lin1(layer, in, out).
 * @param img The image hidden rows (n_img x w), updated.
 * @param txt The text hidden rows (n_txt x w), updated.
 */
template <class Stream, class Lin, class Lin1>
void double_block_cpu(const backend::Ops& op, const Stream& img_s,
                      const Stream& txt_s, const CpuShape& sh,
                      std::span<const float> st, std::span<const float> cosv,
                      std::span<const float> sinv, std::vector<float>& img,
                      std::vector<float>& txt, CpuBufs& b, const Lin& lin,
                      const Lin1& lin1) {
    const std::uint32_t w = sh.w, nh = sh.heads, hd = sh.hd;
    const std::uint32_t n = sh.n_txt + sh.n_img;
    // q / k / v of one stream into rows [row0, row0 + rows) of q, k, v.
    const auto qkv = [&](const Stream& s, std::span<const float> in,
                         std::uint32_t rows, std::uint32_t row0) {
        const std::size_t at = std::size_t{row0} * w;
        const std::size_t len = std::size_t{rows} * w;
        const auto sub = [&](std::vector<float>& v) {
            return std::span<float>(v).subspan(at, len);
        };
        lin(s.q, in, rows, sub(b.q));
        lin(s.k, in, rows, sub(b.k));
        lin(s.v, in, rows, sub(b.v));
        dense::rms_norm(sub(b.q), s.q_norm, sh.eps, rows * nh, hd, sub(b.q));
        dense::rms_norm(sub(b.k), s.k_norm, sh.eps, rows * nh, hd, sub(b.k));
    };
    // Gated residual of one stream's attention output, then its gated
    // FFN on the modulated LayerNorm.
    const auto finish = [&](const Stream& s, std::vector<float>& h,
                            std::uint32_t rows, std::uint32_t row0,
                            const std::vector<float>& mod) {
        const std::size_t len = std::size_t{rows} * w;
        const auto a = std::span<const float>(b.att).subspan(
            std::size_t{row0} * w, len);
        auto ob = std::span<float>(b.o).subspan(0, len);
        lin(s.out, a, rows, ob);
        dense::gated_add(h, ob, &mod[2 * w], rows, w);
        auto nb = std::span<float>(b.nrm).subspan(0, len);
        dense::ln_modulate(h, rows, w, sh.eps, &mod[3 * w], &mod[4 * w], nb);
        auto fb =
            std::span<float>(b.ff).subspan(0, std::size_t{rows} * sh.ffn);
        lin(s.ff1, nb, rows, fb);
        for (float& z : fb) z = dense::gelu_tanh(z);
        lin(s.ff2, fb, rows, ob);
        dense::gated_add(h, ob, &mod[5 * w], rows, w);
    };

    lin1(img_s.mod, st, b.mod_i);
    lin1(txt_s.mod, st, b.mod_t);
    auto ni = std::span<float>(b.nrm).subspan(0, std::size_t{sh.n_img} * w);
    dense::ln_modulate(img, sh.n_img, w, sh.eps, &b.mod_i[0], &b.mod_i[w], ni);
    qkv(img_s, ni, sh.n_img, sh.n_txt);
    auto nt = std::span<float>(b.nrm).subspan(0, std::size_t{sh.n_txt} * w);
    dense::ln_modulate(txt, sh.n_txt, w, sh.eps, &b.mod_t[0], &b.mod_t[w], nt);
    qkv(txt_s, nt, sh.n_txt, 0);
    dense::rope_pairs(b.q, n, nh, hd, cosv, sinv);
    dense::rope_pairs(b.k, n, nh, hd, cosv, sinv);
    dense::attention(op, b.q, n, b.k, b.v, n, nh, hd, b.att);
    finish(txt_s, txt, sh.n_txt, 0, b.mod_t);
    finish(img_s, img, sh.n_img, sh.n_txt, b.mod_i);
}

}  // namespace locus::image::mmdit
