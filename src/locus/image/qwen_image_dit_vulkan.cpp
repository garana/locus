#include <cstddef>

#include "locus/image/mmdit_vulkan.hpp"
#include "locus/image/qwen_image_dit.hpp"

namespace locus::image {

std::vector<float> QwenImageDit::forward_vulkan(
    std::span<const float> x, std::uint32_t n_img,
    std::span<const float> text_normed, std::uint32_t n_txt,
    const std::vector<float>& st, const std::vector<float>& cosv,
    const std::vector<float>& sinv, Trace* trace) const {
    using namespace mmdit;
    const Dims d{width_, head_dim_, heads_, ffn_, n_img, n_txt, eps_};
    const std::uint32_t w = width_;
    const std::uint32_t ti = n_txt * w;  // first image row of a [n][w]
    Seq q;
    setup(q, d, x, text_normed, st, cosv, sinv);

    const auto lin = [&](const Linear& l, Rows in, Rows out,
                         std::uint32_t rows) {
        q.linear(l.w, l.b, in, out, rows);
    };
    lin(img_in_, {kIn, 0, in_ch_}, {kH, ti, w}, n_img);
    lin(txt_in_, {kIn, static_cast<std::uint32_t>(x.size()), joint_},
        {kH, 0, w}, n_txt);
    q.reserve(kScores, 1);  // kIn's inputs are no longer needed

    for (std::size_t bi = 0; bi < blocks_.size(); ++bi) {
        double_block(q, d, blocks_[bi].img, blocks_[bi].txt, lin);
        if (trace != nullptr && bi == 0) {
            trace->block0_img = read_rows(q, d, n_txt, n_img);
            trace->block0_txt = read_rows(q, d, 0, n_txt);
        }
    }
    return output(q, d, norm_out_, proj_out_, out_, lin);
}

}  // namespace locus::image
