#include <cstddef>

#include "locus/backend/vulkan_seq.hpp"
#include "locus/image/janus_vision.hpp"

namespace locus::image {

namespace {

using Seq = backend::VulkanSeq;
using Rows = Seq::Rows;

/** The scratch slots of one tower call. */
enum : Seq::Slot {
    kX,       ///< residual rows [n][w]
    kH,       ///< LayerNorm output [n][w]
    kQkv,     ///< q | k | v per token [n][3 w]
    kAtt,     ///< attention output [n][w]
    kO,       ///< projection output [n][w]
    kUp,      ///< MLP hidden [n][ffn]
    kKh,      ///< attention planes and scores
    kVt,
    kScores,
    kPatch,   ///< the image's patches, one row each [n][3 p p]
    kPos,     ///< position embeddings [n][w]
};

}  // namespace

std::vector<float> JanusVision::features_vulkan(
    std::span<const float> chw) const {
    const std::uint32_t n = tokens(), w = width_, hd = w / heads_;
    // The stride-patch conv is a matmul: token t's row holds its patch
    // as [channel][y][x], the conv weight's column order.
    const std::uint32_t pp = patch_ * patch_, pk = 3 * pp;
    std::vector<float> patches(std::size_t{n} * pk);
    for (std::uint32_t gy = 0; gy < grid_; ++gy) {
        for (std::uint32_t gx = 0; gx < grid_; ++gx) {
            float* row = &patches[std::size_t{gy * grid_ + gx} * pk];
            for (std::uint32_t c = 0; c < 3; ++c) {
                for (std::uint32_t y = 0; y < patch_; ++y) {
                    const float* src =
                        &chw[(std::size_t{c} * image_ + gy * patch_ + y) *
                                 image_ +
                             gx * patch_];
                    std::copy(src, src + patch_, row + c * pp + y * patch_);
                }
            }
        }
    }
    Seq q;
    const std::size_t nw = std::size_t{n} * w;
    for (Seq::Slot s : {kX, kH, kAtt, kO, kPos}) q.reserve(s, nw);
    q.reserve(kQkv, 3 * nw);
    q.reserve(kUp, std::size_t{n} * ffn_);
    q.reserve(kPatch, patches.size());
    q.write(kPatch, 0, patches);
    q.write(kPos, 0, pos_);
    const backend::Mat pw{gguf::TensorType::kF32,
                          reinterpret_cast<const std::byte*>(patch_w_.data()),
                          w, pk};
    q.linear(pw, patch_b_, {kPatch, 0, pk}, {kX, 0, w}, n);
    q.add({kX, 0, w}, {kPos, 0, w}, n, w);
    for (const Block& k : blocks_) {
        // x += proj(attention(LN1(x))); qkv[t] holds q | k | v, each
        // heads x hd, head-major.
        q.layer_norm({kX, 0, w}, {kH, 0, w}, n, w, k.ln1_w, k.ln1_b, eps_);
        q.linear(k.qkv, k.qkv_b, {kH, 0, w}, {kQkv, 0, 3 * w}, n);
        q.attention({kQkv, 0, 3 * w}, {kQkv, w, 3 * w}, {kQkv, 2 * w, 3 * w},
                    {kAtt, 0, w}, n, heads_, hd, kKh, kVt, kScores);
        q.linear(k.out, k.out_b, {kAtt, 0, w}, {kO, 0, w}, n);
        q.add({kX, 0, w}, {kO, 0, w}, n, w);
        // x += fc2(GELU(fc1(LN2(x))))
        q.layer_norm({kX, 0, w}, {kH, 0, w}, n, w, k.ln2_w, k.ln2_b, eps_);
        q.linear(k.up, k.up_b, {kH, 0, w}, {kUp, 0, ffn_}, n);
        q.gelu_erf({kUp, 0, ffn_}, n, ffn_);
        q.linear(k.down, k.down_b, {kUp, 0, ffn_}, {kO, 0, w}, n);
        q.add({kX, 0, w}, {kO, 0, w}, n, w);
    }
    q.layer_norm({kX, 0, w}, {kH, 0, w}, n, w, post_w_, post_b_, eps_);
    std::vector<float> x(nw);
    q.read(kH, 0, x);
    return x;
}

}  // namespace locus::image
