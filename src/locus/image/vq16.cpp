#include "locus/image/vq16.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "locus/backend/cpu_ops.hpp"

namespace locus::image {

namespace {

/** A channels-major image activation: `data` is [ch][h][w]. */
struct Act {
    std::vector<float> data;
    std::uint32_t ch = 0;
    std::uint32_t h = 0;
    std::uint32_t w = 0;
    std::uint32_t plane() const { return h * w; }
};

}  // namespace

std::span<const float> Vq16Decoder::t(const std::string& name,
                                      std::size_t n) const {
    auto it = w_.find(name);
    if (it == w_.end()) {
        throw gguf::Error("vq16: missing tensor " + name);
    }
    // The kernels check sizes only with assert(), so a short tensor
    // from the GGUF would be read out of bounds in Release builds.
    if (it->second.size() != n) {
        throw gguf::Error("vq16: tensor " + name + " has " +
                          std::to_string(it->second.size()) +
                          " values, expected " + std::to_string(n));
    }
    return it->second;
}

Vq16Decoder Vq16Decoder::load(const gguf::GgufFile& g) {
    const auto arch = g.get_string("general.architecture");
    if (!arch || *arch != "vq16") {
        throw gguf::Error("vq16: not a vq16 GGUF");
    }
    Vq16Decoder d;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto v = g.get_uint(k);
        if (!v) {
            throw gguf::Error("vq16: missing " + std::string(k));
        }
        return static_cast<std::uint32_t>(*v);
    };
    auto u32_array = [&](std::string_view k) {
        const auto* a = g.get_array(k);
        if (!a) throw gguf::Error("vq16: missing " + std::string(k));
        std::vector<std::uint32_t> out;
        for (const auto& e : *a) {
            out.push_back(static_cast<std::uint32_t>(
                e.type == gguf::ValueType::kInt32 ||
                        e.type == gguf::ValueType::kInt64
                    ? e.i
                    : static_cast<std::int64_t>(e.u)));
        }
        return out;
    };
    d.codebook_size_ = u("vq16.codebook_size");
    d.embed_dim_ = u("vq16.embed_dim");
    d.z_channels_ = u("vq16.z_channels");
    d.ch_ = u("vq16.ch");
    d.num_res_blocks_ = u("vq16.num_res_blocks");
    d.norm_groups_ = u("vq16.norm_groups");
    d.ch_mult_ = u32_array("vq16.ch_mult");
    const auto attn = u32_array("vq16.attn_levels");
    const gguf::Value* eps = g.find("vq16.norm_eps");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32) {
        throw gguf::Error("vq16: missing f32 vq16.norm_eps");
    }
    d.norm_eps_ = static_cast<float>(eps->f);
    const auto l2 = g.get_bool("vq16.codebook_l2_norm");
    // The graph this class runs: an l2-normalised codebook, attention
    // only at the lowest resolution, every width a multiple of the norm
    // groups. Refuse anything else rather than decode it wrongly.
    if (!l2 || !*l2) {
        throw gguf::Error("vq16: needs codebook_l2_norm = true");
    }
    if (d.ch_mult_.empty() || d.ch_mult_.size() > 8 ||
        attn != std::vector<std::uint32_t>{
                    static_cast<std::uint32_t>(d.ch_mult_.size() - 1)}) {
        throw gguf::Error("vq16: needs attention at the last level only");
    }
    if (d.codebook_size_ == 0 || d.embed_dim_ == 0 || d.z_channels_ == 0 ||
        d.ch_ == 0 || d.norm_groups_ == 0) {
        throw gguf::Error("vq16: zero size in the metadata");
    }
    for (auto m : d.ch_mult_) {
        if (m == 0 || (d.ch_ * m) % d.norm_groups_ != 0) {
            throw gguf::Error("vq16: width not divisible by norm groups");
        }
    }
    for (const auto& ti : g.tensors()) {
        if (ti.type != gguf::TensorType::kF32) continue;
        auto bytes = g.tensor_data(ti);
        const auto* p = reinterpret_cast<const float*>(bytes.data());
        d.w_.emplace(ti.name, std::vector<float>(p, p + ti.nelements()));
    }
    return d;
}

std::vector<float> Vq16Decoder::decode(std::span<const std::int32_t> codes,
                                       std::uint32_t grid_h,
                                       std::uint32_t grid_w) const {
    using backend::conv2d_mt;
    if (codes.size() != std::size_t{grid_h} * grid_w || codes.empty()) {
        throw std::invalid_argument("vq16 decode: need grid_h * grid_w "
                                    "codes");
    }
    const std::uint32_t groups = norm_groups_;
    const float eps = norm_eps_;

    // A conv with bias; k is 1 or 3 (padding k / 2), stride 1.
    auto conv = [&](const Act& in, const std::string& name,
                    std::uint32_t out_ch, std::uint32_t k) {
        Act o{std::vector<float>(std::size_t{out_ch} * in.plane()), out_ch,
              in.h, in.w};
        conv2d_mt(in.data, in.ch, in.h, in.w,
                  t(name + ".weight", std::size_t{out_ch} * in.ch * k * k),
                  out_ch, k, k, t(name + ".bias", out_ch), 1, k / 2, 1,
                  o.data);
        return o;
    };
    auto norm = [&](Act& a, const std::string& name) {
        backend::group_norm(a.data, a.ch, a.plane(), groups,
                            t(name + ".weight", a.ch),
                            t(name + ".bias", a.ch), eps);
    };
    auto add = [](Act& dst, const Act& src) {
        for (std::size_t i = 0; i < dst.data.size(); ++i) {
            dst.data[i] += src.data[i];
        }
    };
    // ResnetBlock: two GN / SiLU / conv3 steps, plus x (through a 1x1
    // nin_shortcut when the width changes).
    auto res = [&](const Act& x, const std::string& p,
                   std::uint32_t out_ch) {
        Act h = x;
        norm(h, p + ".norm1");
        backend::silu(h.data);
        h = conv(h, p + ".conv1", out_ch, 3);
        norm(h, p + ".norm2");
        backend::silu(h.data);
        h = conv(h, p + ".conv2", out_ch, 3);
        if (x.ch != out_ch) {
            add(h, conv(x, p + ".nin_shortcut", out_ch, 1));
        } else {
            add(h, x);
        }
        return h;
    };
    // AttnBlock: single head over the h*w positions, scale c^-0.5.
    auto attention = [&](const Act& x, const std::string& p) {
        Act h = x;
        norm(h, p + ".norm");
        const Act q = conv(h, p + ".q", x.ch, 1);
        const Act k = conv(h, p + ".k", x.ch, 1);
        const Act v = conv(h, p + ".v", x.ch, 1);
        const std::uint32_t n = x.plane(), c = x.ch;
        const float scale = 1.0f / std::sqrt(static_cast<float>(c));
        Act o{std::vector<float>(std::size_t{c} * n, 0.0f), c, x.h, x.w};
        std::vector<float> wrow(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            // Scores of query i over every key j, then softmax over j.
            float mx = -INFINITY;
            for (std::uint32_t j = 0; j < n; ++j) {
                float s = 0.0f;
                for (std::uint32_t cc = 0; cc < c; ++cc) {
                    s += q.data[std::size_t{cc} * n + i] *
                         k.data[std::size_t{cc} * n + j];
                }
                wrow[j] = s * scale;
                mx = std::max(mx, wrow[j]);
            }
            double sum = 0.0;
            for (std::uint32_t j = 0; j < n; ++j) {
                wrow[j] = std::exp(wrow[j] - mx);
                sum += wrow[j];
            }
            const float inv = static_cast<float>(1.0 / sum);
            for (std::uint32_t cc = 0; cc < c; ++cc) {
                const float* vr = &v.data[std::size_t{cc} * n];
                float acc = 0.0f;
                for (std::uint32_t j = 0; j < n; ++j) {
                    acc += vr[j] * wrow[j];
                }
                o.data[std::size_t{cc} * n + i] = acc * inv;
            }
        }
        Act out = conv(o, p + ".proj_out", c, 1);
        add(out, x);
        return out;
    };

    // Lookup: z[c][y][x] = normalize(E[code])[c], then post_quant_conv.
    const auto cb =
        t("quantize.embedding.weight",
          std::size_t{codebook_size_} * embed_dim_);
    Act z{std::vector<float>(std::size_t{embed_dim_} * codes.size()),
          embed_dim_, grid_h, grid_w};
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const std::int32_t id = codes[i];
        if (id < 0 || static_cast<std::uint32_t>(id) >= codebook_size_) {
            throw std::invalid_argument(
                "vq16 decode: code " + std::to_string(id) +
                " outside the codebook (size " +
                std::to_string(codebook_size_) + ")");
        }
        const float* e = &cb[std::size_t(id) * embed_dim_];
        double nrm = 0.0;
        for (std::uint32_t c = 0; c < embed_dim_; ++c) {
            nrm += static_cast<double>(e[c]) * e[c];
        }
        const float inv =
            static_cast<float>(1.0 / std::max(std::sqrt(nrm), 1e-12));
        for (std::uint32_t c = 0; c < embed_dim_; ++c) {
            z.data[std::size_t{c} * codes.size() + i] = e[c] * inv;
        }
    }
    Act h = conv(z, "post_quant_conv", z_channels_, 1);

    const std::size_t levels = ch_mult_.size();
    std::uint32_t block_in = ch_ * ch_mult_.back();
    h = conv(h, "decoder.conv_in", block_in, 3);
    h = res(h, "decoder.mid.0", block_in);
    h = attention(h, "decoder.mid.1");
    h = res(h, "decoder.mid.2", block_in);
    for (std::size_t i = 0; i < levels; ++i) {
        const std::size_t level = levels - 1 - i;  // widest first
        const std::uint32_t block_out = ch_ * ch_mult_[level];
        const std::string p = "decoder.conv_blocks." + std::to_string(i);
        for (std::uint32_t b = 0; b <= num_res_blocks_; ++b) {
            h = res(h, p + ".res." + std::to_string(b), block_out);
            if (level == levels - 1) {
                h = attention(h, p + ".attn." + std::to_string(b));
            }
        }
        if (i != levels - 1) {
            Act up{std::vector<float>(h.data.size() * 4), h.ch, h.h * 2,
                   h.w * 2};
            backend::upsample_nearest2x(h.data, h.ch, h.h, h.w, up.data);
            h = conv(up, p + ".upsample.conv", h.ch, 3);
        }
    }
    norm(h, "decoder.norm_out");
    backend::silu(h.data);
    h = conv(h, "decoder.conv_out", 3, 3);
    return std::move(h.data);
}

}  // namespace locus::image
