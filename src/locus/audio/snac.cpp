#include "locus/audio/snac.hpp"

#include <cmath>
#include <stdexcept>

#include "locus/backend/cpu_ops.hpp"

namespace locus::audio {

namespace {

/** A channels-major activation buffer: `data` is [ch][len], row-major. */
struct Act {
    std::vector<float> data;
    std::uint32_t ch = 0;
    std::uint32_t len = 0;
};

}  // namespace

std::span<const float> SnacDecoder::t(const std::string& name) const {
    auto it = w_.find(name);
    if (it == w_.end()) {
        throw gguf::Error("snac: missing tensor " + name);
    }
    return it->second;
}

SnacDecoder SnacDecoder::load(const gguf::GgufFile& g) {
    const auto arch = g.get_string("general.architecture");
    if (!arch || *arch != "snac") {
        throw gguf::Error("snac: not a snac GGUF");
    }
    SnacDecoder d;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto v = g.get_uint(k);
        if (!v) throw gguf::Error(std::string("snac: missing ") +
                                  std::string(k));
        return static_cast<std::uint32_t>(*v);
    };
    d.latent_dim_ = u("snac.latent_dim");
    d.decoder_dim_ = u("snac.decoder_dim");
    d.codebook_dim_ = u("snac.codebook_dim");
    d.codebook_size_ = u("snac.codebook_size");
    d.n_codebooks_ = u("snac.n_codebooks");
    d.sample_rate_ = u("snac.sampling_rate");
    // Array elements carry their own type; a GGUF INT32 lands in
    // Value.i, not .u, so read whichever the element type populates.
    auto as_u32 = [](const gguf::Value& e) -> std::uint32_t {
        switch (e.type) {
            case gguf::ValueType::kInt8:
            case gguf::ValueType::kInt16:
            case gguf::ValueType::kInt32:
            case gguf::ValueType::kInt64:
                return static_cast<std::uint32_t>(e.i);
            default:
                return static_cast<std::uint32_t>(e.u);
        }
    };
    for (const auto* key : {"snac.decoder_rates", "snac.vq_strides"}) {
        const auto* a = g.get_array(key);
        if (!a) throw gguf::Error(std::string("snac: missing ") + key);
        auto& dst = (std::string(key) == "snac.decoder_rates")
                        ? d.rates_ : d.strides_;
        for (const auto& e : *a) {
            dst.push_back(as_u32(e));
        }
    }
    // Copy every F32 tensor by name so the mapping need not outlive us.
    for (const auto& ti : g.tensors()) {
        if (ti.type != gguf::TensorType::kF32) continue;
        auto bytes = g.tensor_data(ti);
        const auto* p = reinterpret_cast<const float*>(bytes.data());
        d.w_.emplace(ti.name,
                     std::vector<float>(p, p + ti.nelements()));
    }
    return d;
}

std::vector<float> SnacDecoder::decode(
    const model::SnacCodes& codes) const {
    const std::uint32_t F =
        static_cast<std::uint32_t>(codes.coarse.size());
    if (codes.mid.size() != std::size_t(F) * 2 ||
        codes.fine.size() != std::size_t(F) * 4) {
        throw std::invalid_argument(
            "snac decode: stream sizes must be F, 2F, 4F");
    }
    const std::vector<const std::vector<int>*> streams = {
        &codes.coarse, &codes.mid, &codes.fine};

    using backend::conv1d;
    using backend::conv1d_out_len;
    using backend::conv1d_transpose;
    using backend::conv1d_transpose_out_len;
    using backend::snake1d;

    // --- from_codes: sum of the stride-upsampled out-projections. ---
    const std::uint32_t T = F * strides_.front();  // aligned length.
    Act zq{std::vector<float>(std::size_t(latent_dim_) * T, 0.0f),
           latent_dim_, T};
    for (std::uint32_t i = 0; i < n_codebooks_; ++i) {
        const auto& s = *streams[i];
        const std::uint32_t li = static_cast<std::uint32_t>(s.size());
        const std::string q =
            "quantizer.quantizers." + std::to_string(i);
        const auto cb = t(q + ".codebook.weight");
        // Codebook lookup, transposed to [codebook_dim][li].
        std::vector<float> zp(std::size_t(codebook_dim_) * li);
        for (std::uint32_t j = 0; j < li; ++j) {
            const std::size_t base =
                std::size_t(s[j]) * codebook_dim_;
            for (std::uint32_t c = 0; c < codebook_dim_; ++c) {
                zp[std::size_t(c) * li + j] = cb[base + c];
            }
        }
        // 1x1 out-projection: codebook_dim -> latent_dim.
        std::vector<float> zo(std::size_t(latent_dim_) * li);
        conv1d(zp, codebook_dim_, li, t(q + ".out_proj.weight"),
               latent_dim_, 1, t(q + ".out_proj.bias"), 1, 1, 0, 1, zo);
        // repeat_interleave(stride) along time, accumulate into zq.
        const std::uint32_t st = strides_[i];
        for (std::uint32_t ch = 0; ch < latent_dim_; ++ch) {
            for (std::uint32_t j = 0; j < li; ++j) {
                const float v = zo[std::size_t(ch) * li + j];
                float* dst = &zq.data[std::size_t(ch) * T + j * st];
                for (std::uint32_t r = 0; r < st; ++r) dst[r] += v;
            }
        }
    }

    // A conv step that sizes its own output from the shared out_len.
    auto conv = [&](const Act& in, const std::string& name,
                    std::uint32_t out_ch, std::uint32_t k,
                    std::uint32_t dil, std::uint32_t pad,
                    std::uint32_t groups) {
        const std::uint32_t ol = conv1d_out_len(in.len, k, 1, dil, pad);
        Act o{std::vector<float>(std::size_t(out_ch) * ol), out_ch, ol};
        conv1d(in.data, in.ch, in.len, t(name + ".weight"), out_ch, k,
               t(name + ".bias"), 1, dil, pad, groups, o.data);
        return o;
    };
    auto snake = [&](Act& a, const std::string& name) {
        snake1d(a.data, a.ch, a.len, t(name));
    };

    // Decoder in-convs: depthwise k7 then 1x1 to the decoder width.
    Act x = conv(zq, "decoder.model.0", latent_dim_, 7, 1, 3,
                 latent_dim_);
    x = conv(x, "decoder.model.1", decoder_dim_, 1, 1, 0, 1);

    std::uint32_t ch = decoder_dim_;
    const int dils[3] = {1, 3, 9};
    for (std::size_t b = 0; b < rates_.size(); ++b) {
        const std::string p =
            "decoder.model." + std::to_string(2 + b) + ".block";
        const std::uint32_t rate = rates_[b];
        const std::uint32_t out_ch = ch / 2;
        // Upsample: snake, then a transposed conv (k=2*rate, stride
        // rate, pad rate/2) that scales length by exactly rate.
        snake(x, p + ".0.alpha");
        const std::uint32_t ol = conv1d_transpose_out_len(
            x.len, 2 * rate, rate, 1, rate / 2);
        Act up{std::vector<float>(std::size_t(out_ch) * ol), out_ch, ol};
        conv1d_transpose(x.data, x.ch, x.len, t(p + ".1.weight"), out_ch,
                         2 * rate, t(p + ".1.bias"), rate, 1, rate / 2, 1,
                         up.data);
        x = std::move(up);
        ch = out_ch;
        // Three dilated residual units (block.3/4/5): each a depthwise
        // k7 conv and a 1x1 conv around snakes, added back to x (same
        // length, so no crop).
        for (int uidx = 0; uidx < 3; ++uidx) {
            const std::string rp =
                p + "." + std::to_string(3 + uidx) + ".block";
            Act y = x;
            snake(y, rp + ".0.alpha");
            y = conv(y, rp + ".1", ch, 7, dils[uidx], 3 * dils[uidx], ch);
            snake(y, rp + ".2.alpha");
            y = conv(y, rp + ".3", ch, 1, 1, 0, 1);
            for (std::size_t n = 0; n < x.data.size(); ++n) {
                x.data[n] += y.data[n];
            }
        }
    }

    // Tail: snake, 1-channel k7 conv, tanh.
    snake(x, "decoder.model.6.alpha");
    x = conv(x, "decoder.model.7", 1, 7, 1, 3, 1);
    for (auto& v : x.data) v = std::tanh(v);
    return std::move(x.data);
}

}  // namespace locus::audio
