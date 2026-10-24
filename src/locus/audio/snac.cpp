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

std::span<const float> SnacDecoder::t(const std::string& name,
                                      std::size_t n) const {
    auto it = w_.find(name);
    if (it == w_.end()) {
        throw gguf::Error("snac: missing tensor " + name);
    }
    // The conv kernels check sizes only with assert(), which Release
    // builds compile out, so a short tensor from the GGUF would be read
    // out of bounds. Check the size the graph needs here instead.
    if (it->second.size() != n) {
        throw gguf::Error("snac: tensor " + name + " has " +
                          std::to_string(it->second.size()) +
                          " values, expected " + std::to_string(n));
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
    // decode() carries exactly the three SnacCodes streams (F, 2F and 4F
    // codes) and aligns them at T = 4F, which holds only for strides
    // [4, 2, 1]. Each upsample halves the width and its transposed conv
    // uses pad rate/2 with no output_padding, which needs an even rate.
    if (d.n_codebooks_ != 3 ||
        d.strides_ != std::vector<std::uint32_t>{4, 2, 1}) {
        throw gguf::Error("snac: need 3 codebooks with vq_strides [4,2,1]");
    }
    if (d.rates_.empty() || d.rates_.size() > 16 ||
        (d.decoder_dim_ >> d.rates_.size()) == 0) {
        throw gguf::Error("snac: decoder_dim too small for its rates");
    }
    for (auto r : d.rates_) {
        if (r == 0 || r % 2 != 0) {
            throw gguf::Error("snac: decoder rates must be even");
        }
    }
    if (d.latent_dim_ == 0 || d.codebook_dim_ == 0 ||
        d.codebook_size_ == 0) {
        throw gguf::Error("snac: zero latent/codebook size");
    }
    // The tensor names follow SNAC's depthwise + noise, attention-free
    // decoder layout; refuse a GGUF that records another one (keys
    // written by scripts/convert-snac.py; older files lack them). A key
    // of another type is an error: get_bool would read it as absent.
    const auto flag = [&g](const char* key) {
        const gguf::Value* v = g.find(key);
        if (v != nullptr && v->type != gguf::ValueType::kBool) {
            throw gguf::Error(std::string("snac: ") + key +
                              " must be a bool");
        }
        return g.get_bool(key);
    };
    const auto depthwise = flag("snac.depthwise");
    const auto noise = flag("snac.noise");
    const auto attention = flag("snac.attention");
    if ((depthwise && !*depthwise) || (noise && !*noise) ||
        (attention && *attention)) {
        throw gguf::Error("snac: unsupported decoder layout (needs "
                          "depthwise, noise, no attention)");
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

    // The threaded kernels: bit-identical to the scalar conv1d /
    // conv1d_transpose / snake1d, about N cores times faster (i#173).
    const auto conv1d = backend::conv1d_mt;
    const auto conv1d_transpose = backend::conv1d_transpose_mt;
    const auto snake1d = [](std::span<float> x, std::uint32_t ch,
                            std::uint32_t len,
                            std::span<const float> alpha) {
        backend::snake1d_mt(x, ch, len, alpha);
    };
    using backend::conv1d_out_len;
    using backend::conv1d_transpose_out_len;

    // --- from_codes: sum of the stride-upsampled out-projections. ---
    const std::uint32_t T = F * strides_.front();  // aligned length.
    Act zq{std::vector<float>(std::size_t(latent_dim_) * T, 0.0f),
           latent_dim_, T};
    for (std::uint32_t i = 0; i < n_codebooks_; ++i) {
        const auto& s = *streams[i];
        const std::uint32_t li = static_cast<std::uint32_t>(s.size());
        const std::string q =
            "quantizer.quantizers." + std::to_string(i);
        const auto cb = t(q + ".codebook.weight",
                          std::size_t(codebook_size_) * codebook_dim_);
        // Codebook lookup, transposed to [codebook_dim][li].
        std::vector<float> zp(std::size_t(codebook_dim_) * li);
        for (std::uint32_t j = 0; j < li; ++j) {
            if (s[j] < 0 ||
                static_cast<std::uint32_t>(s[j]) >= codebook_size_) {
                throw std::invalid_argument(
                    "snac decode: code " + std::to_string(s[j]) +
                    " outside the codebook (size " +
                    std::to_string(codebook_size_) + ")");
            }
            const std::size_t base =
                std::size_t(s[j]) * codebook_dim_;
            for (std::uint32_t c = 0; c < codebook_dim_; ++c) {
                zp[std::size_t(c) * li + j] = cb[base + c];
            }
        }
        // 1x1 out-projection: codebook_dim -> latent_dim.
        std::vector<float> zo(std::size_t(latent_dim_) * li);
        conv1d(zp, codebook_dim_, li,
               t(q + ".out_proj.weight",
                 std::size_t(latent_dim_) * codebook_dim_),
               latent_dim_, 1, t(q + ".out_proj.bias", latent_dim_), 1, 1,
               0, 1, zo);
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
        conv1d(in.data, in.ch, in.len,
               t(name + ".weight",
                 std::size_t(out_ch) * (in.ch / groups) * k),
               out_ch, k, t(name + ".bias", out_ch), 1, dil, pad, groups,
               o.data);
        return o;
    };
    auto snake = [&](Act& a, const std::string& name) {
        snake1d(a.data, a.ch, a.len, t(name, a.ch));
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
        conv1d_transpose(
            x.data, x.ch, x.len,
            t(p + ".1.weight", std::size_t(x.ch) * out_ch * 2 * rate),
            out_ch, 2 * rate, t(p + ".1.bias", out_ch), rate, 1, rate / 2,
            1, up.data);
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

    // Tail: snake, 1-channel k7 conv, tanh. It sits right after the
    // last DecoderBlock in the Sequential, so its index follows the
    // number of upsample rates (model.6/.7 for the 24 kHz model's 4).
    const std::string tail = std::to_string(2 + rates_.size());
    const std::string tail_conv = std::to_string(3 + rates_.size());
    snake(x, "decoder.model." + tail + ".alpha");
    x = conv(x, "decoder.model." + tail_conv, 1, 7, 1, 3, 1);
    for (auto& v : x.data) v = std::tanh(v);
    return std::move(x.data);
}

}  // namespace locus::audio
