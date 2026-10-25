#include "locus/image/janus.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "locus/backend/cpu_ops.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/sampling.hpp"

namespace locus::image {

namespace {

/** Exact GELU (torch.nn.GELU(), approximate="none"). */
void gelu(std::span<float> x) {
    for (float& v : x) {
        v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f));
    }
}

}  // namespace

std::span<const float> JanusImageGen::t(const std::string& name,
                                        std::size_t n) const {
    auto it = w_.find(name);
    if (it == w_.end()) {
        throw gguf::Error("janus: missing tensor " + name);
    }
    if (it->second.size() != n) {
        throw gguf::Error("janus: tensor " + name + " has " +
                          std::to_string(it->second.size()) +
                          " values, expected " + std::to_string(n));
    }
    return it->second;
}

JanusImageGen JanusImageGen::load(const gguf::GgufFile& g,
                                  std::uint32_t n_embd) {
    JanusImageGen j;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto v = g.get_uint(k);
        if (!v) throw gguf::Error("janus: missing " + std::string(k));
        return static_cast<std::uint32_t>(*v);
    };
    j.n_embd_ = n_embd;
    j.codes_ = u("janus.image_tokens");
    j.grid_ = u("janus.image_grid");
    j.begin_of_image_ = static_cast<tok::TokenId>(u("janus.begin_of_image"));
    j.pad_ = static_cast<tok::TokenId>(u("janus.pad"));
    const gguf::Value* cfg = g.find("janus.cfg_scale");
    j.cfg_scale_ = cfg != nullptr && cfg->type == gguf::ValueType::kFloat32
                       ? static_cast<float>(cfg->f)
                       : 5.0f;
    if (j.codes_ == 0 || j.grid_ == 0 || n_embd == 0) {
        throw gguf::Error("janus: zero size in the metadata");
    }
    for (const auto& ti : g.tensors()) {
        if (ti.name.rfind("janus.", 0) != 0) continue;
        if (ti.type != gguf::TensorType::kF32) {
            throw gguf::Error("janus: " + ti.name + " must be F32");
        }
        auto bytes = g.tensor_data(ti);
        const auto* p = reinterpret_cast<const float*>(bytes.data());
        j.w_.emplace(ti.name, std::vector<float>(p, p + ti.nelements()));
    }
    // Every tensor at its expected size now, so a bad file fails at
    // load, not mid-generation.
    const std::size_t e = n_embd, c = j.codes_;
    j.t("janus.gen_embed.weight", c * 8);
    j.t("janus.gen_aligner.0.weight", e * 8);
    j.t("janus.gen_aligner.0.bias", e);
    j.t("janus.gen_aligner.2.weight", e * e);
    j.t("janus.gen_aligner.2.bias", e);
    j.t("janus.gen_head.proj.weight", e * e);
    j.t("janus.gen_head.proj.bias", e);
    j.t("janus.gen_head.out.weight", c * e);
    j.t("janus.gen_head.out.bias", c);
    return j;
}

void JanusImageGen::image_embed(std::int32_t id,
                                std::span<float> out) const {
    if (id < 0 || static_cast<std::uint32_t>(id) >= codes_) {
        throw std::invalid_argument("janus: image code out of range");
    }
    const std::size_t e = n_embd_;
    const auto emb = t("janus.gen_embed.weight", std::size_t{codes_} * 8)
                         .subspan(std::size_t(id) * 8, 8);
    std::vector<float> h(e);
    backend::matvec({gguf::TensorType::kF32,
                     reinterpret_cast<const std::byte*>(
                         t("janus.gen_aligner.0.weight", e * 8).data()),
                     n_embd_, 8},
                    emb, h);
    const auto b0 = t("janus.gen_aligner.0.bias", e);
    for (std::size_t i = 0; i < e; ++i) h[i] += b0[i];
    gelu(h);
    backend::matvec({gguf::TensorType::kF32,
                     reinterpret_cast<const std::byte*>(
                         t("janus.gen_aligner.2.weight", e * e).data()),
                     n_embd_, n_embd_},
                    h, out);
    const auto b2 = t("janus.gen_aligner.2.bias", e);
    for (std::size_t i = 0; i < e; ++i) out[i] += b2[i];
}

void JanusImageGen::image_logits(std::span<const float> h,
                                 std::span<float> out) const {
    const std::size_t e = n_embd_;
    std::vector<float> p(e);
    backend::matvec({gguf::TensorType::kF32,
                     reinterpret_cast<const std::byte*>(
                         t("janus.gen_head.proj.weight", e * e).data()),
                     n_embd_, n_embd_},
                    h, p);
    const auto pb = t("janus.gen_head.proj.bias", e);
    for (std::size_t i = 0; i < e; ++i) p[i] += pb[i];
    gelu(p);
    backend::matvec({gguf::TensorType::kF32,
                     reinterpret_cast<const std::byte*>(
                         t("janus.gen_head.out.weight",
                           std::size_t{codes_} * e)
                             .data()),
                     codes_, n_embd_},
                    p, out);
    const auto ob = t("janus.gen_head.out.bias", codes_);
    for (std::uint32_t i = 0; i < codes_; ++i) out[i] += ob[i];
}

std::vector<tok::TokenId> JanusImageGen::unconditional(
    std::span<const tok::TokenId> prompt) const {
    std::vector<tok::TokenId> u(prompt.begin(), prompt.end());
    for (std::size_t i = 1; i + 1 < u.size(); ++i) u[i] = pad_;
    return u;
}

std::vector<std::int32_t> JanusImageGen::generate(
    const model::TransformerModel& m, std::span<const tok::TokenId> prompt,
    const Params& p, const std::function<bool()>& alive) const {
    if (prompt.size() < 2) {
        throw std::invalid_argument("janus: prompt too short");
    }
    const auto& hp = m.hparams();
    const std::uint32_t total =
        static_cast<std::uint32_t>(prompt.size()) + image_tokens();
    auto cache = m.make_cache();
    auto ws = m.make_workspace();
    std::vector<float> logits(hp.n_vocab);
    // The backbone's text head is not needed: one row keeps it cheap,
    // and ws.xb still holds the final normed hidden state.
    const model::LogitRows one{0, 1};
    const std::vector<tok::TokenId> uncond = unconditional(prompt);
    kv::PagedKvCache::Seq seq[2];
    const std::span<const tok::TokenId> ids[2] = {prompt, uncond};
    std::vector<float> img[2] = {std::vector<float>(codes_),
                                 std::vector<float>(codes_)};
    for (int s = 0; s < 2; ++s) {
        if (!cache.ensure_capacity(seq[s], total)) {
            throw std::runtime_error("janus: KV cache too small");
        }
        for (const tok::TokenId id : ids[s]) {
            m.forward_layers(id, {}, 0, hp.n_layers, cache, seq[s], ws,
                             logits, one);
        }
        image_logits(std::span<const float>(ws.xb).first(hp.n_embd),
                     img[s]);
    }

    model::SamplingParams sp;
    sp.temperature = p.temperature;
    sp.top_p = 1.0f;
    sp.repeat_penalty = 1.0f;
    std::mt19937_64 rng(p.seed);
    std::vector<float> guided(codes_), emb(hp.n_embd);
    std::vector<std::int32_t> out;
    out.reserve(image_tokens());
    for (std::uint32_t i = 0; i < image_tokens(); ++i) {
        if (alive && !alive()) break;
        for (std::uint32_t k = 0; k < codes_; ++k) {
            guided[k] = img[1][k] + p.cfg * (img[0][k] - img[1][k]);
        }
        const auto code = static_cast<std::int32_t>(
            model::sample(guided, sp, {}, rng));
        out.push_back(code);
        if (i + 1 == image_tokens()) break;
        image_embed(code, emb);
        for (int s = 0; s < 2; ++s) {
            m.forward_layers(0, emb, 0, hp.n_layers, cache, seq[s], ws,
                             logits, one);
            image_logits(std::span<const float>(ws.xb).first(hp.n_embd),
                         img[s]);
        }
    }
    for (auto& s : seq) cache.release(s);
    return out;
}

}  // namespace locus::image
