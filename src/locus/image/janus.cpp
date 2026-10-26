#include "locus/image/janus.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>

#include "locus/backend/cpu_ops.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/sampling.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

/** Exact GELU (torch.nn.GELU(), approximate="none"). */
void gelu(std::span<float> x) {
    for (float& v : x) {
        v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752440f));
    }
}

/** Python's str.isspace() set (what str.strip() removes). */
bool py_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) ||
           c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

/** Decodes the UTF-8 code point at s[at] (bad bytes decode as 0xFFFD,
 * which is not space); n is set to its byte length. */
char32_t utf8_at(std::string_view s, std::size_t at, std::size_t& n) {
    const auto b0 = static_cast<unsigned char>(s[at]);
    n = b0 < 0x80 ? 1 : b0 >= 0xF0 ? 4 : b0 >= 0xE0 ? 3 : b0 >= 0xC0 ? 2 : 1;
    if (n == 1) return b0 < 0x80 ? b0 : 0xFFFD;
    if (at + n > s.size()) {
        n = 1;
        return 0xFFFD;
    }
    char32_t c = b0 & (0x7F >> n);
    for (std::size_t k = 1; k < n; ++k) {
        c = (c << 6) | (static_cast<unsigned char>(s[at + k]) & 0x3F);
    }
    return c;
}

}  // namespace

std::string_view python_strip(std::string_view s) {
    std::size_t n = 0;
    while (!s.empty() && py_space(utf8_at(s, 0, n))) s.remove_prefix(n);
    while (!s.empty()) {
        std::size_t at = s.size() - 1;  // back to the last lead byte
        while (at > 0 && (static_cast<unsigned char>(s[at]) & 0xC0) == 0x80) {
            --at;
        }
        if (!py_space(utf8_at(s, at, n)) || at + n != s.size()) break;
        s.remove_suffix(n);
    }
    return s;
}

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
        j.w_.emplace(ti.name, std::span<const float>(p, ti.nelements()));
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

namespace {

/**
 * backend::matvec (the scalar f32 dot) with the rows split across the
 * thread pool. Each row is still computed whole by one thread with the
 * same code, so the result is bit-identical to the single call (i#246:
 * the image heads were single-threaded).
 */
void matvec_pool(const backend::Mat& w, std::span<const float> x,
                 std::span<float> out) {
    const std::uint32_t threads =
        std::max(1u, std::thread::hardware_concurrency());
    const std::uint32_t slices =
        std::max(1u, std::min(w.rows / 256, 2 * threads));
    const std::uint32_t per = (w.rows + slices - 1) / slices;
    sys::ThreadPool::instance().parallel_for(slices, [&](std::size_t s) {
        const std::uint32_t r0 = static_cast<std::uint32_t>(s) * per;
        if (r0 >= w.rows) return;
        const std::uint32_t nr = std::min(per, w.rows - r0);
        backend::matvec(backend::mat_rows(w, r0, nr), x,
                        out.subspan(r0, nr));
    });
}

/** matvec_pool, or one whole-matrix gpu->matvec call when gpu is set
 * (i#247: the heads on the GPU next to the fused forward). */
void head_matvec(const backend::Mat& w, std::span<const float> x,
                 std::span<float> out, const backend::Ops* gpu) {
    if (gpu != nullptr) {
        gpu->matvec(w, x, out);
        return;
    }
    matvec_pool(w, x, out);
}

}  // namespace

void JanusImageGen::image_embed(std::int32_t id, std::span<float> out,
                                const backend::Ops* gpu) const {
    if (id < 0 || static_cast<std::uint32_t>(id) >= codes_) {
        throw std::invalid_argument("janus: image code out of range");
    }
    const std::size_t e = n_embd_;
    const auto emb = t("janus.gen_embed.weight", std::size_t{codes_} * 8)
                         .subspan(std::size_t(id) * 8, 8);
    std::vector<float> h(e);
    head_matvec({gguf::TensorType::kF32,
                 reinterpret_cast<const std::byte*>(
                     t("janus.gen_aligner.0.weight", e * 8).data()),
                 n_embd_, 8},
                emb, h, gpu);
    const auto b0 = t("janus.gen_aligner.0.bias", e);
    for (std::size_t i = 0; i < e; ++i) h[i] += b0[i];
    gelu(h);
    head_matvec({gguf::TensorType::kF32,
                 reinterpret_cast<const std::byte*>(
                     t("janus.gen_aligner.2.weight", e * e).data()),
                 n_embd_, n_embd_},
                h, out, gpu);
    const auto b2 = t("janus.gen_aligner.2.bias", e);
    for (std::size_t i = 0; i < e; ++i) out[i] += b2[i];
}

void JanusImageGen::image_logits(std::span<const float> h,
                                 std::span<float> out,
                                 const backend::Ops* gpu) const {
    const std::size_t e = n_embd_;
    std::vector<float> p(e);
    head_matvec({gguf::TensorType::kF32,
                 reinterpret_cast<const std::byte*>(
                     t("janus.gen_head.proj.weight", e * e).data()),
                 n_embd_, n_embd_},
                h, p, gpu);
    const auto pb = t("janus.gen_head.proj.bias", e);
    for (std::size_t i = 0; i < e; ++i) p[i] += pb[i];
    gelu(p);
    head_matvec({gguf::TensorType::kF32,
                 reinterpret_cast<const std::byte*>(
                     t("janus.gen_head.out.weight", std::size_t{codes_} * e)
                         .data()),
                 codes_, n_embd_},
                p, out, gpu);
    const auto ob = t("janus.gen_head.out.bias", codes_);
    for (std::uint32_t i = 0; i < codes_; ++i) out[i] += ob[i];
}

std::vector<tok::TokenId> JanusImageGen::prompt_ids(
    const tok::Tokenizer& tok, std::string_view text) const {
    // The reference strips the message (Python str.strip()) before the
    // wrap; the roles are Janus-Pro's own special tokens (the README's
    // Janus-Pro example). The tag is appended as its id: the
    // reference's text ends in "<begin_of_image>", which its tokenizer
    // maps to this one id.
    auto ids = tok.encode("<|User|>: " + std::string(python_strip(text)) +
                              "\n\n<|Assistant|>:",
                          true);
    ids.push_back(begin_of_image_);
    return ids;
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
    std::vector<float> logits(hp.n_vocab), normed(hp.n_embd);
    // The backbone's text head is not needed: one row keeps it cheap.
    // forward_embedding returns the final normed hidden state (on the
    // fused GPU forward when the backend has it, i#247).
    const model::LogitRows one{0, 1};
    // On vulkan the heads run on the GPU as well (i#247); the
    // backend's matvec caches their weights by mapped address.
    const backend::Ops* gpu = m.active_backend().name == "vulkan"
                                  ? &m.active_backend().ops
                                  : nullptr;
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
            m.forward_embedding(id, {}, cache, seq[s], ws, logits, one,
                                normed);
        }
        image_logits(normed, img[s], gpu);
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
        image_embed(code, emb, gpu);
        for (int s = 0; s < 2; ++s) {
            m.forward_embedding(0, emb, cache, seq[s], ws, logits, one,
                                normed);
            image_logits(normed, img[s], gpu);
        }
    }
    for (auto& s : seq) cache.release(s);
    return out;
}

}  // namespace locus::image
