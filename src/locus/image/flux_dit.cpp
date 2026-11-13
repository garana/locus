#include "locus/image/flux_dit.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_seq.hpp"
#include "locus/image/dense.hpp"
#include "locus/image/pixart_dit.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

using dense::attention;
using dense::gelu_tanh;
using dense::linear;
using dense::linear1;
using dense::ln_modulate;
using dense::parallel;
using dense::silu;

/** One layer on one vector, with its LoRA parts: out = w x + b + LoRA. */
template <class L>
void run1(const backend::Ops& op, const L& l, std::span<const float> x,
          std::span<float> out) {
    linear1(op, l.w, l.b, x, out);
    if (!l.lora.empty()) apply_lora(op, l.lora, x, 1, out);
}

/** One layer on n token-major rows, with its LoRA parts. */
template <class L>
void run(const backend::Ops& op, const L& l, std::span<const float> x,
         std::uint32_t n, std::span<float> out) {
    linear(op, l.w, l.b, x, n, out);
    if (!l.lora.empty()) apply_lora(op, l.lora, x, n, out);
}

/**
 * diffusers' RMSNorm on each head of n rows (heads x hd wide), in
 * place: x / sqrt(mean(x^2) + eps) * weight.
 */
void head_rms_norm(std::span<float> x, std::uint32_t n, std::uint32_t heads,
                   std::uint32_t hd, float eps,
                   const std::vector<float>& weight) {
    parallel(std::size_t{n} * heads, [&](std::size_t r) {
        float* xi = &x[r * hd];
        float ss = 0.0f;
        for (std::uint32_t i = 0; i < hd; ++i) ss += xi[i] * xi[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(hd) + eps);
        for (std::uint32_t i = 0; i < hd; ++i) xi[i] = xi[i] * inv * weight[i];
    });
}

/**
 * apply_rotary_emb (use_real, unbind -1) on n rows of heads x hd, in
 * place: each pair (2 i, 2 i + 1) of each head rotates by the token's
 * angle, (x0, x1) -> (x0 cos - x1 sin, x1 cos + x0 sin).
 */
void apply_rope(std::span<float> x, std::uint32_t n, std::uint32_t heads,
                std::uint32_t hd, const std::vector<float>& cosv,
                const std::vector<float>& sinv) {
    parallel(n, [&](std::size_t t) {
        const float* c = &cosv[t * hd];
        const float* s = &sinv[t * hd];
        for (std::uint32_t h = 0; h < heads; ++h) {
            float* xi = &x[(t * heads + h) * hd];
            for (std::uint32_t i = 0; i + 1 < hd; i += 2) {
                const float x0 = xi[i], x1 = xi[i + 1];
                xi[i] = x0 * c[i] - x1 * s[i];
                xi[i + 1] = x1 * c[i + 1] + x0 * s[i + 1];
            }
        }
    });
}

/** out[i] = in[i] + gate[i % d] * add[i], over n rows of width d. */
void gated_add(std::span<float> x, std::span<const float> add,
               const float* gate, std::uint32_t n, std::uint32_t d) {
    for (std::size_t t = 0; t < n; ++t) {
        for (std::uint32_t i = 0; i < d; ++i) {
            x[t * d + i] += gate[i] * add[t * d + i];
        }
    }
}

}  // namespace

void FluxDit::rope(std::span<const float> ids, std::uint32_t n,
                   std::vector<float>& cos_out,
                   std::vector<float>& sin_out) const {
    if (ids.size() != std::size_t{3} * n) {
        throw std::invalid_argument("flux: rope ids must be n x 3");
    }
    cos_out.assign(std::size_t{n} * head_dim_, 0.0f);
    sin_out.assign(cos_out.size(), 0.0f);
    for (std::uint32_t t = 0; t < n; ++t) {
        std::uint32_t off = 0;
        for (std::size_t a = 0; a < axes_.size(); ++a) {
            const std::uint32_t d = axes_[a];
            // FluxPosEmbed: float64 frequencies on a float32 position,
            // cos and sin cast to float32, each repeated for its pair.
            const double pos = static_cast<double>(ids[std::size_t{t} * 3 + a]);
            for (std::uint32_t j = 0; j < d / 2; ++j) {
                const double f =
                    1.0 / std::pow(static_cast<double>(theta_),
                                   static_cast<double>(2 * j) / d);
                const double ang = pos * f;
                const auto c = static_cast<float>(std::cos(ang));
                const auto s = static_cast<float>(std::sin(ang));
                const std::size_t k = std::size_t{t} * head_dim_ + off + 2 * j;
                cos_out[k] = cos_out[k + 1] = c;
                sin_out[k] = sin_out[k + 1] = s;
            }
            off += d;
        }
    }
}

bool FluxDit::may_edit_of(const gguf::GgufFile& g) {
    return g.get_bool("flux.kontext").value_or(true);
}

FluxDit FluxDit::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "flux") {
        throw gguf::Error("flux: general.architecture is not flux");
    }
    FluxDit d;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0 || *x > (1u << 20)) {
            throw gguf::Error("flux: missing or bad " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    auto f = [&](std::string_view k) {
        const gguf::Value* v = g.find(k);
        if (v == nullptr || v->type != gguf::ValueType::kFloat32) {
            throw gguf::Error("flux: missing " + std::string(k));
        }
        return static_cast<float>(v->f);
    };
    const std::uint32_t n_double = u("flux.block_count");
    const std::uint32_t n_single = u("flux.single_block_count");
    d.width_ = u("flux.embedding_length");
    d.ffn_ = u("flux.feed_forward_length");
    d.heads_ = u("flux.attention.head_count");
    d.head_dim_ = u("flux.attention.key_length");
    d.eps_ = f("flux.norm_epsilon");
    d.in_ch_ = u("flux.in_channels");
    d.joint_ = u("flux.joint_attention_dim");
    d.pooled_ = u("flux.pooled_projection_dim");
    d.t_ch_ = u("flux.timestep.channels");
    d.theta_ = f("flux.rope.theta");
    const auto gb = g.get_bool("flux.guidance_embeds");
    if (!gb) throw gguf::Error("flux: missing flux.guidance_embeds");
    d.guidance_ = *gb;
    d.may_edit_ = may_edit_of(g);
    if (!g.get_bool("flux.kontext")) {
        std::fprintf(stderr,
                     "flux: no flux.kontext (converted before it existed); "
                     "edits are on. Re-run scripts/convert-flux.py so a "
                     "FLUX.1 dev or schnell checkpoint stops serving "
                     "edits.\n");
    }
    const gguf::Value* ax = g.find("flux.rope.axes");
    if (ax == nullptr || ax->type != gguf::ValueType::kArray ||
        ax->arr.empty() || ax->arr.size() > 8) {
        throw gguf::Error("flux: rope.axes must be an array");
    }
    std::uint32_t axes_sum = 0;
    for (const gguf::Value& e : ax->arr) {
        // The converter writes Python ints, which gguf stores as int32.
        using VT = gguf::ValueType;
        const std::int64_t x = e.type == VT::kInt32 || e.type == VT::kInt64
                                   ? e.i
                                   : static_cast<std::int64_t>(e.u);
        if (x <= 0 || x % 2 != 0 || x > 4096) {
            throw gguf::Error("flux: bad rope.axes entry");
        }
        d.axes_.push_back(static_cast<std::uint32_t>(x));
        axes_sum += static_cast<std::uint32_t>(x);
    }
    if (std::uint64_t{d.heads_} * d.head_dim_ != d.width_ ||
        axes_sum != d.head_dim_ || d.t_ch_ % 2 != 0 ||
        !(d.theta_ > 1.0f) || !(d.eps_ > 0.0f)) {
        throw gguf::Error("flux: inconsistent sizes");
    }

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) throw gguf::Error("flux: missing tensor " + name);
        return ti;
    };
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->type != gguf::TensorType::kF32 || ti->nelements() != n) {
            throw gguf::Error("flux: " + name + " must be F32 with " +
                              std::to_string(n) + " values");
        }
        std::vector<float> out(n);
        std::memcpy(out.data(), g.tensor_data(*ti).data(), n * sizeof(float));
        return out;
    };
    // A Linear [rows, cols], BF16, Q8_0 (convert-flux.py --dit-type
    // q8_0) or F32.
    const auto lin = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name + ".weight");
        if ((ti->type != gguf::TensorType::kBF16 &&
             ti->type != gguf::TensorType::kQ8_0 &&
             ti->type != gguf::TensorType::kF32) ||
            ti->n_dims != 2 || ti->ne[0] != cols || ti->ne[1] != rows) {
            throw gguf::Error("flux: " + name + ".weight is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        Linear l;
        l.w = backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
        l.b = vec(name + ".bias", rows);
        l.name = name;
        return l;
    };
    const auto mlp = [&](const std::string& p, std::uint32_t in) {
        Mlp m;
        m.l1 = lin(p + "linear_1", d.width_, in);
        m.l2 = lin(p + "linear_2", d.width_, d.width_);
        return m;
    };

    const std::uint32_t w = d.width_, hd = d.head_dim_;
    d.x_embed_ = lin("x_embedder", w, d.in_ch_);
    d.ctx_embed_ = lin("context_embedder", w, d.joint_);
    d.time_ = mlp("time_text_embed.timestep_embedder.", d.t_ch_);
    if (d.guidance_) {
        d.guide_ = mlp("time_text_embed.guidance_embedder.", d.t_ch_);
    }
    d.text_ = mlp("time_text_embed.text_embedder.", d.pooled_);
    d.norm_out_ = lin("norm_out.linear", 2 * w, w);
    d.proj_out_ = lin("proj_out", d.in_ch_, w);
    for (std::uint32_t i = 0; i < n_double; ++i) {
        const std::string b = "transformer_blocks." + std::to_string(i) + ".";
        DoubleBlock k;
        k.img.mod = lin(b + "norm1.linear", 6 * w, w);
        k.txt.mod = lin(b + "norm1_context.linear", 6 * w, w);
        k.img.q = lin(b + "attn.to_q", w, w);
        k.img.k = lin(b + "attn.to_k", w, w);
        k.img.v = lin(b + "attn.to_v", w, w);
        k.img.out = lin(b + "attn.to_out.0", w, w);
        k.img.q_norm = vec(b + "attn.norm_q.weight", hd);
        k.img.k_norm = vec(b + "attn.norm_k.weight", hd);
        k.txt.q = lin(b + "attn.add_q_proj", w, w);
        k.txt.k = lin(b + "attn.add_k_proj", w, w);
        k.txt.v = lin(b + "attn.add_v_proj", w, w);
        k.txt.out = lin(b + "attn.to_add_out", w, w);
        k.txt.q_norm = vec(b + "attn.norm_added_q.weight", hd);
        k.txt.k_norm = vec(b + "attn.norm_added_k.weight", hd);
        k.img.ff1 = lin(b + "ff.net.0.proj", d.ffn_, w);
        k.img.ff2 = lin(b + "ff.net.2", w, d.ffn_);
        k.txt.ff1 = lin(b + "ff_context.net.0.proj", d.ffn_, w);
        k.txt.ff2 = lin(b + "ff_context.net.2", w, d.ffn_);
        d.doubles_.push_back(std::move(k));
    }
    for (std::uint32_t i = 0; i < n_single; ++i) {
        const std::string b =
            "single_transformer_blocks." + std::to_string(i) + ".";
        SingleBlock k;
        k.mod = lin(b + "norm.linear", 3 * w, w);
        k.q = lin(b + "attn.to_q", w, w);
        k.k = lin(b + "attn.to_k", w, w);
        k.v = lin(b + "attn.to_v", w, w);
        k.q_norm = vec(b + "attn.norm_q.weight", hd);
        k.k_norm = vec(b + "attn.norm_k.weight", hd);
        k.mlp = lin(b + "proj_mlp", d.ffn_, w);
        k.out = lin(b + "proj_out", w, w + d.ffn_);
        d.singles_.push_back(std::move(k));
    }
    return d;
}

template <class F>
void FluxDit::each_linear(F&& fn) {
    for (Linear* l : {&x_embed_, &ctx_embed_, &norm_out_, &proj_out_,
                      &time_.l1, &time_.l2, &text_.l1, &text_.l2}) {
        fn(*l);
    }
    if (guidance_) {
        fn(guide_.l1);
        fn(guide_.l2);
    }
    for (DoubleBlock& b : doubles_) {
        for (Stream* s : {&b.img, &b.txt}) {
            for (Linear* l : {&s->mod, &s->q, &s->k, &s->v, &s->out,
                              &s->ff1, &s->ff2}) {
                fn(*l);
            }
        }
    }
    for (SingleBlock& b : singles_) {
        for (Linear* l : {&b.mod, &b.q, &b.k, &b.v, &b.mlp, &b.out}) {
            fn(*l);
        }
    }
}

void FluxDit::set_lora(const LoraSet* set) {
    each_linear([](Linear& l) { l.lora.clear(); });
    has_lora_ = false;
    if (set == nullptr || set->empty()) return;
    std::map<std::string, Linear*, std::less<>> by_name;
    each_linear([&](Linear& l) { by_name[l.name] = &l; });
    for (const std::string& layer : set->layers()) {
        const auto it = by_name.find(layer);
        const auto fail = [&](const std::string& why) {
            set_lora(nullptr);
            throw std::invalid_argument("flux lora: " + layer + " " + why);
        };
        if (it == by_name.end()) fail("is not a linear layer of this model");
        Linear& l = *it->second;
        for (const LoraPart& p : set->parts(layer)) {
            if (p.a.cols != l.w.cols || p.b.rows != l.w.rows) {
                fail("does not fit the " + std::to_string(l.w.rows) + " x " +
                     std::to_string(l.w.cols) + " weight");
            }
            l.lora.push_back(p);
        }
    }
    has_lora_ = true;
}

std::vector<float> FluxDit::forward(std::span<const float> x,
                                    std::span<const float> img_ids,
                                    std::uint32_t n_img,
                                    std::span<const float> text,
                                    std::span<const float> txt_ids,
                                    std::uint32_t n_txt,
                                    std::span<const float> pooled,
                                    float sigma, float guidance,
                                    const backend::Ops& op,
                                    Trace* trace) const {
    if (n_img == 0 || n_txt == 0 ||
        x.size() != std::size_t{n_img} * in_ch_ ||
        img_ids.size() != std::size_t{n_img} * 3 ||
        text.size() != std::size_t{n_txt} * joint_ ||
        txt_ids.size() != std::size_t{n_txt} * 3 ||
        pooled.size() != pooled_) {
        throw std::invalid_argument("flux: forward sizes");
    }
    const std::uint32_t w = width_, hd = head_dim_, nh = heads_;
    const std::uint32_t n = n_txt + n_img;  // joined: text, then image

    // The conditioning vector, then silu of it for every modulation.
    const auto embed = [&](const Mlp& m, std::span<const float> in,
                           std::vector<float>& acc) {
        std::vector<float> h(w), o(w);
        run1(op, m.l1, in, h);
        for (float& v : h) v = silu(v);
        run1(op, m.l2, h, o);
        for (std::uint32_t i = 0; i < w; ++i) acc[i] += o[i];
    };
    std::vector<float> temb(w, 0.0f);
    embed(time_, PixArtDit::timestep_features(sigma * 1000.0f, t_ch_), temb);
    if (guidance_) {
        embed(guide_, PixArtDit::timestep_features(guidance * 1000.0f, t_ch_),
              temb);
    }
    embed(text_, pooled, temb);
    std::vector<float> st(w);
    for (std::uint32_t i = 0; i < w; ++i) st[i] = silu(temb[i]);

    // RoPE angles over the joined sequence.
    std::vector<float> ids(std::size_t{n} * 3);
    std::copy(txt_ids.begin(), txt_ids.end(), ids.begin());
    std::copy(img_ids.begin(), img_ids.end(),
              ids.begin() + static_cast<std::ptrdiff_t>(n_txt) * 3);
    std::vector<float> cosv, sinv;
    rope(ids, n, cosv, sinv);
    if (&op == &backend::vulkan_image_ops() &&
        backend::VulkanSeq::available()) {
        const char* env = std::getenv("LOCUS_FLUX_GPU");
        if (env == nullptr || std::string(env) != "0") {
            return forward_vulkan(x, n_img, text, n_txt, st, cosv, sinv,
                                  trace);
        }
    }

    std::vector<float> img(std::size_t{n_img} * w), txt(std::size_t{n_txt} * w);
    run(op, x_embed_, x, n_img, img);
    run(op, ctx_embed_, text, n_txt, txt);

    const std::size_t nw = std::size_t{n} * w;
    std::vector<float> q(nw), k(nw), v(nw), att(nw), nrm(nw), o(nw);
    std::vector<float> ff(std::size_t{n} * ffn_);
    std::vector<float> mod_i(std::size_t{6} * w), mod_t(mod_i.size());

    // q / k / v of one stream into rows [row0, row0 + rows) of q, k, v.
    const auto qkv = [&](const Stream& s, std::span<const float> in,
                         std::uint32_t rows, std::uint32_t row0) {
        const std::size_t at = std::size_t{row0} * w;
        const std::size_t len = std::size_t{rows} * w;
        const auto sub = [&](std::vector<float>& b) {
            return std::span<float>(b).subspan(at, len);
        };
        run(op, s.q, in, rows, sub(q));
        run(op, s.k, in, rows, sub(k));
        run(op, s.v, in, rows, sub(v));
        head_rms_norm(sub(q), rows, nh, hd, eps_, s.q_norm);
        head_rms_norm(sub(k), rows, nh, hd, eps_, s.k_norm);
    };
    // Gated residual of one stream's attention output, then its gated
    // FFN on the modulated LayerNorm. mod: shift, scale, gate (msa),
    // shift, scale, gate (mlp).
    const auto finish = [&](const Stream& s, std::vector<float>& h,
                            std::uint32_t rows, std::uint32_t row0,
                            const std::vector<float>& mod) {
        const std::size_t len = std::size_t{rows} * w;
        const auto a = std::span<const float>(att).subspan(
            std::size_t{row0} * w, len);
        auto ob = std::span<float>(o).subspan(0, len);
        run(op, s.out, a, rows, ob);
        gated_add(h, ob, &mod[2 * w], rows, w);
        auto nb = std::span<float>(nrm).subspan(0, len);
        ln_modulate(h, rows, w, eps_, &mod[3 * w], &mod[4 * w], nb);
        auto fb = std::span<float>(ff).subspan(0, std::size_t{rows} * ffn_);
        run(op, s.ff1, nb, rows, fb);
        for (float& z : fb) z = gelu_tanh(z);
        run(op, s.ff2, fb, rows, ob);
        gated_add(h, ob, &mod[5 * w], rows, w);
    };

    for (std::size_t bi = 0; bi < doubles_.size(); ++bi) {
        const DoubleBlock& b = doubles_[bi];
        run1(op, b.img.mod, st, mod_i);
        run1(op, b.txt.mod, st, mod_t);
        auto ni = std::span<float>(nrm).subspan(0, std::size_t{n_img} * w);
        ln_modulate(img, n_img, w, eps_, &mod_i[0], &mod_i[w], ni);
        qkv(b.img, ni, n_img, n_txt);
        auto nt = std::span<float>(nrm).subspan(0, std::size_t{n_txt} * w);
        ln_modulate(txt, n_txt, w, eps_, &mod_t[0], &mod_t[w], nt);
        qkv(b.txt, nt, n_txt, 0);
        apply_rope(q, n, nh, hd, cosv, sinv);
        apply_rope(k, n, nh, hd, cosv, sinv);
        attention(op, q, n, k, v, n, nh, hd, att);
        finish(b.txt, txt, n_txt, 0, mod_t);
        finish(b.img, img, n_img, n_txt, mod_i);
        if (trace != nullptr && bi == 0) {
            trace->double0_img = img;
            trace->double0_txt = txt;
        }
    }

    // Single blocks on the joined rows: text, then image.
    std::vector<float> h(nw);
    std::copy(txt.begin(), txt.end(), h.begin());
    std::copy(img.begin(), img.end(),
              h.begin() + static_cast<std::ptrdiff_t>(std::size_t{n_txt} * w));
    std::vector<float> mod(std::size_t{3} * w);
    std::vector<float> cat(std::size_t{n} * (w + ffn_));
    for (std::size_t bi = 0; bi < singles_.size(); ++bi) {
        const SingleBlock& b = singles_[bi];
        run1(op, b.mod, st, mod);
        ln_modulate(h, n, w, eps_, &mod[0], &mod[w], nrm);
        run(op, b.mlp, nrm, n, ff);
        run(op, b.q, nrm, n, q);
        run(op, b.k, nrm, n, k);
        run(op, b.v, nrm, n, v);
        head_rms_norm(q, n, nh, hd, eps_, b.q_norm);
        head_rms_norm(k, n, nh, hd, eps_, b.k_norm);
        apply_rope(q, n, nh, hd, cosv, sinv);
        apply_rope(k, n, nh, hd, cosv, sinv);
        attention(op, q, n, k, v, n, nh, hd, att);
        // proj_out on [attention, gelu(mlp)] per token.
        parallel(n, [&](std::size_t t) {
            float* c = &cat[t * (w + ffn_)];
            std::memcpy(c, &att[t * w], std::size_t{w} * sizeof(float));
            for (std::uint32_t i = 0; i < ffn_; ++i) {
                c[w + i] = gelu_tanh(ff[t * ffn_ + i]);
            }
        });
        run(op, b.out, cat, n, o);
        gated_add(h, o, &mod[2 * w], n, w);
        if (trace != nullptr && bi == 0) trace->single0 = h;
    }

    // norm_out (AdaLayerNormContinuous: scale, then shift) on the image
    // rows, then proj_out.
    std::vector<float> ss(std::size_t{2} * w);
    run1(op, norm_out_, st, ss);
    const auto hi = std::span<const float>(h).subspan(
        std::size_t{n_txt} * w, std::size_t{n_img} * w);
    auto ni = std::span<float>(nrm).subspan(0, std::size_t{n_img} * w);
    ln_modulate(hi, n_img, w, eps_, &ss[w], &ss[0], ni);
    std::vector<float> out(std::size_t{n_img} * in_ch_);
    run(op, proj_out_, ni, n_img, out);
    return out;
}

}  // namespace locus::image
