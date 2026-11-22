#include "locus/embed/deberta.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "locus/image/dense.hpp"
#include "locus/image/tensor_read.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::embed {

namespace {

using image::dense::gelu;  // exact erf form, as hidden_act "gelu"
using image::dense::layer_norm;

/** Queries per block of the attention: keeps a block's rows in cache. */
constexpr std::uint32_t kQueryBlock = 64;

/** out[r * n + t] = w[r] . x[t] for n token-major inputs (the
 *  matvec_batch layout), or n matvec calls when op has none. */
void mat_batch(const backend::Ops& op, const backend::Mat& w,
               std::span<const float> x, std::uint32_t n,
               std::span<float> out) {
    if (op.matvec_batch != nullptr) {
        op.matvec_batch(w, x.first(std::size_t{n} * w.cols),
                        out.first(std::size_t{n} * w.rows), n);
        return;
    }
    std::vector<float> o(w.rows);
    for (std::uint32_t t = 0; t < n; ++t) {
        op.matvec(w, x.subspan(std::size_t{t} * w.cols, w.cols), o);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[std::size_t{r} * n + t] = o[r];
        }
    }
}

/** An F32 host matrix of rows x cols at p, for mat_batch. */
backend::Mat f32_mat(const float* p, std::uint32_t rows, std::uint32_t cols) {
    return backend::Mat{gguf::TensorType::kF32,
                        reinterpret_cast<const std::byte*>(p), rows, cols};
}

}  // namespace

std::int32_t DebertaEncoder::bucket(std::int32_t rel, std::uint32_t buckets,
                                    std::uint32_t max_position) {
    const auto mid = static_cast<std::int32_t>(buckets / 2);
    const std::int32_t abs_pos =
        rel < mid && rel > -mid ? mid - 1 : std::abs(rel);
    if (abs_pos <= mid) return rel;
    // As torch runs it: float32 throughout, the ratio's log as a
    // float32 tensor.
    const float log_pos =
        std::ceil(std::log(static_cast<float>(abs_pos) /
                           static_cast<float>(mid)) /
                  std::log(static_cast<float>(max_position - 1) /
                           static_cast<float>(mid)) *
                  static_cast<float>(mid - 1)) +
        static_cast<float>(mid);
    return static_cast<std::int32_t>(log_pos) * (rel > 0 ? 1 : -1);
}

DebertaEncoder DebertaEncoder::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "deberta-v2") {
        throw gguf::Error("deberta: general.architecture is not deberta-v2");
    }
    const std::string ns = "deberta-v2.";
    const auto u = [&](const std::string& key) {
        const auto v = g.get_uint(ns + key);
        if (!v || *v == 0 || *v > (1u << 20)) {
            throw gguf::Error("deberta: missing or bad " + ns + key);
        }
        return static_cast<std::uint32_t>(*v);
    };
    DebertaEncoder e;
    e.width_ = u("embedding_length");
    e.heads_ = u("attention.head_count");
    e.ffn_ = u("feed_forward_length");
    e.context_ = u("context_length");
    e.buckets_ = u("position_buckets");
    e.max_rel_ = u("max_relative_positions");
    const std::uint32_t blocks = u("block_count");
    if (blocks > 1024) throw gguf::Error("deberta: too many blocks");
    if (e.width_ % e.heads_ != 0) {
        throw gguf::Error("deberta: width is not a multiple of the heads");
    }
    if (e.buckets_ < 2 || e.max_rel_ <= e.buckets_ / 2 + 1) {
        throw gguf::Error("deberta: position_buckets must be at least 2 "
                          "and below 2 * (max_relative_positions - 1)");
    }
    const gguf::Value* eps = g.find(ns + "attention.layer_norm_epsilon");
    if (eps == nullptr || eps->type != gguf::ValueType::kFloat32 ||
        !(eps->f > 0.0f)) {
        throw gguf::Error("deberta: missing or bad layer_norm_epsilon");
    }
    e.eps_ = static_cast<float>(eps->f);
    if (g.get_uint(ns + "pooling_type") != 4u) {
        throw gguf::Error("deberta: pooling_type must be 4 (a classifier)");
    }

    const auto info = [&](const std::string& name) {
        return image::require_tensor(g, "deberta", name);
    };
    const auto vec = [&](const std::string& name, std::size_t n) {
        return image::read_floats(g, "deberta", name, n);
    };
    const auto mat = [&](const std::string& name, std::uint32_t rows,
                         std::uint32_t cols) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->n_dims != 2 || ti->ne[0] != cols || ti->ne[1] != rows) {
            throw gguf::Error("deberta: " + name + " is not " +
                              std::to_string(rows) + " x " +
                              std::to_string(cols));
        }
        return backend::Mat{ti->type, g.tensor_data(*ti).data(), rows, cols};
    };
    const auto lin = [&](const std::string& p, std::uint32_t rows,
                         std::uint32_t cols) {
        return Linear{mat(p + ".weight", rows, cols), vec(p + ".bias", rows)};
    };
    const auto norm = [&](const std::string& p) {
        return Norm{vec(p + ".weight", e.width_), vec(p + ".bias", e.width_)};
    };

    const std::uint32_t w = e.width_;
    const gguf::TensorInfo* tok = info("token_embd.weight");
    if (tok->n_dims != 2 || tok->ne[0] != w || tok->ne[1] == 0 ||
        tok->ne[1] > 0x7fffffff) {
        throw gguf::Error("deberta: token_embd.weight has the wrong shape");
    }
    e.vocab_ = static_cast<std::uint32_t>(tok->ne[1]);
    e.tok_embd_ = mat("token_embd.weight", e.vocab_, w);
    e.embd_norm_ = norm("token_embd_norm");
    // The relative embeddings are the same for every input: normalized
    // once here (norm_rel_ebd "layer_norm").
    const std::uint32_t rows = 2 * e.buckets_;
    e.rel_ = vec("rel_embd.weight", std::size_t{rows} * w);
    const Norm rel_norm = norm("rel_embd_norm");
    layer_norm(e.rel_, rows, w, rel_norm.w, rel_norm.b, e.eps_, e.rel_);
    for (std::uint32_t i = 0; i < blocks; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        Layer l;
        l.q = lin(p + "attn_q", w, w);
        l.k = lin(p + "attn_k", w, w);
        l.v = lin(p + "attn_v", w, w);
        l.o = lin(p + "attn_output", w, w);
        l.attn_norm = norm(p + "attn_output_norm");
        l.up = lin(p + "ffn_up", e.ffn_, w);
        l.down = lin(p + "ffn_down", w, e.ffn_);
        l.out_norm = norm(p + "layer_output_norm");
        e.layers_.push_back(std::move(l));
    }
    e.cls_ = lin("cls", w, w);
    const gguf::TensorInfo* out = info("cls.output.weight");
    if (out->n_dims != 2 || out->ne[0] != w || out->ne[1] == 0 ||
        out->ne[1] > 1024) {
        throw gguf::Error("deberta: cls.output.weight has the wrong shape");
    }
    const auto n_labels = static_cast<std::uint32_t>(out->ne[1]);
    e.cls_out_ = lin("cls.output", n_labels, w);
    const auto* names = g.get_array(ns + "classifier.output_labels");
    if (names == nullptr || names->size() != n_labels) {
        throw gguf::Error("deberta: classifier.output_labels must name each "
                          "of the " + std::to_string(n_labels) + " labels");
    }
    for (const gguf::Value& v : *names) {
        if (v.type != gguf::ValueType::kString) {
            throw gguf::Error("deberta: a label name is not a string");
        }
        e.labels_.push_back(v.s);
    }
    for (const Layer& l : e.layers_) {
        for (const Linear* x : {&l.q, &l.k, &l.v, &l.o, &l.up, &l.down}) {
            e.token_macs_ += std::uint64_t{x->w.rows} * x->w.cols;
        }
    }
    return e;
}

void DebertaEncoder::attend(std::span<const float> q, std::span<const float> k,
                            std::span<const float> v, std::uint32_t m,
                            std::span<const float> pk,
                            std::span<const float> pq, std::uint32_t lo,
                            std::uint32_t nr, std::span<float> out) const {
    const std::uint32_t w = width_, heads = heads_, hd = width_ / heads_;
    // The three terms share one scale, sqrt(head_dim * 3) in float32.
    const float inv_scale =
        1.0f / std::sqrt(static_cast<float>(hd) * 3.0f);
    // Per distance d = i - j, the relative row the two position terms
    // use, minus lo: rix[d + m - 1].
    std::vector<std::uint32_t> rix(2 * std::size_t{m} - 1);
    const auto span = static_cast<std::int32_t>(buckets_);
    for (std::int32_t d = -static_cast<std::int32_t>(m - 1);
         d <= static_cast<std::int32_t>(m - 1); ++d) {
        const std::int32_t c =
            std::clamp(bucket(d, buckets_, max_rel_) + span, 0, 2 * span - 1);
        rix[static_cast<std::size_t>(d + static_cast<std::int32_t>(m) - 1)] =
            static_cast<std::uint32_t>(c) - lo;
    }
    // Per head, contiguous copies: keys [m][hd], values [hd][m], the
    // position keys and queries [nr][hd], and the position-to-content
    // term pt[r * m + j] = pq_r . k_j, which every query reads.
    std::vector<float> kh(std::size_t{heads} * m * hd),
        vt(std::size_t{heads} * hd * m), pkh(std::size_t{heads} * nr * hd),
        pqh(std::size_t{heads} * nr * hd), pt(std::size_t{heads} * nr * m);
    // These are temporary host buffers, so they run on the CPU backend
    // (a GPU backend caches its weights by host pointer).
    const backend::Ops& cpu = backend::best_backend().ops;
    auto& pool = sys::ThreadPool::instance();
    pool.parallel_for(heads, [&](std::size_t h) {
        float* kd = &kh[h * m * hd];
        float* vd = &vt[h * hd * m];
        for (std::uint32_t j = 0; j < m; ++j) {
            std::memcpy(kd + std::size_t{j} * hd,
                        &k[std::size_t{j} * w + h * hd], hd * sizeof(float));
            const float* vs = &v[std::size_t{j} * w + h * hd];
            for (std::uint32_t e = 0; e < hd; ++e) {
                vd[std::size_t{e} * m + j] = vs[e];
            }
        }
        float* pkd = &pkh[h * nr * hd];
        float* pqd = &pqh[h * nr * hd];
        for (std::uint32_t r = 0; r < nr; ++r) {
            std::memcpy(pkd + std::size_t{r} * hd,
                        &pk[std::size_t{r} * w + h * hd], hd * sizeof(float));
            std::memcpy(pqd + std::size_t{r} * hd,
                        &pq[std::size_t{r} * w + h * hd], hd * sizeof(float));
        }
        mat_batch(cpu, f32_mat(pqd, nr, hd), {kd, std::size_t{m} * hd}, m,
                  {&pt[h * nr * m], std::size_t{nr} * m});
    });
    const std::uint32_t qblocks = (m + kQueryBlock - 1) / kQueryBlock;
    pool.parallel_for(std::size_t{heads} * qblocks, [&](std::size_t item) {
        const auto h = static_cast<std::uint32_t>(item / qblocks);
        const auto q0 =
            static_cast<std::uint32_t>(item % qblocks) * kQueryBlock;
        const std::uint32_t mb = std::min(kQueryBlock, m - q0);
        std::vector<float> qb(std::size_t{mb} * hd), s(std::size_t{m} * mb),
            a(std::size_t{nr} * mb), p(std::size_t{mb} * m),
            ob(std::size_t{hd} * mb);
        for (std::uint32_t t = 0; t < mb; ++t) {
            std::memcpy(&qb[std::size_t{t} * hd],
                        &q[std::size_t{q0 + t} * w + h * hd],
                        hd * sizeof(float));
        }
        // s[j * mb + t] = q . k_j; a[r * mb + t] = q . pk_r (c2p).
        mat_batch(cpu, f32_mat(&kh[std::size_t{h} * m * hd], m, hd), qb, mb,
                  s);
        mat_batch(cpu, f32_mat(&pkh[std::size_t{h} * nr * hd], nr, hd), qb,
                  mb, a);
        const float* ph = &pt[std::size_t{h} * nr * m];
        for (std::uint32_t t = 0; t < mb; ++t) {
            const std::uint32_t i = q0 + t;
            float* row = &p[std::size_t{t} * m];
            float mx = -INFINITY;
            for (std::uint32_t j = 0; j < m; ++j) {
                const std::uint32_t c = rix[std::size_t{i} + m - 1 - j];
                row[j] = (s[std::size_t{j} * mb + t] +
                          a[std::size_t{c} * mb + t] +
                          ph[std::size_t{c} * m + j]) *
                         inv_scale;
                mx = std::max(mx, row[j]);
            }
            float sum = 0.0f;
            for (std::uint32_t j = 0; j < m; ++j) {
                row[j] = std::exp(row[j] - mx);
                sum += row[j];
            }
            const float inv = 1.0f / sum;
            for (std::uint32_t j = 0; j < m; ++j) row[j] *= inv;
        }
        mat_batch(cpu, f32_mat(&vt[std::size_t{h} * hd * m], hd, m), p, mb,
                  ob);  // ob[e * mb + t]
        for (std::uint32_t t = 0; t < mb; ++t) {
            float* o = &out[std::size_t{q0 + t} * w + h * hd];
            for (std::uint32_t e = 0; e < hd; ++e) {
                o[e] = ob[std::size_t{e} * mb + t];
            }
        }
    });
}

std::vector<std::vector<float>> DebertaEncoder::encode(
    std::span<const std::vector<tok::TokenId>> batch,
    const backend::Ops& op) const {
    const std::uint32_t w = width_;
    // Each sequence's first row in the batch, the total and the longest.
    std::vector<std::uint32_t> start;
    std::uint32_t n = 0, longest = 0;
    for (const auto& ids : batch) {
        if (ids.empty() || ids.size() > kMaxTokens) {
            throw std::invalid_argument(
                "deberta: a sequence needs 1 to " +
                std::to_string(kMaxTokens) + " tokens, not " +
                std::to_string(ids.size()));
        }
        for (const tok::TokenId id : ids) {
            if (id < 0 || static_cast<std::uint32_t>(id) >= vocab_) {
                throw std::invalid_argument(
                    "deberta: id outside the vocabulary");
            }
        }
        start.push_back(n);
        n += static_cast<std::uint32_t>(ids.size());
        longest = std::max(longest, static_cast<std::uint32_t>(ids.size()));
    }
    if (n == 0) return {};

    // The relative rows any pair of the longest sequence can reach,
    // [lo, lo + nr): the position terms need only these.
    const auto span = static_cast<std::int32_t>(buckets_);
    const auto reach = [&](std::int32_t d) {
        return static_cast<std::uint32_t>(std::clamp(
            bucket(d, buckets_, max_rel_) + span, 0, 2 * span - 1));
    };
    const auto far = static_cast<std::int32_t>(longest - 1);
    const std::uint32_t lo = reach(-far);
    const std::uint32_t nr = reach(far) - lo + 1;
    const std::span<const float> rel(rel_.data() + std::size_t{lo} * w,
                                     std::size_t{nr} * w);

    const std::size_t nw = std::size_t{n} * w;
    std::vector<float> x(nw), h(nw), q(nw), k(nw), v(nw), a(nw);
    std::vector<float> f(std::size_t{n} * ffn_), pk(std::size_t{nr} * w),
        pq(std::size_t{nr} * w);
    for (std::size_t s = 0; s < batch.size(); ++s) {
        for (std::uint32_t t = 0; t < batch[s].size(); ++t) {
            op.dequant_row(tok_embd_, static_cast<std::uint32_t>(batch[s][t]),
                           std::span<float>(x).subspan(
                               std::size_t{start[s] + t} * w, w));
        }
    }
    layer_norm(x, n, w, embd_norm_.w, embd_norm_.b, eps_, x);
    for (const Layer& l : layers_) {
        image::dense::linear(op, l.q.w, l.q.b, x, n, q);
        image::dense::linear(op, l.k.w, l.k.b, x, n, k);
        image::dense::linear(op, l.v.w, l.v.b, x, n, v);
        // share_att_key: the layer's own k and q projections on the
        // relative embeddings.
        image::dense::linear(op, l.k.w, l.k.b, rel, nr, pk);
        image::dense::linear(op, l.q.w, l.q.b, rel, nr, pq);
        for (std::size_t s = 0; s < batch.size(); ++s) {
            const std::size_t o = std::size_t{start[s]} * w;
            const auto m = static_cast<std::uint32_t>(batch[s].size());
            const std::size_t len = std::size_t{m} * w;
            attend(std::span<const float>(q).subspan(o, len),
                   std::span<const float>(k).subspan(o, len),
                   std::span<const float>(v).subspan(o, len), m, pk, pq, lo,
                   nr, std::span<float>(a).subspan(o, len));
        }
        image::dense::linear(op, l.o.w, l.o.b, a, n, h);
        for (std::size_t i = 0; i < nw; ++i) x[i] += h[i];
        layer_norm(x, n, w, l.attn_norm.w, l.attn_norm.b, eps_, x);
        image::dense::linear(op, l.up.w, l.up.b, x, n, f);
        gelu(f);
        image::dense::linear(op, l.down.w, l.down.b, f, n, h);
        for (std::size_t i = 0; i < nw; ++i) x[i] += h[i];
        layer_norm(x, n, w, l.out_norm.w, l.out_norm.b, eps_, x);
    }
    std::vector<std::vector<float>> out;
    out.reserve(batch.size());
    for (std::size_t s = 0; s < batch.size(); ++s) {
        const auto first = x.begin() + static_cast<std::ptrdiff_t>(
                                           std::size_t{start[s]} * w);
        out.emplace_back(first, first + static_cast<std::ptrdiff_t>(
                                            batch[s].size() * w));
    }
    return out;
}

std::vector<std::vector<float>> DebertaEncoder::classify(
    std::span<const std::vector<tok::TokenId>> batch,
    const backend::Ops& op) const {
    const auto states = encode(batch, op);
    const std::uint32_t w = width_;
    const auto n = static_cast<std::uint32_t>(states.size());
    const auto nl = static_cast<std::uint32_t>(labels_.size());
    // The head over the whole batch: each sequence's first ([CLS])
    // state through the pooler (cls, then GELU), then cls.output.
    std::vector<float> first(std::size_t{n} * w), h(std::size_t{n} * w),
        logits(std::size_t{n} * nl);
    for (std::uint32_t s = 0; s < n; ++s) {
        std::copy_n(states[s].begin(), w, first.begin() + std::size_t{s} * w);
    }
    image::dense::linear(op, cls_.w, cls_.b, first, n, h);
    gelu(h);
    image::dense::linear(op, cls_out_.w, cls_out_.b, h, n, logits);
    std::vector<std::vector<float>> out;
    out.reserve(n);
    for (std::uint32_t s = 0; s < n; ++s) {
        const auto row = logits.begin() + std::size_t{s} * nl;
        out.emplace_back(row, row + nl);
    }
    return out;
}

}  // namespace locus::embed
