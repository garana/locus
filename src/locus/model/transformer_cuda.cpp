// CUDA forward of a plain llama model (i#517): each layer's activations
// stay on the GPU between the matvecs, the norms, RoPE, the attention
// over the device KV mirror and the residual adds.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/model/arch.hpp"
#include "locus/model/transformer.hpp"

namespace locus::model {

namespace {

/** Grow-only device buffers of one thread's CUDA forward. */
struct DevBufs {
    float *x = nullptr, *xb = nullptr, *q = nullptr, *k = nullptr,
          *v = nullptr, *att = nullptr, *o = nullptr, *gate = nullptr,
          *up = nullptr, *scratch = nullptr, *logits = nullptr,
          *cs = nullptr, *stage = nullptr;
    void *rows = nullptr, *blocks = nullptr, *meta = nullptr;
    std::map<const void*, std::size_t> cap;

    /** Makes p hold at least n floats. @returns false when it cannot. */
    bool floats(float*& p, std::size_t n) {
        std::size_t& have = cap[&p];
        if (n <= have && p != nullptr) {
            return true;
        }
        backend::cuda_dev_free(p);
        p = backend::cuda_dev_alloc(n);
        have = p != nullptr ? n : 0;
        return p != nullptr;
    }
    bool bytes(void*& p, std::size_t n) {
        std::size_t& have = cap[&p];
        if (n <= have && p != nullptr) {
            return true;
        }
        backend::cuda_dev_free_bytes(p);
        p = backend::cuda_dev_alloc_bytes(n);
        have = p != nullptr ? n : 0;
        return p != nullptr;
    }
};

}  // namespace

bool TransformerModel::cuda_forward(
    std::span<const tok::TokenId> toks,
    std::span<kv::PagedKvCache::Seq* const> seqs,
    std::span<const std::uint32_t> pos, kv::PagedKvCache& cache,
    Workspace& ws, std::span<float> logits, bool all_logits,
    std::span<const float* const> embd, LogitRows rows_out) const {
    using namespace locus::backend;
    static const bool off = [] {
        const char* e = std::getenv("LOCUS_CUDA_FORWARD");
        return e != nullptr && std::string_view(e) == "0";
    }();
    const auto n = static_cast<std::uint32_t>(toks.size());
    if (off || n == 0 || backend_->name != "cuda" ||
        !spec_->plain_attention || cache.quantized() ||
        cache.mirror_data() == nullptr || layer_begin_ != 0 ||
        layer_end_ != hp_.n_layers || hp_.norm_type != NormType::kRmsNorm ||
        hp_.embd_norm ||
        hp_.n_kv_heads == 0 || out_w_.rows != hp_.n_vocab) {
        return false;
    }
    for (const Layer& lay : layers_) {
        if (lay.is_moe()) {
            return false;
        }
    }
    if (!embd.empty() && embd.size() != n) {
        return false;
    }
    const auto embedded = [&](std::uint32_t t) {
        return !embd.empty() && embd[t] != nullptr;
    };
    for (std::uint32_t t = 0; t < n; ++t) {
        if ((!embedded(t) &&
             (toks[t] < 0 ||
              static_cast<std::uint32_t>(toks[t]) >= hp_.n_vocab)) ||
            pos[t] + 1 > hp_.n_ctx || pos[t] >= cache.capacity(*seqs[t])) {
            return false;  // the host path reports it
        }
    }
    const Ops op = ops();
    if (op.kv_attend == nullptr) {
        return false;
    }
    const std::uint32_t E = hp_.n_embd, hd = hp_.head_dim,
                        H = hp_.n_heads, KV = hp_.n_kv_heads,
                        ff = hp_.n_ff, V = hp_.n_vocab, L = hp_.n_layers;
    const std::size_t qd = std::size_t{H} * hd;
    const std::size_t kvd = std::size_t{KV} * hd;
    const std::uint32_t nh = all_logits ? n : 1;  // tokens with logits
    const std::uint32_t bt = cache.geometry().block_tokens;
    // The head rows to compute, [lo, lo + R) (speech needs only its own
    // tokens', i#689); the rest of each token's logits are -inf. A bad
    // range is checked before the subtraction, which would wrap.
    if (rows_out.end != 0 &&
        (rows_out.begin >= rows_out.end || rows_out.end > V)) {
        return false;  // the host path reports it
    }
    const std::uint32_t lo = rows_out.end == 0 ? 0 : rows_out.begin;
    const std::uint32_t R = (rows_out.end == 0 ? V : rows_out.end) - lo;
    const Mat head = R == V ? out_w_ : mat_rows(out_w_, lo, R);

    // Host tables: embeddings, RoPE (cos, sin) pairs, each token's row
    // offset in the mirror, and the attention's block tables (one per
    // distinct sequence) and query table (positions, then offsets).
    std::vector<float> xh(std::size_t{n} * E), cs(std::size_t{n} * hd);
    std::vector<std::uint64_t> rows(n);
    std::vector<std::uint32_t> blocks, meta(std::size_t{2} * n);
    std::map<const kv::PagedKvCache::Seq*, std::uint32_t> table_of;
    for (std::uint32_t t = 0; t < n; ++t) {
        const kv::PagedKvCache::Seq& s = *seqs[t];
        if (embedded(t)) {
            std::copy_n(embd[t], E, xh.data() + std::size_t{t} * E);
        } else {
            op.dequant_row(embd_, static_cast<std::uint32_t>(toks[t]),
                           {xh.data() + std::size_t{t} * E, E});
        }
        rope_norm_table(hd, pos[t], hp_.rope_freq_base, rope_factors(),
                        {cs.data() + std::size_t{t} * hd, hd});
        rows[t] = std::uint64_t{s.blocks[pos[t] / bt]} * cache.block_floats() +
                  std::uint64_t{pos[t] % bt} * kvd;
        auto [it, fresh] = table_of.try_emplace(
            &s, static_cast<std::uint32_t>(blocks.size()));
        if (fresh) {
            blocks.insert(blocks.end(), s.blocks.begin(), s.blocks.end());
        }
        meta[t] = pos[t];
        meta[n + t] = it->second;
    }

    thread_local DevBufs b;
    // The batch kernels' row-major slices go through scratch before the
    // transpose: rows * min(n, 32) floats for the widest weight.
    const std::size_t widest =
        std::max<std::size_t>({qd, E, ff, all_logits ? R : 0});
    const std::size_t stage_stride = std::size_t{L} * 2 * kvd;
    if (!b.floats(b.x, std::size_t{n} * E) ||
        !b.floats(b.xb, std::size_t{n} * E) ||
        !b.floats(b.q, std::size_t{n} * qd) ||
        !b.floats(b.k, std::size_t{n} * kvd) ||
        !b.floats(b.v, std::size_t{n} * kvd) ||
        !b.floats(b.att, std::size_t{n} * qd) ||
        !b.floats(b.o, std::size_t{n} * E) ||
        !b.floats(b.gate, std::size_t{n} * ff) ||
        !b.floats(b.up, std::size_t{n} * ff) ||
        !b.floats(b.scratch, widest * std::min<std::uint32_t>(n, 32)) ||
        !b.floats(b.logits, std::size_t{nh} * R) ||
        !b.floats(b.cs, cs.size()) ||
        !b.floats(b.stage, std::size_t{n} * stage_stride) ||
        !b.bytes(b.rows, rows.size() * sizeof(std::uint64_t)) ||
        !b.bytes(b.blocks, blocks.size() * sizeof(std::uint32_t)) ||
        !b.bytes(b.meta, meta.size() * sizeof(std::uint32_t))) {
        return false;
    }
    if (!cuda_dev_upload(b.x, xh.data(), xh.size()) ||
        !cuda_dev_upload(b.cs, cs.data(), cs.size()) ||
        !cuda_dev_upload_bytes(b.rows, rows.data(),
                               rows.size() * sizeof(std::uint64_t)) ||
        !cuda_dev_upload_bytes(b.blocks, blocks.data(),
                               blocks.size() * sizeof(std::uint32_t)) ||
        !cuda_dev_upload_bytes(b.meta, meta.data(),
                               meta.size() * sizeof(std::uint32_t))) {
        return false;
    }

    auto* mirror = const_cast<float*>(
        static_cast<const float*>(cache.mirror_data()));
    const float eps = hp_.rms_eps;
    // From here every exit goes through cuda_dev_finish, which unpins.
    bool ok = true;
    for (std::uint32_t l = 0; ok && l < L; ++l) {
        const Layer& lay = layers_[l];
        const float* an = cuda_dev_const(lay.attn_norm.data(), E);
        const float* fn = cuda_dev_const(lay.ffn_norm.data(), E);
        ok = an != nullptr && fn != nullptr &&
             cuda_dev_rmsnorm(b.x, an, eps, b.xb, E, n) &&
             cuda_dev_matvec_batch(lay.wq, b.xb, b.q, n, b.scratch) &&
             cuda_dev_matvec_batch(lay.wk, b.xb, b.k, n, b.scratch) &&
             cuda_dev_matvec_batch(lay.wv, b.xb, b.v, n, b.scratch) &&
             cuda_dev_rope(b.q, b.cs, H, hd, qd, n) &&
             cuda_dev_rope(b.k, b.cs, KV, hd, kvd, n) &&
             cuda_dev_kv_scatter(
                 b.k, b.v, n, static_cast<std::uint32_t>(kvd),
                 static_cast<const std::uint64_t*>(b.rows),
                 cache.layer_offset(l), cache.layer_floats() / 2, mirror,
                 b.stage, stage_stride, std::size_t{l} * 2 * kvd);
        if (!ok) {
            break;
        }
        KvAttend a;
        a.mirror = mirror;
        a.blocks = blocks.data();
        a.n_blocks = static_cast<std::uint32_t>(blocks.size());
        a.block_stride = cache.block_floats();
        a.layer_off = cache.layer_offset(l);
        a.v_off = cache.layer_floats() / 2;
        a.block_tokens = bt;
        a.n_heads = H;
        a.n_kv_heads = KV;
        a.head_dim = hd;
        a.kq_scale = hp_.kq_scale;
        a.n_q = n;
        a.q_pos = meta.data();
        a.q_table = meta.data() + n;
        a.q = b.q;
        a.out = b.att;
        a.q_stride = qd;
        a.device_io = true;
        a.dev_blocks = b.blocks;
        a.dev_meta = b.meta;
        ok = op.kv_attend(a) &&
             cuda_dev_matvec_batch(lay.wo, b.att, b.o, n, b.scratch) &&
             cuda_dev_add(b.x, b.o, std::size_t{n} * E) &&
             cuda_dev_rmsnorm(b.x, fn, eps, b.xb, E, n) &&
             cuda_dev_matvec_batch(lay.w_gate, b.xb, b.gate, n, b.scratch) &&
             cuda_dev_matvec_batch(lay.w_up, b.xb, b.up, n, b.scratch) &&
             cuda_dev_silu_mul(b.gate, b.up, std::size_t{n} * ff) &&
             cuda_dev_matvec_batch(lay.w_down, b.gate, b.o, n, b.scratch) &&
             cuda_dev_add(b.x, b.o, std::size_t{n} * E);
    }
    if (ok) {
        const float* on = cuda_dev_const(out_norm_.data(), E);
        ok = on != nullptr && cuda_dev_rmsnorm(b.x, on, eps, b.xb, E, n) &&
             (all_logits
                  ? cuda_dev_matvec_batch(head, b.xb, b.logits, n,
                                          b.scratch)
                  : cuda_dev_matvec(head, b.xb + std::size_t{n - 1} * E,
                                    b.logits));
    }
    ok = cuda_dev_finish() && ok;
    std::vector<float> stage(std::size_t{n} * stage_stride);
    // The whole head lands in logits as is; a row range lands in a
    // packed buffer, then at its rows.
    std::vector<float> packed(R == V ? 0 : std::size_t{nh} * R);
    if (!ok ||
        !cuda_dev_download(R == V ? logits.data() : packed.data(), b.logits,
                           std::size_t{nh} * R) ||
        !cuda_dev_download(stage.data(), b.stage, stage.size()) ||
        !cuda_dev_download(ws.xb.data(), b.xb + std::size_t{n - 1} * E, E)) {
        return false;
    }
    // The host pool stays the source of truth: the new rows go there too.
    for (std::uint32_t t = 0; t < n; ++t) {
        for (std::uint32_t l = 0; l < L; ++l) {
            const float* src =
                stage.data() + t * stage_stride + std::size_t{l} * 2 * kvd;
            std::memcpy(cache.k(*seqs[t], l, pos[t]), src,
                        kvd * sizeof(float));
            std::memcpy(cache.v(*seqs[t], l, pos[t]), src + kvd,
                        kvd * sizeof(float));
        }
    }
    const float ninf = -std::numeric_limits<float>::infinity();
    for (std::uint32_t t = 0; t < nh; ++t) {
        const std::span<float> row = logits.subspan(std::size_t{t} * V, V);
        if (R != V) {
            std::fill(row.begin(), row.begin() + lo, ninf);
            std::copy_n(packed.data() + std::size_t{t} * R, R,
                        row.begin() + lo);
            std::fill(row.begin() + lo + R, row.end(), ninf);
        }
        scale_logits(row.subspan(lo, R));
    }
    return true;
}

}  // namespace locus::model
