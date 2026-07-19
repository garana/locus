#include <bit>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "cppllm/backend/variants.hpp"
#include "cppllm/backend/vulkan/context.hpp"
#include "cppllm/backend/vulkan_forward.hpp"

namespace cppllm::backend {

namespace {

using vk::Kernel;
using vk::VulkanContext;

/** @returns On-disk bytes of one weight matrix, 4-byte padded. */
std::size_t weight_bytes(const Mat& w) {
    std::size_t b;
    switch (w.type) {
        case gguf::TensorType::kF32:
            b = static_cast<std::size_t>(w.rows) * w.cols * 4;
            break;
        case gguf::TensorType::kF16:
            b = static_cast<std::size_t>(w.rows) * w.cols * 2;
            break;
        case gguf::TensorType::kQ8_0:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 32) * 34;
            break;
        case gguf::TensorType::kQ4_0:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 32) * 18;
            break;
        case gguf::TensorType::kQ4_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 144;
            break;
        case gguf::TensorType::kQ5_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 176;
            break;
        case gguf::TensorType::kQ6_K:
            b = static_cast<std::size_t>(w.rows) *
                (w.cols / 256) * 210;
            break;
        default:
            return 0;  // no GPU kernel for this type
    }
    return (b + 3) & ~std::size_t{3};
}

/**
 * Process-lifetime GPU state: one context, resident buffers for
 * weights and norm vectors (keyed by their mmap pointers, stable
 * for the model's lifetime), grow-only activation scratch, and
 * the GPU-mapped KV pools handed to PagedKvCache.
 *
 * Single-threaded by contract: ops are called only from the
 * engine/model thread.
 */
struct State {
    VulkanContext ctx;
    std::unordered_map<const void*, VulkanContext::Buffer>
        resident;
    std::unordered_map<const void*, VulkanContext::Buffer>
        kv_pools;

    /** Activation buffers, sized on first use per model dims. */
    VulkanContext::Buffer x{}, xb{}, q{}, gate{}, up{}, attout{},
        att{}, logits{}, table{}, ones{};
    std::size_t x_n = 0, xb_n = 0, q_n = 0, gate_n = 0, up_n = 0,
                attout_n = 0, att_n = 0, logits_n = 0,
                table_n = 0, ones_n = 0;

    /** Rope divisor buffer: factors when scaled, else 1.0s. */
    VulkanContext::Buffer rope_divisors(
        std::span<const float> factors, std::uint32_t half_dim) {
        if (!factors.empty()) {
            return upload(factors.data(), factors.size() * 4);
        }
        if (half_dim > ones_n) {
            if (ones_n != 0) {
                ctx.destroy_buffer(ones);
            }
            ones = ctx.create_buffer(half_dim * 4);
            auto* p = static_cast<float*>(ctx.mapped(ones));
            for (std::uint32_t i = 0; i < half_dim; ++i) {
                p[i] = 1.0f;
            }
            ones_n = half_dim;
        }
        return ones;
    }

    VulkanContext::Buffer upload(const void* ptr,
                                 std::size_t bytes) {
        auto it = resident.find(ptr);
        if (it != resident.end()) {
            return it->second;
        }
        auto buf = ctx.create_buffer(bytes);
        ctx.write_buffer(
            buf, {static_cast<const std::byte*>(ptr), bytes});
        resident.emplace(ptr, buf);
        return buf;
    }

    void grow(VulkanContext::Buffer& b, std::size_t& cap,
              std::size_t n_floats) {
        if (n_floats > cap) {
            if (cap != 0) {
                ctx.destroy_buffer(b);
            }
            b = ctx.create_buffer(n_floats * 4);
            cap = n_floats;
        }
    }
};

State& state() {
    static State s;
    return s;
}

/** @returns The matvec kernel for w, or kCount_ if unsupported. */
Kernel matvec_kernel(const Mat& w) {
    switch (w.type) {
        case gguf::TensorType::kF32: return Kernel::kMatvecF32;
        case gguf::TensorType::kF16: return Kernel::kMatvecF16;
        case gguf::TensorType::kQ8_0: return Kernel::kMatvecQ8_0;
        case gguf::TensorType::kQ4_0: return Kernel::kMatvecQ4_0;
        case gguf::TensorType::kQ4_K: return Kernel::kMatvecQ4_K;
        case gguf::TensorType::kQ5_K: return Kernel::kMatvecQ5_K;
        case gguf::TensorType::kQ6_K: return Kernel::kMatvecQ6_K;
        default: return Kernel::kCount_;
    }
}

/** Records one matvec dispatch into the open batch. */
void rec_matvec(State& s, const Mat& w, VulkanContext::Buffer x,
                VulkanContext::Buffer out,
                std::uint32_t out_offset, bool accumulate) {
    const VulkanContext::Buffer bufs[] = {
        s.upload(w.data, weight_bytes(w)), x, out};
    const std::uint32_t push[] = {w.rows, w.cols, out_offset,
                                  accumulate ? 1u : 0u};
    s.ctx.dispatch(matvec_kernel(w), bufs, push,
                   (w.rows + 63) / 64);
}

}  // namespace

bool vulkan_backend_usable() {
    static const bool ok = VulkanContext::available();
    return ok;
}

float* vulkan_alloc_kv(std::size_t n_floats) {
    State& s = state();
    auto buf = s.ctx.create_buffer(n_floats * 4);
    void* map = s.ctx.mapped(buf);
    std::memset(map, 0, n_floats * 4);
    s.kv_pools.emplace(map, buf);
    return static_cast<float*>(map);
}

void matvec_vulkan(const Mat& w, std::span<const float> x,
                   std::span<float> out) {
    if (matvec_kernel(w) == Kernel::kCount_) {
        matvec(w, x, out);  // no shader for this weight type
        return;
    }
    State& s = state();
    auto wb = s.upload(w.data, weight_bytes(w));
    s.grow(s.x, s.x_n, x.size());
    s.grow(s.xb, s.xb_n, out.size());
    s.ctx.write_buffer(s.x, std::as_bytes(x));
    s.ctx.begin_batch();
    const VulkanContext::Buffer bufs[] = {wb, s.x, s.xb};
    const std::uint32_t push[] = {w.rows, w.cols, 0, 0};
    s.ctx.dispatch(matvec_kernel(w), bufs, push,
                   (w.rows + 63) / 64);
    s.ctx.end_batch();
    s.ctx.read_buffer(s.xb, std::as_writable_bytes(out));
}

bool vulkan_forward(const model::LlamaModel& m, tok::TokenId token,
                    kv::PagedKvCache& cache,
                    kv::PagedKvCache::Seq& seq,
                    std::span<float> logits) {
    if (m.hparams().n_expert > 0) {
        // MoE routing is CPU-side for now (R4); the hybrid op
        // path still runs matmuls on the GPU.
        return false;
    }
    State& s = state();
    auto pool_it = s.kv_pools.find(cache.pool_data());
    if (pool_it == s.kv_pools.end()) {
        return false;  // cache pool is not GPU-mapped
    }
    for (const auto& l : m.layers()) {
        for (const Mat* w : {&l.wq, &l.wk, &l.wv, &l.wo,
                             &l.w_gate, &l.w_up, &l.w_down}) {
            if (matvec_kernel(*w) == Kernel::kCount_) {
                return false;
            }
        }
    }
    if (matvec_kernel(m.output_weight()) == Kernel::kCount_) {
        return false;
    }

    const auto& hp = m.hparams();
    if (seq.n_tokens >= hp.n_ctx ||
        seq.n_tokens >= cache.capacity(seq)) {
        throw std::invalid_argument("seq capacity not ensured");
    }
    const std::uint32_t pos = seq.n_tokens;
    const auto& geom = cache.geometry();
    const std::uint32_t kv_dim = geom.kv_dim;
    const std::uint32_t layer_stride =
        geom.block_tokens * kv_dim * 2;
    const std::uint32_t block_stride =
        hp.n_layers * layer_stride;
    const std::uint32_t group = hp.n_heads / hp.n_kv_heads;

    s.grow(s.x, s.x_n, hp.n_embd);
    s.grow(s.xb, s.xb_n,
           std::max<std::size_t>(hp.n_embd, hp.n_ff));
    s.grow(s.q, s.q_n, hp.n_embd);
    s.grow(s.gate, s.gate_n, hp.n_ff);
    s.grow(s.up, s.up_n, hp.n_ff);
    s.grow(s.attout, s.attout_n, hp.n_embd);
    s.grow(s.att, s.att_n,
           static_cast<std::size_t>(hp.n_heads) * hp.n_ctx);
    s.grow(s.logits, s.logits_n, hp.n_vocab);
    s.grow(s.table, s.table_n, cache.total_blocks());

    // CPU side: embedding lookup and this token's block table.
    std::vector<float> x_host(hp.n_embd);
    dequant_row(m.embedding(), static_cast<std::uint32_t>(token),
                x_host);
    s.ctx.write_buffer(
        s.x, std::as_bytes(std::span<const float>(x_host)));
    s.ctx.write_buffer(
        s.table,
        std::as_bytes(std::span<const kv::BlockId>(seq.blocks)));

    const auto pool = pool_it->second;
    const float rope_base = hp.rope_freq_base;
    const std::uint32_t rope_bits =
        std::bit_cast<std::uint32_t>(rope_base);
    const std::uint32_t eps_bits =
        std::bit_cast<std::uint32_t>(hp.rms_eps);
    // K row of this position inside the pool, in floats.
    const std::uint32_t block =
        seq.blocks[pos / geom.block_tokens];
    const std::uint32_t in_block = pos % geom.block_tokens;

    s.ctx.begin_batch();
    for (std::uint32_t l = 0; l < hp.n_layers; ++l) {
        const auto& lay = m.layers()[l];
        const std::uint32_t k_off = block * block_stride +
                                    l * layer_stride +
                                    in_block * kv_dim;

        auto norm = [&](std::span<const float> w,
                        VulkanContext::Buffer in,
                        VulkanContext::Buffer out) {
            const VulkanContext::Buffer bufs[] = {
                in, s.upload(w.data(), w.size() * 4), out};
            const std::uint32_t push[] = {hp.n_embd, eps_bits};
            s.ctx.dispatch(Kernel::kRmsNorm, bufs, push, 1);
        };

        const VulkanContext::Buffer divisors =
            s.rope_divisors(m.rope_factors(), hp.head_dim / 2);
        norm(lay.attn_norm, s.x, s.xb);
        rec_matvec(s, lay.wq, s.xb, s.q, 0, false);
        rec_matvec(s, lay.wk, s.xb, pool, k_off, false);
        rec_matvec(s, lay.wv, s.xb, pool,
                   k_off + layer_stride / 2, false);
        {
            const VulkanContext::Buffer bufs[] = {s.q, divisors};
            const std::uint32_t push[] = {hp.n_heads,
                                          hp.head_dim, pos, 0,
                                          rope_bits};
            s.ctx.dispatch(Kernel::kRope, bufs, push,
                           hp.n_heads);
        }
        {
            const VulkanContext::Buffer bufs[] = {pool,
                                                  divisors};
            const std::uint32_t push[] = {hp.n_kv_heads,
                                          hp.head_dim, pos,
                                          k_off, rope_bits};
            s.ctx.dispatch(Kernel::kRope, bufs, push,
                           hp.n_kv_heads);
        }
        {
            const VulkanContext::Buffer bufs[] = {
                s.q, pool, s.table, s.att, s.attout};
            const std::uint32_t push[] = {
                hp.n_heads,   group,
                hp.head_dim,  kv_dim,
                pos + 1,      geom.block_tokens,
                block_stride, l * layer_stride,
                layer_stride / 2};
            s.ctx.dispatch(Kernel::kAttnPaged, bufs, push,
                           hp.n_heads);
        }
        rec_matvec(s, lay.wo, s.attout, s.x, 0, true);

        norm(lay.ffn_norm, s.x, s.xb);
        rec_matvec(s, lay.w_gate, s.xb, s.gate, 0, false);
        rec_matvec(s, lay.w_up, s.xb, s.up, 0, false);
        {
            const VulkanContext::Buffer bufs[] = {s.gate, s.up,
                                                  s.gate};
            const std::uint32_t push[] = {hp.n_ff};
            s.ctx.dispatch(Kernel::kSiluMul, bufs, push,
                           (hp.n_ff + 63) / 64);
        }
        rec_matvec(s, lay.w_down, s.gate, s.x, 0, true);
    }
    {
        const VulkanContext::Buffer bufs[] = {
            s.x, s.upload(m.output_norm().data(), hp.n_embd * 4),
            s.xb};
        const std::uint32_t push[] = {hp.n_embd, eps_bits};
        s.ctx.dispatch(Kernel::kRmsNorm, bufs, push, 1);
    }
    rec_matvec(s, m.output_weight(), s.xb, s.logits, 0, false);
    s.ctx.end_batch();

    s.ctx.read_buffer(s.logits, std::as_writable_bytes(logits));
    seq.n_tokens = pos + 1;
    return true;
}

}  // namespace cppllm::backend
