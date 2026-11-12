#include "locus/engine/disk_prefix_key.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "locus/backend/variants.hpp"

namespace locus::engine {

namespace {

constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr std::uint64_t kFnvPrime = 0x100000001b3ull;

/** FNV-1a over raw bytes. */
std::uint64_t fnv(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * kFnvPrime;
    }
    return h;
}

template <typename T>
std::uint64_t fnv(std::uint64_t h, const T& v) {
    return fnv(h, &v, sizeof v);
}

}  // namespace

DiskKey disk_key_for(const gguf::GgufFile& g, const std::string& path,
                     const model::TransformerModel& m, kv::KvType kv_type,
                     std::size_t& block_bytes) {
    namespace fs = std::filesystem;
    DiskKey k;
    const auto mtime = static_cast<std::int64_t>(
        fs::last_write_time(path).time_since_epoch().count());
    k.model_size = fs::file_size(path);
    std::uint64_t h = fnv(fnv(kFnvOffset, mtime), k.model_size);
    // Every shard of a split model: its header, tables and size.
    g.for_each_file([&](const gguf::GgufFile& f) {
        const auto head = f.header_bytes();
        h = fnv(fnv(h, head.data(), head.size()), f.file_bytes());
    });
    k.model_hash = h;
    k.backend = m.active_backend().name;

    const auto& hp = m.hparams();
    std::uint64_t o = kFnvOffset;
    // Numeric modes that change K and V bits: activations quantized to
    // Q8_K before the K/V matvecs, and on Vulkan f16 prefill staging
    // and the batch-invariant kernels. The device too: two GPUs on one
    // backend do not round alike.
    o = fnv(o, m.q8k_activations());
    // The MoE hybrid as decided, not only its variable: unset turns it
    // on where the weights do not fit (i#652), and its cold experts
    // round as the CPU does.
    o = fnv(o, m.moe_hybrid_on());
    // Debug and offload switches read from the environment that pick
    // other kernels or another expert split: their raw values.
    for (const char* var : {"LOCUS_BATCH_DEQUANT", "LOCUS_MOE_HYBRID",
                            "LOCUS_MOE_HYBRID_SLOTS", "LOCUS_VK_CPU_ROUTE",
                            "LOCUS_GPU_POOL_MB", "LOCUS_MOE_STATS",
                            "LOCUS_CUDA_FORWARD"}) {
        const char* v = std::getenv(var);
        const std::string s = v != nullptr ? std::string("=") + v : "";
        o = fnv(fnv(o, var, std::strlen(var)), s.data(), s.size());
    }
    if (k.backend == "vulkan") {
        o = fnv(o, backend::vulkan_prefill_f16());
        o = fnv(o, backend::vulkan_batch_invariant());
        const std::string dev = backend::vulkan_device_name();
        o = fnv(o, dev.data(), dev.size());
    } else if (k.backend == "cuda") {
        const std::string dev = backend::cuda_device_name();
        o = fnv(o, dev.data(), dev.size());
    }
    o = fnv(o, hp.n_ctx);
    o = fnv(o, hp.rope_freq_base);
    o = fnv(o, hp.rope_freq_base_swa);
    o = fnv(o, hp.kq_scale);
    o = fnv(o, hp.yarn.freq_scale);
    o = fnv(o, hp.yarn.mscale);
    o = fnv(o, hp.yarn.n_ctx_orig);
    o = fnv(o, hp.yarn.beta_fast);
    o = fnv(o, hp.yarn.beta_slow);
    k.overrides = o;

    // One block is enough to read the geometry the engine will use.
    const kv::PagedKvCache probe = m.make_cache(1, kv_type);
    const kv::PagedKvCache::Geometry& geo = probe.geometry();
    k.kv_type = geo.kv_type;
    k.block_tokens = geo.block_tokens;
    k.kv_dim = geo.kv_dim;
    k.n_layers = geo.n_layers;
    k.layer_base = geo.layer_base;
    block_bytes = probe.block_bytes();
    return k;
}

std::shared_ptr<DiskPrefixStore> open_disk_prefix(
    const std::string& dir, std::uint64_t cap_bytes,
    const gguf::GgufFile& g, const std::string& path,
    const model::TransformerModel& m, kv::KvType kv_type) {
    const std::string_view be = m.active_backend().name;
    if (m.hparams().n_expert > 0 && (be == "vulkan" || be == "cuda")) {
        // On a GPU the expert route (device or host) also depends on
        // the free device memory at startup, which no key can follow.
        throw std::invalid_argument(
            "--prefix-cache-dir: not supported for MoE models on a GPU "
            "backend yet");
    }
    std::size_t bb = 0;
    DiskKey k = disk_key_for(g, path, m, kv_type, bb);
    return std::make_shared<DiskPrefixStore>(dir, std::move(k), bb,
                                             cap_bytes);
}

}  // namespace locus::engine
