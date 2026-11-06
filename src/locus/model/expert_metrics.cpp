#include "locus/model/expert_metrics.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>

#include "locus/backend/variants.hpp"

namespace locus::model {

namespace {

/** The counters of one api, sized when the api first records. */
struct Table {
    std::string_view api;
    bool gpu = false;
    std::uint32_t n_layers = 0;
    std::uint32_t n_expert = 0;
    std::unique_ptr<std::atomic<std::uint64_t>[]> calls;
};

// One table per api; a process uses one GPU api and one CPU kernel set
// at most, plus a backend switch now and then.
constexpr std::size_t kTables = 8;

struct Registry {
    std::array<Table, kTables> tables;
    std::atomic<std::size_t> ready{0};  // tables[0, ready) are complete
    std::mutex mu;                      // serializes adding a table
};

Registry& registry() {
    static Registry r;
    return r;
}

Table* find(Registry& r, std::string_view api, bool gpu, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        Table& t = r.tables[i];
        if (t.gpu == gpu && t.api == api) return &t;
    }
    return nullptr;
}

}  // namespace

void ExpertCalls::record(std::uint32_t layer, std::uint32_t expert,
                         std::uint32_t n_layers, std::uint32_t n_expert,
                         bool gpu, std::string_view api) {
    Registry& r = registry();
    Table* t = find(r, api, gpu, r.ready.load(std::memory_order_acquire));
    if (t == nullptr) {
        std::lock_guard<std::mutex> lk(r.mu);
        const std::size_t n = r.ready.load(std::memory_order_relaxed);
        t = find(r, api, gpu, n);
        if (t == nullptr) {
            if (n == kTables || n_layers == 0 || n_expert == 0) return;
            t = &r.tables[n];
            t->api = api;
            t->gpu = gpu;
            t->n_layers = n_layers;
            t->n_expert = n_expert;
            const std::size_t cells =
                static_cast<std::size_t>(n_layers) * n_expert;
            t->calls = std::make_unique<std::atomic<std::uint64_t>[]>(cells);
            for (std::size_t i = 0; i < cells; ++i) {
                t->calls[i].store(0, std::memory_order_relaxed);
            }
            r.ready.store(n + 1, std::memory_order_release);
        }
    }
    if (layer >= t->n_layers || expert >= t->n_expert) return;
    t->calls[static_cast<std::size_t>(layer) * t->n_expert + expert]
        .fetch_add(1, std::memory_order_relaxed);
}

std::vector<ExpertCalls::Series> ExpertCalls::snapshot() {
    Registry& r = registry();
    std::vector<Series> out;
    const std::size_t n = r.ready.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        const Table& t = r.tables[i];
        for (std::uint32_t l = 0; l < t.n_layers; ++l) {
            for (std::uint32_t e = 0; e < t.n_expert; ++e) {
                const std::uint64_t c =
                    t.calls[static_cast<std::size_t>(l) * t.n_expert + e]
                        .load(std::memory_order_relaxed);
                if (c > 0) {
                    out.push_back({l, e, t.gpu ? "gpu" : "cpu", t.api, c});
                }
            }
        }
    }
    return out;
}

void ExpertCalls::reset() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    const std::size_t n = r.ready.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < n; ++i) r.tables[i] = Table{};
    r.ready.store(0, std::memory_order_release);
}

std::string_view expert_api(std::string_view backend, bool& gpu) {
    gpu = backend == "cuda" || backend == "vulkan";
    if (backend == "vulkan") return backend::vulkan_api_name();
    return backend;
}

}  // namespace locus::model
