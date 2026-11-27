#include "locus/model/expert_metrics.hpp"

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "locus/backend/variants.hpp"

namespace locus::model {

namespace {

thread_local RunPhase t_phase = RunPhase::kPrefill;

constexpr std::size_t kRunPhases = 3;

// Tokens a run serves: one decode token up to a full prefill chunk
// (i#1219, the step histograms' bounds).
constexpr double kRunTokenBounds[] = {1,  2,   4,   8,   16,   32,
                                      64, 128, 256, 512, 1024, 2048};

using Cells = std::unique_ptr<std::atomic<std::uint64_t>[]>;

Cells zeroed(std::size_t n) {
    Cells c = std::make_unique<std::atomic<std::uint64_t>[]>(n);
    for (std::size_t i = 0; i < n; ++i) {
        c[i].store(0, std::memory_order_relaxed);
    }
    return c;
}

/** The counters of one api, sized when the api first records. */
struct Table {
    std::string_view api;
    bool gpu = false;
    std::uint32_t n_layers = 0;
    std::uint32_t n_expert = 0;
    Cells calls;
    Cells runs;     ///< i#1219: weight passes
    Cells run_ns;   ///< their host-clock time
    Cells disk;     ///< bytes not in RAM when picked
    /** Tokens per run, a histogram a RunPhase. */
    std::unique_ptr<std::array<sys::Histogram, kRunPhases>> tokens;
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

/** The api's table, made on first use; nullptr when none can be. */
Table* table(std::string_view api, bool gpu, std::uint32_t n_layers,
             std::uint32_t n_expert) {
    Registry& r = registry();
    Table* t = find(r, api, gpu, r.ready.load(std::memory_order_acquire));
    if (t != nullptr) return t;
    std::lock_guard<std::mutex> lk(r.mu);
    const std::size_t n = r.ready.load(std::memory_order_relaxed);
    t = find(r, api, gpu, n);
    if (t != nullptr) return t;
    if (n == kTables || n_layers == 0 || n_expert == 0) return nullptr;
    t = &r.tables[n];
    t->api = api;
    t->gpu = gpu;
    t->n_layers = n_layers;
    t->n_expert = n_expert;
    const std::size_t cells = static_cast<std::size_t>(n_layers) * n_expert;
    t->calls = zeroed(cells);
    t->runs = zeroed(cells);
    t->run_ns = zeroed(cells);
    t->disk = zeroed(cells);
    // Built in place: a Histogram (atomics) does not copy or move.
    t->tokens.reset(new std::array<sys::Histogram, kRunPhases>{
        sys::Histogram(kRunTokenBounds, 1),
        sys::Histogram(kRunTokenBounds, 1),
        sys::Histogram(kRunTokenBounds, 1)});
    r.ready.store(n + 1, std::memory_order_release);
    return t;
}

/** @returns The cell of (layer, expert), or -1 past the table. */
std::ptrdiff_t cell(const Table& t, std::uint32_t layer,
                    std::uint32_t expert) {
    if (layer >= t.n_layers || expert >= t.n_expert) return -1;
    return static_cast<std::ptrdiff_t>(layer) * t.n_expert + expert;
}

}  // namespace

RunPhase run_phase() { return t_phase; }

RunPhaseScope::RunPhaseScope(RunPhase p) : prev_(t_phase) { t_phase = p; }

RunPhaseScope::~RunPhaseScope() { t_phase = prev_; }

const char* run_phase_name(RunPhase p) {
    switch (p) {
        case RunPhase::kPrefill: return "prefill";
        case RunPhase::kDecode: return "decode";
        case RunPhase::kVerify: return "verify";
    }
    return "";
}

void ExpertCalls::record(std::uint32_t layer, std::uint32_t expert,
                         std::uint32_t n_layers, std::uint32_t n_expert,
                         bool gpu, std::string_view api) {
    Table* t = table(api, gpu, n_layers, n_expert);
    if (t == nullptr) return;
    const std::ptrdiff_t c = cell(*t, layer, expert);
    if (c < 0) return;
    t->calls[c].fetch_add(1, std::memory_order_relaxed);
}

void ExpertCalls::record_run(std::uint32_t layer, std::uint32_t expert,
                             std::uint32_t n_layers,
                             std::uint32_t n_expert, bool gpu,
                             std::string_view api, std::uint32_t tokens,
                             std::uint64_t ns) {
    if (!timed(api)) return;
    Table* t = table(api, gpu, n_layers, n_expert);
    if (t == nullptr) return;
    const std::ptrdiff_t c = cell(*t, layer, expert);
    if (c < 0) return;
    t->runs[c].fetch_add(1, std::memory_order_relaxed);
    t->run_ns[c].fetch_add(ns, std::memory_order_relaxed);
    (*t->tokens)[static_cast<std::size_t>(t_phase)].observe(tokens);
}

void ExpertCalls::record_disk(std::uint32_t layer, std::uint32_t expert,
                              std::uint32_t n_layers,
                              std::uint32_t n_expert, bool gpu,
                              std::string_view api, std::uint64_t bytes) {
    if (bytes == 0) return;
    Table* t = table(api, gpu, n_layers, n_expert);
    if (t == nullptr) return;
    const std::ptrdiff_t c = cell(*t, layer, expert);
    if (c < 0) return;
    t->disk[c].fetch_add(bytes, std::memory_order_relaxed);
}

bool ExpertCalls::timed(std::string_view api) {
    return api != "vulkan" && api != "moltenvk" && api != "metal";
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

std::vector<ExpertCalls::RunSeries> ExpertCalls::run_snapshot() {
    Registry& r = registry();
    std::vector<RunSeries> out;
    const std::size_t n = r.ready.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        const Table& t = r.tables[i];
        for (std::uint32_t l = 0; l < t.n_layers; ++l) {
            for (std::uint32_t e = 0; e < t.n_expert; ++e) {
                const std::size_t c =
                    static_cast<std::size_t>(l) * t.n_expert + e;
                const std::uint64_t runs =
                    t.runs[c].load(std::memory_order_relaxed);
                if (runs > 0) {
                    out.push_back({l, e, t.gpu ? "gpu" : "cpu", t.api, runs,
                                   t.run_ns[c].load(
                                       std::memory_order_relaxed)});
                }
            }
        }
    }
    return out;
}

std::vector<ExpertCalls::DiskSeries> ExpertCalls::disk_snapshot() {
    Registry& r = registry();
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> sum;
    const std::size_t n = r.ready.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        const Table& t = r.tables[i];
        for (std::uint32_t l = 0; l < t.n_layers; ++l) {
            for (std::uint32_t e = 0; e < t.n_expert; ++e) {
                const std::uint64_t b =
                    t.disk[static_cast<std::size_t>(l) * t.n_expert + e]
                        .load(std::memory_order_relaxed);
                if (b > 0) sum[{l, e}] += b;
            }
        }
    }
    std::vector<DiskSeries> out;
    for (const auto& [k, b] : sum) out.push_back({k.first, k.second, b});
    return out;
}

std::vector<ExpertCalls::RunTokens> ExpertCalls::run_tokens() {
    Registry& r = registry();
    std::vector<RunTokens> out;
    const std::size_t n = r.ready.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        const Table& t = r.tables[i];
        for (std::size_t p = 0; p < kRunPhases; ++p) {
            auto s = (*t.tokens)[p].snapshot();
            if (s.count > 0) {
                out.push_back({t.gpu ? "gpu" : "cpu", t.api,
                               static_cast<RunPhase>(p), std::move(s)});
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
