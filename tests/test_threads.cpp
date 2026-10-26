#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/model/gqa_attend.hpp"
#include "locus/model/transformer.hpp"
#include "locus/sys/thread_pool.hpp"

using locus::sys::ThreadPool;

TEST_CASE("parallel_for covers every index exactly once",
          "[threads]") {
    auto& pool = ThreadPool::instance();
    REQUIRE(pool.parallelism() >= 1);
    // Two rounds through the same pool to exercise job reuse.
    for (int round = 0; round < 2; ++round) {
        constexpr std::size_t n = 100;
        std::vector<std::atomic<int>> hits(n);
        pool.parallel_for(
            n, [&](std::size_t i) { hits[i].fetch_add(1); });
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(hits[i].load() == 1);
        }
    }
}

TEST_CASE("parallel_for wakes only the workers a job needs",
          "[threads]") {
    // Jobs smaller than the pool wake n - 1 workers (the caller drains
    // too); thousands of back-to-back tiny jobs must neither lose an
    // index nor hang on a wakeup that went to a worker still busy.
    auto& pool = ThreadPool::instance();
    for (std::size_t n : {2u, 3u, 5u}) {
        for (int round = 0; round < 2000; ++round) {
            std::vector<std::atomic<int>> hits(n);
            pool.parallel_for(
                n, [&](std::size_t i) { hits[i].fetch_add(1); });
            for (std::size_t i = 0; i < n; ++i) {
                REQUIRE(hits[i].load() == 1);
            }
        }
    }
}

TEST_CASE("parallel_for rethrows a worker exception",
          "[threads]") {
    auto& pool = ThreadPool::instance();
    REQUIRE_THROWS_AS(
        pool.parallel_for(8,
                          [&](std::size_t i) {
                              if (i == 5) {
                                  throw std::runtime_error("x");
                              }
                          }),
        std::runtime_error);
    // The pool must stay usable afterwards.
    std::atomic<int> ok{0};
    pool.parallel_for(4, [&](std::size_t) { ok.fetch_add(1); });
    REQUIRE(ok.load() == 4);
}

TEST_CASE("matvec_mt is bit-identical to the plain matvec",
          "[threads]") {
    using locus::backend::Mat;
    constexpr std::uint32_t rows = 301;  // not a slice multiple
    constexpr std::uint32_t cols = 32;
    std::vector<float> w(static_cast<std::size_t>(rows) * cols);
    std::vector<float> x(cols);
    std::uint32_t s = 12345;
    auto rnd = [&s] {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>(s >> 8) /
                    static_cast<float>(1u << 24) -
                0.5f);
    };
    for (auto& v : w) {
        v = rnd();
    }
    for (auto& v : x) {
        v = rnd();
    }
    Mat m;
    m.type = locus::gguf::TensorType::kF32;
    m.data = reinterpret_cast<const std::byte*>(w.data());
    m.rows = rows;
    m.cols = cols;

    const auto& op =
        locus::backend::find_backend("scalar")->ops;
    std::vector<float> ref(rows), got(rows);
    op.matvec(m, x, ref);

    for (const char* t : {"1", "2", "5"}) {
        setenv("LOCUS_THREADS", t, 1);
        std::fill(got.begin(), got.end(), 0.0f);
        locus::model::matvec_mt(op, m, x, got);
        REQUIRE(got == ref);
    }
    unsetenv("LOCUS_THREADS");
    std::fill(got.begin(), got.end(), 0.0f);
    locus::model::matvec_mt(op, m, x, got);
    REQUIRE(got == ref);
}

TEST_CASE("weights_stream: madvise readahead only for models that "
          "may not stay resident",
          "[threads]") {
    using locus::model::weights_stream;
    constexpr std::uint64_t gib = 1ull << 30;
    REQUIRE_FALSE(weights_stream(2 * gib, 36 * gib));  // 3B on 36 GB
    REQUIRE_FALSE(weights_stream(18 * gib, 36 * gib));  // exactly half
    REQUIRE(weights_stream(19 * gib, 36 * gib));
    REQUIRE(weights_stream(250 * gib, 36 * gib));  // GLM-class stream
    REQUIRE(weights_stream(1, 0));  // RAM unknown: keep the old default
}

TEST_CASE("matvec_mt_group is bit-identical to one matvec_mt per job",
          "[threads]") {
    using locus::backend::Mat;
    using locus::model::MatvecJob;
    // Q4_K (a matvec_q8k type) and F32 jobs of uneven heights sharing
    // one x, like q / k / v.
    constexpr std::uint32_t cols = 512;
    std::uint32_t s = 4242;
    auto next = [&s] {
        s = s * 1664525u + 1013904223u;
        return s;
    };
    std::vector<std::byte> wq(static_cast<std::size_t>(300) * cols / 256 *
                              144);
    for (auto& b : wq) {
        b = static_cast<std::byte>(next() >> 24);
    }
    for (std::size_t b = 0; b < wq.size() / 144; ++b) {
        wq[b * 144 + 1] = std::byte{0x20};
        wq[b * 144 + 3] = std::byte{0x20};
    }
    std::vector<float> wf(static_cast<std::size_t>(70) * cols), x(cols);
    for (auto& v : wf) {
        v = static_cast<float>(next() >> 8) / (1u << 24) - 0.5f;
    }
    for (auto& v : x) {
        v = static_cast<float>(next() >> 8) / (1u << 24) - 0.5f;
    }
    const Mat a{locus::gguf::TensorType::kQ4_K, wq.data(), 300, cols};
    const Mat b{locus::gguf::TensorType::kF32,
                reinterpret_cast<const std::byte*>(wf.data()), 70, cols};
    const auto& op = locus::backend::find_backend("scalar")->ops;
    std::vector<float> ra(a.rows), rb(b.rows);
    locus::model::matvec_mt(op, a, x, ra);
    locus::model::matvec_mt(op, b, x, rb);
    for (const char* t : {"1", "3", "8"}) {
        setenv("LOCUS_THREADS", t, 1);
        std::vector<float> ga(a.rows), gb(b.rows);
        const MatvecJob jobs[] = {{a, x, ga}, {b, x, gb}};
        locus::model::matvec_mt_group(op, jobs);
        REQUIRE(ga == ra);
        REQUIRE(gb == rb);
    }
    unsetenv("LOCUS_THREADS");
}

TEST_CASE("gqa_attend is bit-identical across thread counts",
          "[threads]") {
    // 6 query heads over 2 KV heads (group 3), 37 positions; min_work 0
    // forces the pool path, LOCUS_THREADS 1 the inline one.
    constexpr std::uint32_t heads = 6, group = 3, hd = 20, pos = 36;
    constexpr std::uint32_t kv = heads / group * hd;
    std::uint32_t s = 99;
    auto rnd = [&s] {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / (1u << 24) - 0.5f;
    };
    std::vector<float> q(heads * hd), k((pos + 1) * kv), v((pos + 1) * kv);
    for (auto* vec : {&q, &k, &v}) {
        for (auto& x : *vec) {
            x = rnd();
        }
    }
    auto krow = [&](std::uint32_t t) { return k.data() + t * kv; };
    auto vrow = [&](std::uint32_t t) { return v.data() + t * kv; };
    auto run = [&](const char* threads) {
        setenv("LOCUS_THREADS", threads, 1);
        std::vector<float> out(heads * hd, -1.0f);
        locus::model::gqa_attend(q.data(), out.data(), heads, hd, group,
                                 pos, 0.25f, krow, vrow, /*min_work=*/0);
        return out;
    };
    const auto ref = run("1");
    for (const char* t : {"2", "4", "16"}) {
        REQUIRE(run(t) == ref);
    }
    unsetenv("LOCUS_THREADS");
    // Head 0 against a direct evaluation of its softmax-weighted sum.
    std::vector<double> sc(pos + 1);
    double mx = -1e30, sum = 0.0;
    for (std::uint32_t t = 0; t <= pos; ++t) {
        double d = 0.0;
        for (std::uint32_t i = 0; i < hd; ++i) {
            d += q[i] * krow(t)[i];
        }
        sc[t] = d * 0.25;
        mx = std::max(mx, sc[t]);
    }
    for (auto& x : sc) {
        x = std::exp(x - mx);
        sum += x;
    }
    for (std::uint32_t i = 0; i < hd; ++i) {
        double o = 0.0;
        for (std::uint32_t t = 0; t <= pos; ++t) {
            o += sc[t] / sum * vrow(t)[i];
        }
        REQUIRE(ref[i] == Catch::Approx(o).margin(1e-5));
    }
}
