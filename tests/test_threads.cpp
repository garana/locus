#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/model/llama.hpp"
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
