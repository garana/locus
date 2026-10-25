#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
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

TEST_CASE("matvec_mt_f32 threads the f32-activation dot under R16",
          "[threads]") {
    using locus::backend::Mat;
    // Q4_K, a type matvec_mt would send to matvec_q8k when it is set.
    constexpr std::uint32_t rows = 301;
    constexpr std::uint32_t cols = 512;
    std::vector<std::byte> w(static_cast<std::size_t>(rows) * cols /
                             256 * 144);
    std::uint32_t s = 777;
    for (auto& b : w) {
        s = s * 1664525u + 1013904223u;
        b = static_cast<std::byte>(s >> 24);
    }
    for (std::size_t b = 0; b < w.size() / 144; ++b) {
        w[b * 144 + 1] = std::byte{0x20};  // d, dmin: small normals
        w[b * 144 + 3] = std::byte{0x20};
    }
    std::vector<float> x(cols);
    for (auto& v : x) {
        s = s * 1664525u + 1013904223u;
        v = static_cast<float>(s >> 8) /
                static_cast<float>(1u << 24) - 0.5f;
    }
    Mat m;
    m.type = locus::gguf::TensorType::kQ4_K;
    m.data = w.data();
    m.rows = rows;
    m.cols = cols;

    const auto& op = locus::backend::find_backend("scalar")->ops;
    REQUIRE(op.matvec_q8k != nullptr);
    std::vector<float> ref(rows), q8(rows), got(rows);
    op.matvec(m, x, ref);
    op.matvec_q8k(m, x, q8);
    REQUIRE(q8 != ref);  // the two dots differ, so the test can tell
    for (const char* t : {"1", "5"}) {
        setenv("LOCUS_THREADS", t, 1);
        locus::model::matvec_mt_f32(op, m, x, got);
        REQUIRE(got == ref);
        locus::model::matvec_mt(op, m, x, got);
        REQUIRE(got == q8);
    }
    unsetenv("LOCUS_THREADS");
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
    for (const bool f32 : {false, true}) {
        INFO("f32 " << f32);
        std::vector<float> ra(a.rows), rb(b.rows);
        (f32 ? locus::model::matvec_mt_f32 : locus::model::matvec_mt)(
            op, a, x, ra);
        locus::model::matvec_mt(op, b, x, rb);
        for (const char* t : {"1", "3", "8"}) {
            setenv("LOCUS_THREADS", t, 1);
            std::vector<float> ga(a.rows), gb(b.rows);
            const MatvecJob jobs[] = {{a, x, ga}, {b, x, gb}};
            locus::model::matvec_mt_group(op, jobs, f32);
            REQUIRE(ga == ra);
            REQUIRE(gb == rb);
        }
    }
    unsetenv("LOCUS_THREADS");
}
