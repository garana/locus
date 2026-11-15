#include "locus/backend/accelerate.hpp"

#if defined(LOCUS_HAS_ACCELERATE)
// The CBLAS interface of macOS 13.3 and later (the older one is
// deprecated).
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <vector>

#include "locus/sys/thread_pool.hpp"
#endif

namespace locus::backend {

bool accelerate_available() {
#if defined(LOCUS_HAS_ACCELERATE)
    return true;
#else
    return false;
#endif
}

bool sgemm_prefill_on() { return accelerate_available() && cpu_fast(); }

bool sgemm_batch(const Mat& w, std::span<const float> x,
                 std::span<float> out, std::uint32_t n) {
#if defined(LOCUS_HAS_ACCELERATE)
    if (n == 0 || w.rows == 0) {
        return true;
    }
    const std::size_t cols = w.cols;
    // out (n x rows, row-major) = x (n x cols) . w^T (cols x rows).
    const auto gemm = [&](const float* wf, std::uint32_t r0,
                          std::uint32_t nr) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<__LAPACK_int>(n),
                    static_cast<__LAPACK_int>(nr),
                    static_cast<__LAPACK_int>(cols), 1.0f, x.data(),
                    static_cast<__LAPACK_int>(cols), wf,
                    static_cast<__LAPACK_int>(cols), 0.0f, out.data() + r0,
                    static_cast<__LAPACK_int>(w.rows));
    };
    if (w.type == gguf::TensorType::kF32) {
        gemm(reinterpret_cast<const float*>(w.data), 0, w.rows);
        return true;
    }
    // Rows in blocks of at most 4 Mi floats (16 MB), dequantized over the
    // thread pool, then one sgemm each.
    constexpr std::size_t kBlockFloats = std::size_t{4} << 20;
    const auto block = static_cast<std::uint32_t>(
        std::clamp<std::size_t>(kBlockFloats / std::max<std::size_t>(cols, 1),
                                1, w.rows));
    thread_local std::vector<float> wf;
    wf.resize(std::size_t{block} * cols);
    // The caller's buffer: inside the lambda, `wf` would name each pool
    // thread's own thread_local.
    float* const buf = wf.data();
    for (std::uint32_t r0 = 0; r0 < w.rows; r0 += block) {
        const std::uint32_t nr = std::min(block, w.rows - r0);
        sys::ThreadPool::instance().parallel_for(nr, [&](std::size_t i) {
            dequant_row(w, r0 + static_cast<std::uint32_t>(i),
                        {buf + i * cols, cols});
        });
        gemm(buf, r0, nr);
    }
    return true;
#else
    (void)w;
    (void)x;
    (void)out;
    (void)n;
    return false;
#endif
}

}  // namespace locus::backend
