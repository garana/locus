#include "locus/image/dense.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include "locus/sys/thread_pool.hpp"

namespace locus::image::dense {

namespace {

/** Tokens per matvec_batch call: keeps the inputs in cache. */
constexpr std::uint32_t kTokenBlock = 64;

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

/**
 * out[r * n + t] = w[r] . x[t] for n token-major inputs: the backend's
 * matvec_batch layout (row-major output), or n matvec calls when the
 * backend has none.
 */
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

}  // namespace

void linear(const backend::Ops& op, const backend::Mat& w,
            std::span<const float> b, std::span<const float> x,
            std::uint32_t n, std::span<float> out) {
    const std::uint32_t rows = w.rows, cols = w.cols;
    const std::uint32_t threads =
        std::max(1u, std::thread::hardware_concurrency());
    const std::uint32_t slices =
        backend::splits_rows(op)
            ? std::max(1u, std::min(rows / 16, 2 * threads))
            : 1;
    const std::uint32_t per = (rows + slices - 1) / slices;
    // A self-parallel backend (a GPU) takes every token in one call;
    // a CPU one goes in blocks that keep the inputs in cache.
    const std::uint32_t block =
        op.self_parallel && op.matvec_batch != nullptr ? std::max(n, 1u)
                                                       : kTokenBlock;
    parallel(slices, [&](std::size_t s) {
        const std::uint32_t r0 = static_cast<std::uint32_t>(s) * per;
        if (r0 >= rows) return;
        const std::uint32_t nr = std::min(per, rows - r0);
        const backend::Mat ws = backend::mat_rows(w, r0, nr);
        std::vector<float> tmp(std::size_t{nr} * block);
        for (std::uint32_t t0 = 0; t0 < n; t0 += block) {
            const std::uint32_t nt = std::min(block, n - t0);
            mat_batch(op, ws, x.subspan(std::size_t{t0} * cols), nt, tmp);
            for (std::uint32_t t = 0; t < nt; ++t) {
                float* o = &out[std::size_t{t0 + t} * rows + r0];
                for (std::uint32_t r = 0; r < nr; ++r) {
                    o[r] = tmp[std::size_t{r} * nt + t] +
                           (b.empty() ? 0.0f : b[r0 + r]);
                }
            }
        }
    });
}

void attention(const backend::Ops& given, std::span<const float> q,
               std::uint32_t nq, std::span<const float> k,
               std::span<const float> v, std::uint32_t nk,
               std::uint32_t heads, std::uint32_t hd, std::span<float> out) {
    // The per-head K and V copies below are temporary host buffers. A
    // GPU backend caches weights by host pointer, so it must never see
    // them: such a backend (self-parallel, or not re-entrant across
    // the threads below) is replaced by the CPU backend.
    const backend::Ops& op = given.self_parallel || !given.mt_safe
                                 ? backend::best_backend().ops
                                 : given;
    const std::uint32_t w = heads * hd;
    std::vector<float> kh(std::size_t{heads} * nk * hd),
        vt(std::size_t{heads} * hd * nk);
    parallel(heads, [&](std::size_t h) {
        float* kd = &kh[h * nk * hd];
        float* vd = &vt[h * hd * nk];
        for (std::uint32_t j = 0; j < nk; ++j) {
            const float* ks = &k[std::size_t{j} * w + h * hd];
            const float* vs = &v[std::size_t{j} * w + h * hd];
            std::memcpy(kd + std::size_t{j} * hd, ks, hd * sizeof(float));
            for (std::uint32_t e = 0; e < hd; ++e) {
                vd[std::size_t{e} * nk + j] = vs[e];
            }
        }
    });
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const std::uint32_t qblocks = (nq + kTokenBlock - 1) / kTokenBlock;
    parallel(std::size_t{heads} * qblocks, [&](std::size_t item) {
        const std::uint32_t h = static_cast<std::uint32_t>(item / qblocks);
        const std::uint32_t q0 =
            static_cast<std::uint32_t>(item % qblocks) * kTokenBlock;
        const std::uint32_t m = std::min(kTokenBlock, nq - q0);
        const backend::Mat km{gguf::TensorType::kF32,
                              reinterpret_cast<const std::byte*>(
                                  &kh[std::size_t{h} * nk * hd]),
                              nk, hd};
        const backend::Mat vm{gguf::TensorType::kF32,
                              reinterpret_cast<const std::byte*>(
                                  &vt[std::size_t{h} * hd * nk]),
                              hd, nk};
        std::vector<float> qb(std::size_t{m} * hd), s(std::size_t{nk} * m),
            p(std::size_t{m} * nk), ob(std::size_t{hd} * m);
        for (std::uint32_t t = 0; t < m; ++t) {
            std::memcpy(&qb[std::size_t{t} * hd],
                        &q[std::size_t{q0 + t} * w + h * hd],
                        hd * sizeof(float));
        }
        mat_batch(op, km, qb, m, s);  // s[j * m + t]
        for (std::uint32_t t = 0; t < m; ++t) {
            float* pt = &p[std::size_t{t} * nk];
            float mx = -INFINITY;
            for (std::uint32_t j = 0; j < nk; ++j) {
                pt[j] = s[std::size_t{j} * m + t] * scale;
                mx = std::max(mx, pt[j]);
            }
            float sum = 0.0f;
            for (std::uint32_t j = 0; j < nk; ++j) {
                pt[j] = std::exp(pt[j] - mx);
                sum += pt[j];
            }
            const float inv = 1.0f / sum;
            for (std::uint32_t j = 0; j < nk; ++j) pt[j] *= inv;
        }
        mat_batch(op, vm, p, m, ob);  // ob[e * m + t]
        for (std::uint32_t t = 0; t < m; ++t) {
            float* o = &out[std::size_t{q0 + t} * w + h * hd];
            for (std::uint32_t e = 0; e < hd; ++e) {
                o[e] = ob[std::size_t{e} * m + t];
            }
        }
    });
}

}  // namespace locus::image::dense
