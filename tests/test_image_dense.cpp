// image::dense, the token-major ops the image and video models share.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/dense.hpp"

namespace {

/** softmax(q k^T / sqrt(hd)) v in double, token-major as dense. */
std::vector<double> attention_ref(const std::vector<float>& q,
                                  std::uint32_t nq,
                                  const std::vector<float>& k,
                                  const std::vector<float>& v,
                                  std::uint32_t nk, std::uint32_t heads,
                                  std::uint32_t hd) {
    const std::uint32_t w = heads * hd;
    std::vector<double> out(std::size_t{nq} * w), s(nk);
    for (std::uint32_t h = 0; h < heads; ++h) {
        for (std::uint32_t t = 0; t < nq; ++t) {
            double mx = -INFINITY;
            for (std::uint32_t j = 0; j < nk; ++j) {
                double d = 0;
                for (std::uint32_t e = 0; e < hd; ++e) {
                    d += double{q[std::size_t{t} * w + h * hd + e]} *
                         k[std::size_t{j} * w + h * hd + e];
                }
                s[j] = d / std::sqrt(double(hd));
                mx = std::max(mx, s[j]);
            }
            double sum = 0;
            for (double& x : s) {
                x = std::exp(x - mx);
                sum += x;
            }
            for (std::uint32_t e = 0; e < hd; ++e) {
                double o = 0;
                for (std::uint32_t j = 0; j < nk; ++j) {
                    o += s[j] * v[std::size_t{j} * w + h * hd + e];
                }
                out[std::size_t{t} * w + h * hd + e] = o / sum;
            }
        }
    }
    return out;
}

/** out[r * n + t] = w[r] . x[t] through op, as dense's mat_batch. */
void mat_batch(const locus::backend::Ops& op, const locus::backend::Mat& w,
               const float* x, std::uint32_t n, float* out) {
    if (op.matvec_batch != nullptr) {
        op.matvec_batch(w, {x, std::size_t{n} * w.cols},
                        {out, std::size_t{n} * w.rows}, n);
        return;
    }
    std::vector<float> o(w.rows);
    for (std::uint32_t t = 0; t < n; ++t) {
        op.matvec(w, {x + std::size_t{t} * w.cols, w.cols}, o);
        for (std::uint32_t r = 0; r < w.rows; ++r) {
            out[std::size_t{r} * n + t] = o[r];
        }
    }
}

/** The default dense attention, one head and 64-query block at a time:
 *  the same backend calls and the std::exp softmax with 1 / sum. */
std::vector<float> attention_default(const locus::backend::Ops& op,
                                     const std::vector<float>& q,
                                     std::uint32_t nq,
                                     const std::vector<float>& k,
                                     const std::vector<float>& v,
                                     std::uint32_t nk, std::uint32_t heads,
                                     std::uint32_t hd) {
    using TT = locus::gguf::TensorType;
    const std::uint32_t w = heads * hd;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> out(std::size_t{nq} * w);
    for (std::uint32_t h = 0; h < heads; ++h) {
        std::vector<float> kh(std::size_t{nk} * hd), vt(std::size_t{hd} * nk);
        for (std::uint32_t j = 0; j < nk; ++j) {
            const std::size_t at = std::size_t{j} * w + h * hd;
            for (std::uint32_t e = 0; e < hd; ++e) {
                kh[std::size_t{j} * hd + e] = k[at + e];
                vt[std::size_t{e} * nk + j] = v[at + e];
            }
        }
        const locus::backend::Mat km{
            TT::kF32, reinterpret_cast<const std::byte*>(kh.data()), nk, hd};
        const locus::backend::Mat vm{
            TT::kF32, reinterpret_cast<const std::byte*>(vt.data()), hd, nk};
        for (std::uint32_t q0 = 0; q0 < nq; q0 += 64) {
            const std::uint32_t m = std::min(64u, nq - q0);
            std::vector<float> qb(std::size_t{m} * hd), s(std::size_t{nk} * m),
                p(std::size_t{m} * nk), ob(std::size_t{hd} * m);
            for (std::uint32_t t = 0; t < m; ++t) {
                for (std::uint32_t e = 0; e < hd; ++e) {
                    qb[std::size_t{t} * hd + e] =
                        q[std::size_t{q0 + t} * w + h * hd + e];
                }
            }
            mat_batch(op, km, qb.data(), m, s.data());
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
            mat_batch(op, vm, p.data(), m, ob.data());
            for (std::uint32_t t = 0; t < m; ++t) {
                for (std::uint32_t e = 0; e < hd; ++e) {
                    out[std::size_t{q0 + t} * w + h * hd + e] =
                        ob[std::size_t{e} * m + t];
                }
            }
        }
    }
    return out;
}

}  // namespace

// i#1035: without LOCUS_CPU_FAST the attention keeps the bits of its
// std::exp softmax (the goldens were checked on them).
TEST_CASE("dense attention without LOCUS_CPU_FAST keeps its bits",
          "[image][fast]") {
    if (locus::backend::cpu_fast()) {
        SKIP("LOCUS_CPU_FAST is set");
    }
    const auto& op = locus::backend::best_backend().ops;
    std::mt19937 rng(10351);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    const std::uint32_t nq = 67, nk = 70, heads = 3, hd = 16, w = heads * hd;
    std::vector<float> q(std::size_t{nq} * w), k(std::size_t{nk} * w),
        v(std::size_t{nk} * w), out(std::size_t{nq} * w);
    for (auto* x : {&q, &k, &v}) {
        for (float& f : *x) f = d(rng);
    }
    locus::image::dense::attention(op, q, nq, k, v, nk, heads, hd, out);
    const auto ref = attention_default(op, q, nq, k, v, nk, heads, hd);
    REQUIRE(std::memcmp(out.data(), ref.data(), out.size() * 4) == 0);
}

// i#1035: the attention's softmax is softmax_fast_inplace under
// LOCUS_CPU_FAST and the std::exp loop otherwise; both track the
// double reference. Span 4 gives scaled scores of several units, far
// down the polynomial exp's range; 70 keys leave a tail past 8 lanes.
TEST_CASE("dense attention tracks a double reference", "[image][fast]") {
    const auto& op = locus::backend::best_backend().ops;
    using Shape = std::pair<std::uint32_t, std::uint32_t>;  // nq, nk
    std::mt19937 rng(1035);
    for (const float span : {0.5f, 4.0f}) {
        for (const auto& [nq, nk] : {Shape{5, 70}, Shape{67, 33}}) {
            const std::uint32_t heads = 3, hd = 16, w = heads * hd;
            std::uniform_real_distribution<float> d(-span, span);
            std::vector<float> q(std::size_t{nq} * w), k(std::size_t{nk} * w),
                v(std::size_t{nk} * w), out(std::size_t{nq} * w);
            for (auto* x : {&q, &k, &v}) {
                for (float& f : *x) f = d(rng);
            }
            locus::image::dense::attention(op, q, nq, k, v, nk, heads, hd,
                                           out);
            const auto ref = attention_ref(q, nq, k, v, nk, heads, hd);
            double worst = 0;
            for (std::size_t i = 0; i < out.size(); ++i) {
                worst = std::max(worst, std::abs(out[i] - ref[i]) /
                                            (1.0 + std::abs(ref[i])));
            }
            INFO("span " << span << ", nq " << nq << ", nk " << nk
                         << ", fast " << locus::backend::cpu_fast());
            REQUIRE(worst < 1e-5 * span);
        }
    }
}
