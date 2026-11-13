#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/backend/registry.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::model {

/**
 * @returns dot(a, b) over n floats in a fixed order: eight interleaved
 * partial sums (lane j takes elements j, j + 8, ...), added as a fixed
 * tree, then the tail. The lanes are explicit, so the compiler
 * vectorizes the loop without reassociating anything (a plain
 * reduction stays one serial add chain under strict FP, i#206), and
 * the result is the same bits on every CPU and thread count.
 */
/**
 * The eight lanes of dot_f32_8: as two 4-wide vectors (W = 4, any CPU)
 * or one 8-wide vector (W = 8, for code built with AVX). Lane j is
 * the same sum of products either way, so the results are the same
 * bits; only the instructions differ.
 */
template <int W>
struct Lanes8;

template <>
struct Lanes8<4> {
    typedef float f4 __attribute__((vector_size(16)));
    f4 lo = {0.0f, 0.0f, 0.0f, 0.0f};
    f4 hi = {0.0f, 0.0f, 0.0f, 0.0f};
    /** Lanes += a[0..7] * b[0..7]. */
    [[gnu::always_inline]] void add_mul(const float* a, const float* b) {
        f4 a0, a1, b0, b1;
        std::memcpy(&a0, a, sizeof a0);
        std::memcpy(&a1, a + 4, sizeof a1);
        std::memcpy(&b0, b, sizeof b0);
        std::memcpy(&b1, b + 4, sizeof b1);
        lo += a0 * b0;
        hi += a1 * b1;
    }
    [[gnu::always_inline]] float lane(int j) const {
        return j < 4 ? lo[j] : hi[j - 4];
    }
};

template <>
struct Lanes8<8> {
    typedef float f8 __attribute__((vector_size(32)));
    f8 v = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    [[gnu::always_inline]] void add_mul(const float* a, const float* b) {
        f8 a8, b8;
        std::memcpy(&a8, a, sizeof a8);
        std::memcpy(&b8, b, sizeof b8);
        v += a8 * b8;
    }
    [[gnu::always_inline]] float lane(int j) const { return v[j]; }
};

/** The fixed tree over the eight lanes. */
template <int W>
[[gnu::always_inline]] inline float lanes8_sum(const Lanes8<W>& l) {
    return ((l.lane(0) + l.lane(1)) + (l.lane(2) + l.lane(3))) +
           ((l.lane(4) + l.lane(5)) + (l.lane(6) + l.lane(7)));
}

template <int W = 4>
[[gnu::always_inline]] inline float dot_f32_8(const float* a, const float* b,
                                              std::uint32_t n) {
    // Lanes 0..3 in lo, 4..7 in hi (GCC / Clang vector types: one
    // 4-wide multiply and add per half; the plain 8-lane array loop
    // vectorizes only 2 wide), or all eight in one AVX vector.
    Lanes8<W> l;
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        l.add_mul(a + i, b + i);
    }
    float s = lanes8_sum(l);
    for (; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

/**
 * dot_f32_8 of four vectors a[0..3] with one b: b is loaded once per
 * step for all four, and each dot keeps dot_f32_8's lanes, tree and
 * tail, so out[r] is the bits dot_f32_8(a[r], b, n) gives.
 */
template <int W = 4>
[[gnu::always_inline]] inline void dot4_f32_8(const float* const a[4],
                                              const float* b,
                                              std::uint32_t n,
                                              float out[4]) {
    Lanes8<W> l[4];
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int r = 0; r < 4; ++r) {
            l[r].add_mul(a[r] + i, b + i);
        }
    }
    for (int r = 0; r < 4; ++r) {
        float s = lanes8_sum(l[r]);
        for (std::uint32_t k = i; k < n; ++k) {
            s += a[r][k] * b[k];
        }
        out[r] = s;
    }
}

/**
 * One head of gqa_attend: head h of q scores positions 0..pos of KV
 * head h / group (dot_f32_8), softmaxes, and weights the value rows
 * into head h of out.
 */
template <int W, class KRow, class VRow>
[[gnu::always_inline]] inline void gqa_head(
    const float* q, float* out, std::uint32_t h, std::uint32_t hd,
    std::uint32_t group, std::uint32_t pos, float kq_scale,
    const KRow& krow_at, const VRow& vrow_at) {
    thread_local std::vector<float> att_buf;
    att_buf.resize(pos + 1);
    std::span<float> att(att_buf.data(), pos + 1);
    const float* qh = q + static_cast<std::size_t>(h) * hd;
    const std::size_t kvh = static_cast<std::size_t>(h / group) * hd;
    for (std::uint32_t t = 0; t <= pos; ++t) {
        att[t] = dot_f32_8<W>(qh, krow_at(t) + kvh, hd) * kq_scale;
    }
    backend::softmax_inplace(att);
    float* oh = out + static_cast<std::size_t>(h) * hd;
    for (std::uint32_t i = 0; i < hd; ++i) {
        oh[i] = 0.0f;
    }
    // Four positions per pass over oh, their terms still added one
    // position at a time (the same bits as one pass per position), so
    // oh is loaded and stored a quarter as often.
    std::uint32_t t = 0;
    for (; t + 4 <= pos + 1; t += 4) {
        const float* v0 = vrow_at(t) + kvh;
        const float* v1 = vrow_at(t + 1) + kvh;
        const float* v2 = vrow_at(t + 2) + kvh;
        const float* v3 = vrow_at(t + 3) + kvh;
        const float a0 = att[t], a1 = att[t + 1];
        const float a2 = att[t + 2], a3 = att[t + 3];
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] = (((oh[i] + a0 * v0[i]) + a1 * v1[i]) + a2 * v2[i]) +
                    a3 * v3[i];
        }
    }
    for (; t <= pos; ++t) {
        const float* vt = vrow_at(t) + kvh;
        const float a = att[t];
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] += a * vt[i];
        }
    }
}

/** Tests set it to run the 4-wide build on an AVX CPU too. */
inline bool& attend_avx_off() {
    static bool off = false;
    return off;
}

/**
 * @returns Whether the attention kernels take their AVX build: an
 * x86-64 CPU that has AVX (and an OS that saves its registers). The
 * AVX build keeps every lane's sums, so the results are the same bits.
 */
inline bool attend_avx() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool on = __builtin_cpu_supports("avx") != 0;
    return on && !attend_avx_off();
#else
    return false;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// The 8-wide builds, compiled for AVX: only these functions get the
// target (the rest of the binary stays baseline), and the bodies they
// inline run their eight lanes in one register.
template <class KRow, class VRow>
[[gnu::target("avx")]] void gqa_head_avx(
    const float* q, float* out, std::uint32_t h, std::uint32_t hd,
    std::uint32_t group, std::uint32_t pos, float kq_scale,
    const KRow& krow_at, const VRow& vrow_at) {
    gqa_head<8>(q, out, h, hd, group, pos, kq_scale, krow_at, vrow_at);
}
#endif

/**
 * Decode attention for one query token over positions 0..pos, GQA:
 * head h of q (hd floats each) scores every position's key row of KV
 * head h / group, softmaxes, and weights the value rows into out.
 * Heads are independent, so they run across the thread pool when the
 * work is large enough; each head's arithmetic is the same either way
 * (bit-identical for every thread count).
 *
 * Call it from the forward's own thread, never from inside a pool task:
 * the pool runs one parallel_for at a time, so a nested one would wait
 * on itself.
 *
 * @param krow_at Position t -> its key row (all KV heads).
 * @param vrow_at Position t -> its value row.
 * @param min_work Multiply-adds below which the heads run inline (a
 *     pool dispatch costs more than it saves under ~64K).
 */
template <class KRow, class VRow>
void gqa_attend(const float* q, float* out, std::uint32_t n_heads,
                std::uint32_t hd, std::uint32_t group, std::uint32_t pos,
                float kq_scale, const KRow& krow_at, const VRow& vrow_at,
                std::uint64_t min_work = std::uint64_t{1} << 16) {
    const bool avx = attend_avx();
    auto head = [&](std::uint32_t h) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        if (avx) {
            gqa_head_avx(q, out, h, hd, group, pos, kq_scale, krow_at,
                         vrow_at);
            return;
        }
#endif
        (void)avx;
        gqa_head<4>(q, out, h, hd, group, pos, kq_scale, krow_at, vrow_at);
    };
    const std::size_t t = std::min<std::size_t>(mt_fanout(), n_heads);
    if (t <= 1 ||
        std::uint64_t{n_heads} * (pos + 1) * hd < min_work) {
        for (std::uint32_t h = 0; h < n_heads; ++h) {
            head(h);
        }
        return;
    }
    sys::ThreadPool::instance().parallel_for(t, [&](std::size_t c) {
        for (auto h = static_cast<std::uint32_t>(c); h < n_heads;
             h += static_cast<std::uint32_t>(t)) {
            head(h);
        }
    });
}

/**
 * gqa_head for every head of KV head g and query tokens t0 .. t1 - 1
 * (token t sees positions 0..base + t), sharing the key and value row
 * reads: each key row is loaded once and scored against every query
 * that sees it, and each group of four value rows is loaded once and
 * added into every such query's output. Each query's arithmetic is
 * gqa_head's, in its order (the same dot, softmax and four-position
 * groups from 0, then the tail one position at a time), so the result
 * is bit-identical.
 */
template <int W, class KRow, class VRow>
[[gnu::always_inline]] inline void gqa_kv_block(
    const float* q, float* out, std::size_t q_stride, std::uint32_t t0,
    std::uint32_t t1, std::uint32_t base, std::uint32_t g, std::uint32_t hd,
    std::uint32_t group, float kq_scale, const KRow& krow_at,
    const VRow& vrow_at) {
    const std::uint32_t nq = (t1 - t0) * group;
    const std::uint32_t last = base + t1 - 1;  // the furthest position
    const std::size_t stride = std::size_t{last} + 1;
    thread_local std::vector<float> att_buf;
    att_buf.resize(nq * stride);
    const std::size_t kvh = std::size_t{g} * hd;
    // Query j: token t0 + j / group, head g * group + j % group.
    const auto qrow = [&](std::uint32_t j) {
        return std::size_t{t0 + j / group} * q_stride +
               std::size_t{g * group + j % group} * hd;
    };
    const auto pos_of = [&](std::uint32_t j) {
        return base + t0 + j / group;
    };
    for (std::uint32_t p = 0; p <= last; ++p) {
        const float* k = krow_at(p) + kvh;
        // pos_of grows with j, so the queries that see p are a suffix:
        // from the first one on, four at a time through dot4_f32_8 (the
        // key row loaded once for four), then one at a time.
        std::uint32_t j = 0;
        while (j < nq && p > pos_of(j)) {
            ++j;
        }
        for (; j + 4 <= nq; j += 4) {
            const float* qs[4] = {q + qrow(j), q + qrow(j + 1),
                                  q + qrow(j + 2), q + qrow(j + 3)};
            float d[4];
            dot4_f32_8<W>(qs, k, hd, d);
            for (std::uint32_t r = 0; r < 4; ++r) {
                att_buf[(j + r) * stride + p] = d[r] * kq_scale;
            }
        }
        for (; j < nq; ++j) {
            att_buf[j * stride + p] =
                dot_f32_8<W>(q + qrow(j), k, hd) * kq_scale;
        }
    }
    for (std::uint32_t j = 0; j < nq; ++j) {
        backend::softmax_inplace(
            {att_buf.data() + j * stride, std::size_t{pos_of(j)} + 1});
        float* oh = out + qrow(j);
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] = 0.0f;
        }
    }
    for (std::uint32_t t = 0; t <= last; t += 4) {
        const float* v[4] = {};
        for (std::uint32_t s = 0; s < 4 && t + s <= last; ++s) {
            v[s] = vrow_at(t + s) + kvh;
        }
        for (std::uint32_t j = 0; j < nq; ++j) {
            const std::uint32_t pos = pos_of(j);
            if (t > pos) continue;
            const float* att = att_buf.data() + j * stride;
            float* oh = out + qrow(j);
            if (t + 4 <= pos + 1) {
                const float* v0 = v[0];
                const float* v1 = v[1];
                const float* v2 = v[2];
                const float* v3 = v[3];
                const float a0 = att[t], a1 = att[t + 1];
                const float a2 = att[t + 2], a3 = att[t + 3];
                for (std::uint32_t i = 0; i < hd; ++i) {
                    oh[i] = (((oh[i] + a0 * v0[i]) + a1 * v1[i]) +
                             a2 * v2[i]) +
                            a3 * v3[i];
                }
                continue;
            }
            // The tail: positions t .. pos, one at a time.
            for (std::uint32_t s = t; s <= pos; ++s) {
                const float* vt = v[s - t];
                const float a = att[s];
                for (std::uint32_t i = 0; i < hd; ++i) {
                    oh[i] += a * vt[i];
                }
            }
        }
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
/** gqa_kv_block's 8-wide build, compiled for AVX (see gqa_head_avx). */
template <class KRow, class VRow>
[[gnu::target("avx")]] void gqa_kv_block_avx(
    const float* q, float* out, std::size_t q_stride, std::uint32_t t0,
    std::uint32_t t1, std::uint32_t base, std::uint32_t g, std::uint32_t hd,
    std::uint32_t group, float kq_scale, const KRow& krow_at,
    const VRow& vrow_at) {
    gqa_kv_block<8>(q, out, q_stride, t0, t1, base, g, hd, group, kq_scale,
                    krow_at, vrow_at);
}
#endif

/**
 * gqa_attend for n query tokens at positions base .. base + n - 1 in
 * one pool dispatch: token t's q row is q + t * q_stride, its output
 * row out + t * q_stride, and it sees positions 0..base + t, whose key
 * and value rows must all be stored. Each task is one KV head and a
 * block of tokens (gqa_kv_block), so a key or value row is read once
 * for all the block's queries of that head, not once per query. The
 * result is bit-identical to n gqa_attend calls. Same threading rule as
 * gqa_attend.
 */
template <class KRow, class VRow>
void gqa_attend_batch(const float* q, float* out, std::size_t q_stride,
                      std::uint32_t n, std::uint32_t base,
                      std::uint32_t n_heads, std::uint32_t hd,
                      std::uint32_t group, float kq_scale,
                      const KRow& krow_at, const VRow& vrow_at) {
    // Query tokens per task: at most kMaxBlock, few enough that every
    // pool thread gets a task (a short chunk, a speculative verify), and
    // few enough at long context that a task's scores (tokens x group x
    // positions floats, thread_local) stay under about 4 MB. Each task
    // reads its KV head's whole K and V history, so larger blocks read
    // it fewer times (i#701: 8 to 32 tokens). The block size changes no
    // query's arithmetic.
    constexpr std::uint64_t kMaxBlock = 32;
    constexpr std::uint64_t kMaxScores = std::uint64_t{1} << 20;
    const std::uint32_t n_kv = n_heads / group;
    const std::uint64_t per_token = std::uint64_t{group} * (base + n);
    const std::uint64_t fan = std::max<std::size_t>(mt_fanout(), 1);
    // ceil(fan / n_kv) blocks per KV head give every thread a task.
    const std::uint64_t by_threads =
        std::uint64_t{n} / ((fan + n_kv - 1) / n_kv);
    const std::uint64_t by_memory =
        kMaxScores / std::max<std::uint64_t>(per_token, 1);
    const auto block = static_cast<std::uint32_t>(
        std::clamp<std::uint64_t>(std::min(by_threads, by_memory), 1,
                                  kMaxBlock));
    const std::uint32_t n_blocks = (n + block - 1) / block;
    const std::size_t items = std::size_t{n_kv} * n_blocks;
    const std::size_t t = std::min<std::size_t>(mt_fanout(), items);
    const bool avx = attend_avx();
    auto item = [&](std::size_t i) {
        const auto g = static_cast<std::uint32_t>(i % n_kv);
        const auto b = static_cast<std::uint32_t>(i / n_kv);
        const std::uint32_t t0 = b * block;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
        if (avx) {
            gqa_kv_block_avx(q, out, q_stride, t0, std::min(n, t0 + block),
                             base, g, hd, group, kq_scale, krow_at, vrow_at);
            return;
        }
#endif
        (void)avx;
        gqa_kv_block<4>(q, out, q_stride, t0, std::min(n, t0 + block), base,
                        g, hd, group, kq_scale, krow_at, vrow_at);
    };
    if (t <= 1) {
        for (std::size_t i = 0; i < items; ++i) {
            item(i);
        }
        return;
    }
    sys::ThreadPool::instance().parallel_for(t, [&](std::size_t c) {
        for (std::size_t i = c; i < items; i += t) {
            item(i);
        }
    });
}

/**
 * GQA attention on the backend's device KV mirror (i#466) for n queries
 * of seq at positions base .. base + n - 1, laid out as gqa_attend_batch
 * lays them out. Their K and V rows must be synced to the mirror first.
 *
 * @returns false when the cache has no mirror, the backend no kv_attend,
 *     or the device call fails; the caller then attends on the host.
 */
inline bool device_gqa_attend(const backend::Ops& op,
                              const kv::PagedKvCache& cache,
                              const kv::PagedKvCache::Seq& seq,
                              std::uint32_t layer, const Hparams& hp,
                              std::uint32_t base, std::uint32_t n,
                              const float* q, float* out,
                              std::size_t q_stride) {
    if (op.kv_attend == nullptr || cache.mirror_data() == nullptr) {
        return false;
    }
    backend::KvAttend a;
    a.mirror = cache.mirror_data();
    a.blocks = seq.blocks.data();
    a.n_blocks = static_cast<std::uint32_t>(seq.blocks.size());
    a.block_stride = cache.block_floats();
    a.layer_off = cache.layer_offset(layer);
    a.v_off = cache.layer_floats() / 2;
    a.block_tokens = cache.geometry().block_tokens;
    a.n_heads = hp.n_heads;
    a.n_kv_heads = hp.n_kv_heads;
    a.head_dim = hp.head_dim;
    a.kq_scale = hp.kq_scale;
    a.n_q = n;
    a.base = base;
    a.q = q;
    a.out = out;
    a.q_stride = q_stride;
    return op.kv_attend(a);
}

/**
 * device_gqa_attend for n queries of n DIFFERENT sequences (batch
 * decode): query t is at position pos[t] of *seqs[t], with its q row at
 * q + t * q_stride and its output at out + t * q_stride. Gives each query
 * the same bits as a device_gqa_attend call of its own.
 *
 * @returns false as device_gqa_attend does; the caller then attends on
 *     the host.
 */
inline bool device_gqa_attend_seqs(
    const backend::Ops& op, const kv::PagedKvCache& cache,
    std::span<kv::PagedKvCache::Seq* const> seqs, std::uint32_t layer,
    const Hparams& hp, std::span<const std::uint32_t> pos, const float* q,
    float* out, std::size_t q_stride) {
    if (op.kv_attend == nullptr || cache.mirror_data() == nullptr) {
        return false;
    }
    const auto n = static_cast<std::uint32_t>(seqs.size());
    // The sequences' block tables, one after another.
    std::vector<std::uint32_t> blocks, table(n);
    for (std::uint32_t t = 0; t < n; ++t) {
        table[t] = static_cast<std::uint32_t>(blocks.size());
        blocks.insert(blocks.end(), seqs[t]->blocks.begin(),
                      seqs[t]->blocks.end());
    }
    backend::KvAttend a;
    a.mirror = cache.mirror_data();
    a.blocks = blocks.data();
    a.n_blocks = static_cast<std::uint32_t>(blocks.size());
    a.block_stride = cache.block_floats();
    a.layer_off = cache.layer_offset(layer);
    a.v_off = cache.layer_floats() / 2;
    a.block_tokens = cache.geometry().block_tokens;
    a.n_heads = hp.n_heads;
    a.n_kv_heads = hp.n_kv_heads;
    a.head_dim = hp.head_dim;
    a.kq_scale = hp.kq_scale;
    a.n_q = n;
    a.q_pos = pos.data();
    a.q_table = table.data();
    a.q = q;
    a.out = out;
    a.q_stride = q_stride;
    return op.kv_attend(a);
}

}  // namespace locus::model
