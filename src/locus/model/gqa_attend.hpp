#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "locus/backend/cpu_ops.hpp"
#include "locus/model/transformer.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::model {

/**
 * One head of gqa_attend: head h of q scores positions 0..pos of KV
 * head h / group, softmaxes, and weights the value rows into head h of
 * out.
 */
template <class KRow, class VRow>
void gqa_head(const float* q, float* out, std::uint32_t h, std::uint32_t hd,
              std::uint32_t group, std::uint32_t pos, float kq_scale,
              const KRow& krow_at, const VRow& vrow_at) {
    thread_local std::vector<float> att_buf;
    att_buf.resize(pos + 1);
    std::span<float> att(att_buf.data(), pos + 1);
    const float* qh = q + static_cast<std::size_t>(h) * hd;
    const std::size_t kvh = static_cast<std::size_t>(h / group) * hd;
    for (std::uint32_t t = 0; t <= pos; ++t) {
        const float* kt = krow_at(t) + kvh;
        float s = 0.0f;
        for (std::uint32_t i = 0; i < hd; ++i) {
            s += qh[i] * kt[i];
        }
        att[t] = s * kq_scale;
    }
    backend::softmax_inplace(att);
    float* oh = out + static_cast<std::size_t>(h) * hd;
    for (std::uint32_t i = 0; i < hd; ++i) {
        oh[i] = 0.0f;
    }
    for (std::uint32_t t = 0; t <= pos; ++t) {
        const float* vt = vrow_at(t) + kvh;
        const float a = att[t];
        for (std::uint32_t i = 0; i < hd; ++i) {
            oh[i] += a * vt[i];
        }
    }
}

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
    auto head = [&](std::uint32_t h) {
        gqa_head(q, out, h, hd, group, pos, kq_scale, krow_at, vrow_at);
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
 * gqa_attend for n query tokens at positions base .. base + n - 1 in
 * one pool dispatch over every (token, head) pair: token t's q row is
 * q + t * q_stride, its output row out + t * q_stride, and it sees
 * positions 0..base + t, whose key and value rows must all be stored.
 * Each pair's arithmetic is gqa_head's, so the result is bit-identical
 * to n gqa_attend calls. Same threading rule as gqa_attend.
 */
template <class KRow, class VRow>
void gqa_attend_batch(const float* q, float* out, std::size_t q_stride,
                      std::uint32_t n, std::uint32_t base,
                      std::uint32_t n_heads, std::uint32_t hd,
                      std::uint32_t group, float kq_scale,
                      const KRow& krow_at, const VRow& vrow_at) {
    const std::size_t items = std::size_t{n} * n_heads;
    const std::size_t t = std::min<std::size_t>(mt_fanout(), items);
    auto item = [&](std::size_t i) {
        const auto tok = static_cast<std::uint32_t>(i / n_heads);
        const auto h = static_cast<std::uint32_t>(i % n_heads);
        gqa_head(q + tok * q_stride, out + tok * q_stride, h, hd, group,
                 base + tok, kq_scale, krow_at, vrow_at);
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

}  // namespace locus::model
