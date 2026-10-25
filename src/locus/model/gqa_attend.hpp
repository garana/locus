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
 * Decode attention for one query token over positions 0..pos, GQA:
 * head h of q (hd floats each) scores every position's key row of KV
 * head h / group, softmaxes, and weights the value rows into out.
 * Heads are independent, so they run across the thread pool when the
 * work is large enough; each head's arithmetic is the same either way
 * (bit-identical for every thread count).
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

}  // namespace locus::model
