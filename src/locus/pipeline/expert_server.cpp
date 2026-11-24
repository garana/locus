#include "locus/pipeline/expert_server.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <vector>

#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

std::uint64_t serve_expert_calls(const model::TransformerModel& m, int fd,
                                 std::uint32_t expert_lo,
                                 std::uint32_t expert_hi) {
    const std::uint32_t n_embd = m.hparams().n_embd;
    auto ws = m.make_workspace();
    std::uint64_t served = 0;
    for (;;) {
        Message in;
        // A call holds n_embd floats, a batched one at most
        // kMaxExpertBatch of them and their ids (about 34 MB at n_embd
        // 2048): a longer frame is refused unread, so a peer cannot make
        // this host allocate more (i#850, i#1186).
        if (read_message(fd, in,
                         kMaxExpertBatch * (n_embd + 1)) != ReadResult::kOk) {
            return served;  // closed, or a broken or oversized frame
        }
        if (in.type == MsgType::kExpertCallBatch) {
            const auto n = static_cast<std::uint32_t>(in.token);
            // n = 0 is the client's probe at connect: an empty reply says
            // this host takes batched calls.
            bool ok = in.token >= 0 && n <= kMaxExpertBatch &&
                      in.ids.size() == n &&
                      in.data.size() == std::size_t{n} * n_embd;
            for (std::uint32_t i = 0; ok && i < n; ++i) {
                ok = in.ids[i] >= expert_lo && in.ids[i] < expert_hi;
            }
            if (!ok) {
                std::fprintf(stderr,
                             "serve_expert_calls: bad batched call (%d "
                             "pairs, %zu ids, %zu floats): closing\n",
                             in.token, in.ids.size(), in.data.size());
                return served;
            }
            // Each expert once over its tokens; the outputs in call
            // order.
            std::vector<std::uint32_t> order(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                order[i] = i;
            }
            std::stable_sort(order.begin(), order.end(),
                             [&](std::uint32_t a, std::uint32_t b) {
                                 return in.ids[a] < in.ids[b];
                             });
            std::vector<float> ds(std::size_t{n} * n_embd), xg, dg;
            try {
                for (std::uint32_t i = 0; i < n;) {
                    std::uint32_t j = i;
                    while (j < n && in.ids[order[j]] == in.ids[order[i]]) {
                        ++j;
                    }
                    const std::uint32_t m_rows = j - i;
                    xg.resize(std::size_t{m_rows} * n_embd);
                    dg.resize(xg.size());
                    for (std::uint32_t q = 0; q < m_rows; ++q) {
                        std::copy_n(in.data.data() +
                                        std::size_t{order[i + q]} * n_embd,
                                    n_embd, xg.data() + std::size_t{q} * n_embd);
                    }
                    m.expert_out_batch(in.position, in.ids[order[i]], xg,
                                       m_rows, dg);
                    for (std::uint32_t q = 0; q < m_rows; ++q) {
                        std::copy_n(dg.data() + std::size_t{q} * n_embd, n_embd,
                                    ds.data() +
                                        std::size_t{order[i + q]} * n_embd);
                    }
                    i = j;
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "serve_expert_calls: %s: closing\n",
                             e.what());
                return served;
            }
            if (!write_message(fd, make_expert_out_batch(in.request_id,
                                                         in.position, n,
                                                         std::move(ds)))) {
                return served;
            }
            served += n;
            continue;
        }
        const auto expert = static_cast<std::uint32_t>(in.token);
        if (in.type != MsgType::kExpertCall || expert < expert_lo ||
            expert >= expert_hi || in.data.size() != n_embd) {
            std::fprintf(stderr,
                         "serve_expert_calls: bad call (type %u, expert "
                         "%u, %zu floats): closing\n",
                         static_cast<unsigned>(in.type), expert,
                         in.data.size());
            return served;
        }
        std::vector<float> d(n_embd);
        try {
            m.expert_out(in.position, expert, in.data, d, ws);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "serve_expert_calls: %s: closing\n",
                         e.what());
            return served;
        }
        if (!write_message(fd, make_expert_out(in.request_id, in.position,
                                               expert, std::move(d)))) {
            return served;
        }
        ++served;
    }
}

}  // namespace locus::pipeline
