#include "locus/pipeline/expert_server.hpp"

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
        // A call holds n_embd floats: a longer frame is refused unread,
        // so a peer cannot make this host allocate more (i#850).
        if (read_message(fd, in, n_embd) != ReadResult::kOk) {
            return served;  // closed, or a broken or oversized frame
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
