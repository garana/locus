#include "locus/pipeline/driver.hpp"

#include <random>

#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

DriverResult drive_generation(int head_fd, int logits_fd,
                              std::span<const tok::TokenId> prompt,
                              tok::TokenId eos_id,
                              const DriverParams& params) {
    if (prompt.empty()) {
        return {{}, false};  // nothing to feed: no logits to sample from
    }
    if (!write_message(head_fd, make_token(params.request_id, 0,
                                           prompt.front()))) {
        return {{}, false};
    }
    Message first;
    if (read_message(logits_fd, first) != ReadResult::kOk ||
        first.type != MsgType::kLogits ||
        first.request_id != params.request_id || first.position != 0) {
        return {{}, false};
    }
    return drive_generation_continue(head_fd, logits_fd, prompt,
                                     std::move(first.data), eos_id, params);
}

DriverResult drive_generation_continue(int head_fd, int logits_fd,
                                       std::span<const tok::TokenId> prompt,
                                       std::vector<float> first_logits,
                                       tok::TokenId eos_id,
                                       const DriverParams& params) {
    std::mt19937_64 rng(params.seed);
    std::vector<tok::TokenId> history;  // penalties look back over this
    history.reserve(prompt.size() + params.max_tokens);
    std::uint32_t pos = 1;  // position 0 is the prompt's first token
    std::vector<float> logits = std::move(first_logits);

    // Sends one token to the head stage and reads the logits the chain
    // produces for the next position. @returns false if the transport
    // ended (EOF/error/timeout are all terminal here), or if the logits
    // belong to another request or position: with several sessions on
    // one chain, a crossed connection must fail, not mix their streams.
    auto round_trip = [&](tok::TokenId t) -> bool {
        if (!write_message(head_fd, make_token(params.request_id, pos, t))) {
            return false;
        }
        Message lg;
        if (read_message(logits_fd, lg) != ReadResult::kOk ||
            lg.type != MsgType::kLogits ||
            lg.request_id != params.request_id || lg.position != pos) {
            return false;
        }
        logits = std::move(lg.data);
        ++pos;
        return true;
    };

    // Feed the rest of the prompt; the last round leaves `logits`
    // holding the distribution for the first generated token.
    if (!prompt.empty()) {
        history.push_back(prompt.front());
    }
    for (std::size_t i = 1; i < prompt.size(); ++i) {
        history.push_back(prompt[i]);
        if (!round_trip(prompt[i])) {
            return {{}, false};  // truncated before any output
        }
    }

    DriverResult out;
    for (std::uint32_t i = 0; i < params.max_tokens; ++i) {
        const tok::TokenId next =
            model::sample(logits, params.sampling, history, rng);
        out.tokens.push_back(next);
        if (next == eos_id) {
            out.complete = true;  // stopped on the eos token
            return out;
        }
        history.push_back(next);
        if (!round_trip(next)) {  // transport ended mid-stream
            return out;  // complete stays false: truncated
        }
    }
    out.complete = true;  // reached max_tokens cleanly
    return out;
}

}  // namespace locus::pipeline
