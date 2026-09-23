#include "locus/pipeline/driver.hpp"

#include <random>

#include "locus/pipeline/message.hpp"

namespace locus::pipeline {

DriverResult drive_generation(int head_fd, int logits_fd,
                              std::span<const tok::TokenId> prompt,
                              tok::TokenId eos_id,
                              const DriverParams& params) {
    std::mt19937_64 rng(params.seed);
    std::vector<tok::TokenId> history;  // penalties look back over this
    history.reserve(prompt.size() + params.max_tokens);
    std::uint32_t pos = 0;
    std::vector<float> logits;  // the last kLogits payload

    // Sends one token to the head stage and reads the logits the chain
    // produces for the next position. @returns false if the transport
    // ended (EOF/error/timeout are all terminal here).
    auto round_trip = [&](tok::TokenId t) -> bool {
        if (!write_message(head_fd, make_token(params.request_id, pos, t))) {
            return false;
        }
        Message lg;
        if (read_message(logits_fd, lg) != ReadResult::kOk ||
            lg.type != MsgType::kLogits) {
            return false;
        }
        logits = std::move(lg.data);
        ++pos;
        return true;
    };

    // Feed the prompt; the last round leaves `logits` holding the
    // distribution for the first generated token.
    for (const tok::TokenId t : prompt) {
        history.push_back(t);
        if (!round_trip(t)) {
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
