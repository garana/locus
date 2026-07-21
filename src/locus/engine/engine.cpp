#include "locus/engine/engine.hpp"

#include <algorithm>
#include <cassert>

namespace locus::engine {

Engine::Engine(const model::LlamaModel& m, tok::TokenId eos,
               Config cfg)
    : model_(m),
      eos_(eos),
      cfg_(cfg),
      cache_(m.make_cache(cfg.n_blocks)),
      ws_(m.make_workspace()),
      logits_(m.hparams().n_vocab) {}

Engine::Engine(const model::LlamaModel& m, tok::TokenId eos)
    : Engine(m, eos, Config{}) {}

std::uint64_t Engine::submit(std::vector<tok::TokenId> prompt,
                             std::uint32_t max_new_tokens) {
    auto r = std::make_unique<Request>();
    r->id = requests_.size();
    r->prompt = std::move(prompt);
    r->max_new_tokens = max_new_tokens;
    requests_.push_back(std::move(r));
    waiting_.push_back(requests_.back()->id);
    return requests_.back()->id;
}

std::uint32_t Engine::blocks_for(std::uint32_t n_tokens) const {
    const std::uint32_t bt = cache_.geometry().block_tokens;
    return (n_tokens + bt - 1) / bt;
}

bool Engine::step() {
    // Admission: FCFS while the pool can cover prompt + headroom.
    while (!waiting_.empty() &&
           running_.size() < cfg_.max_running) {
        Request& r = *requests_[waiting_.front()];
        // Readmitted victims recompute prompt + prior output, so
        // admission must cover their full committed length.
        const std::uint32_t want = blocks_for(
            static_cast<std::uint32_t>(r.prompt.size()) +
            static_cast<std::uint32_t>(r.generated.size()) +
            cfg_.decode_headroom);
        if (want > cache_.free_blocks()) {
            break;  // FCFS: do not admit later arrivals first
        }
        r.status = Status::kRunning;
        running_.push_back(r.id);
        waiting_.pop_front();
    }

    // One iteration: each running sequence advances; prefill is
    // capped by the shared budget, decode always costs 1 token.
    std::uint32_t budget = cfg_.prefill_budget;
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_[running_[i]];
        advance(r, budget);
        if (r.status != Status::kRunning) {
            running_.erase(running_.begin() +
                           static_cast<std::ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    return !waiting_.empty() || !running_.empty();
}

void Engine::advance(Request& r, std::uint32_t& budget) {
    const std::uint32_t n_prompt =
        static_cast<std::uint32_t>(r.prompt.size());

    while (true) {
        const std::uint32_t n_total =
            n_prompt + static_cast<std::uint32_t>(
                           r.generated.size());
        const bool prefilling = r.n_fed < n_prompt;

        // Sample as soon as logits for the last fed token exist.
        if (!prefilling && r.n_fed == n_total && r.n_fed > 0 &&
            r.seq.n_tokens == r.n_fed) {
            const auto next = model::argmax(logits_);
            r.generated.push_back(next);
            if (on_token) {
                on_token(r, next);
            }
            if (next == eos_ ||
                r.generated.size() >= r.max_new_tokens) {
                finish(r, Status::kDone);
            }
            return;  // one decode token per iteration
        }

        if (prefilling && budget == 0) {
            return;  // out of prefill budget this iteration
        }

        if (r.n_fed + 1 > model_.hparams().n_ctx) {
            finish(r, Status::kFailed, "context overflow");
            return;
        }
        if (!cache_.ensure_capacity(r.seq, 1)) {
            // Pool dry: preempt the newest running sequence; if
            // that is us, we cannot make progress right now (or
            // ever, if the pool is just too small).
            const std::uint64_t victim = running_.back();
            if (victim == r.id) {
                if (running_.size() == 1) {
                    finish(r, Status::kFailed,
                           "kv pool too small for request");
                }
                return;
            }
            preempt(victim);
            continue;  // retry the allocation
        }

        const tok::TokenId input =
            r.n_fed < n_prompt
                ? r.prompt[r.n_fed]
                : r.generated[r.n_fed - n_prompt];
        model_.forward(input, cache_, r.seq, ws_, logits_);
        ++r.n_fed;
        if (prefilling) {
            --budget;
        }
    }
}

void Engine::preempt(std::uint64_t victim_id) {
    Request& v = *requests_[victim_id];
    cache_.release(v.seq);
    v.n_fed = 0;  // full recompute when readmitted
    v.status = Status::kWaiting;
    running_.erase(
        std::find(running_.begin(), running_.end(), victim_id));
    // Keep FCFS order: victims go to the front of the wait queue.
    waiting_.push_front(victim_id);
}

void Engine::finish(Request& r, Status s, std::string error) {
    cache_.release(r.seq);
    r.status = s;
    r.error = std::move(error);
    r.n_fed = 0;
}

void Engine::run_to_completion() {
    while (step()) {
    }
}

const Request* Engine::get(std::uint64_t id) const {
    return id < requests_.size() ? requests_[id].get() : nullptr;
}

}  // namespace locus::engine
