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
    // Batched decode needs a model/backend that supports the
    // batched forward (Vulkan drives its own full forward and opts
    // out); fall back to the per-token scheduler otherwise so
    // default-on batched_decode is safe on every backend.
    if (cfg_.batched_decode && model_.supports_batch()) {
        return step_batched();
    }
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
            sample_from(r, logits_);
            return;  // one decode token per iteration
        }

        if (prefilling && budget == 0) {
            return;  // out of prefill budget this iteration
        }

        // R10: ingest the remaining prompt in one batched forward
        // when the model and config allow -- byte-identical to
        // the per-token path, fewer weight reads. Falls through
        // to per-token (which preempts) if the chunk cannot be
        // capacity-ensured.
        if (prefilling && cfg_.batched_prefill &&
            model_.supports_batch()) {
            const std::uint32_t remaining = n_prompt - r.n_fed;
            const std::uint32_t nb = std::min(remaining, budget);
            if (nb >= 2 &&
                r.n_fed + nb <= model_.hparams().n_ctx &&
                cache_.ensure_capacity(r.seq, nb)) {
                model_.forward_batch(
                    std::span<const tok::TokenId>(
                        r.prompt.data() + r.n_fed, nb),
                    cache_, r.seq, ws_, logits_);
                r.n_fed += nb;
                budget -= nb;
                continue;
            }
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

void Engine::sample_from(Request& r,
                         std::span<const float> logits) {
    const auto next = model::argmax(logits);
    r.generated.push_back(next);
    if (on_token) {
        on_token(r, next);
    }
    if (next == eos_ ||
        r.generated.size() >= r.max_new_tokens) {
        finish(r, Status::kDone);
    }
}

bool Engine::prefill_only(Request& r, std::uint32_t& budget) {
    const std::uint32_t n_prompt =
        static_cast<std::uint32_t>(r.prompt.size());
    const std::uint32_t V = model_.hparams().n_vocab;
    if (r.logits.size() != V) {
        r.logits.assign(V, 0.0f);
    }
    while (r.n_fed < n_prompt) {
        if (budget == 0) {
            return true;
        }
        if (cfg_.batched_prefill && model_.supports_batch()) {
            const std::uint32_t remaining = n_prompt - r.n_fed;
            const std::uint32_t nb = std::min(remaining, budget);
            if (nb >= 2 &&
                r.n_fed + nb <= model_.hparams().n_ctx &&
                cache_.ensure_capacity(r.seq, nb)) {
                model_.forward_batch(
                    std::span<const tok::TokenId>(
                        r.prompt.data() + r.n_fed, nb),
                    cache_, r.seq, ws_, r.logits);
                r.n_fed += nb;
                budget -= nb;
                continue;
            }
        }
        if (r.n_fed + 1 > model_.hparams().n_ctx) {
            finish(r, Status::kFailed, "context overflow");
            return false;
        }
        if (!cache_.ensure_capacity(r.seq, 1)) {
            const std::uint64_t victim = running_.back();
            if (victim == r.id) {
                if (running_.size() == 1) {
                    finish(r, Status::kFailed,
                           "kv pool too small for request");
                    return false;
                }
                return true;  // cannot progress this step
            }
            preempt(victim);
            continue;
        }
        model_.forward(r.prompt[r.n_fed], cache_, r.seq, ws_,
                       r.logits);
        ++r.n_fed;
        --budget;
    }
    return true;
}

bool Engine::step_batched() {
    // Admission: same FCFS policy as step().
    while (!waiting_.empty() &&
           running_.size() < cfg_.max_running) {
        Request& r = *requests_[waiting_.front()];
        const std::uint32_t want = blocks_for(
            static_cast<std::uint32_t>(r.prompt.size()) +
            static_cast<std::uint32_t>(r.generated.size()) +
            cfg_.decode_headroom);
        if (want > cache_.free_blocks()) {
            break;
        }
        r.status = Status::kRunning;
        running_.push_back(r.id);
        waiting_.pop_front();
    }

    // Phase 1: prefill each running request still on its prompt.
    std::uint32_t budget = cfg_.prefill_budget;
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_[running_[i]];
        if (r.n_fed <
            static_cast<std::uint32_t>(r.prompt.size())) {
            if (!prefill_only(r, budget)) {
                running_.erase(running_.begin() +
                               static_cast<std::ptrdiff_t>(i));
                continue;
            }
        }
        ++i;
    }

    // Phase 2: sample from every request that has caught up.
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_[running_[i]];
        const std::uint32_t n_total =
            static_cast<std::uint32_t>(r.prompt.size()) +
            static_cast<std::uint32_t>(r.generated.size());
        if (r.n_fed == n_total && r.n_fed > 0 &&
            r.seq.n_tokens == r.n_fed) {
            sample_from(r, r.logits);
            if (r.status != Status::kRunning) {
                running_.erase(running_.begin() +
                               static_cast<std::ptrdiff_t>(i));
                continue;
            }
        }
        ++i;
    }

    // Phase 3: one forward_batch_decode for every running request
    // with a pending token (a just-sampled decode token, or a
    // generated token being recomputed after preemption).
    std::vector<tok::TokenId> toks;
    std::vector<kv::PagedKvCache::Seq*> seqs;
    std::vector<std::uint64_t> ids;
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_[running_[i]];
        const std::uint32_t n_prompt =
            static_cast<std::uint32_t>(r.prompt.size());
        const std::uint32_t n_total =
            n_prompt +
            static_cast<std::uint32_t>(r.generated.size());
        if (r.n_fed < n_prompt || r.n_fed >= n_total) {
            ++i;  // still prefilling (budget-out) or up to date
            continue;
        }
        if (r.n_fed + 1 > model_.hparams().n_ctx) {
            finish(r, Status::kFailed, "context overflow");
            running_.erase(running_.begin() +
                           static_cast<std::ptrdiff_t>(i));
            continue;
        }
        if (!cache_.ensure_capacity(r.seq, 1)) {
            const std::uint64_t victim = running_.back();
            if (victim == r.id) {
                // Nothing newer to preempt. If we are the only
                // running request the pool is just too small for
                // us -- fail rather than wedge forever; otherwise
                // defer and let an earlier request free blocks.
                if (running_.size() == 1) {
                    finish(r, Status::kFailed,
                           "kv pool too small for request");
                    running_.erase(
                        running_.begin() +
                        static_cast<std::ptrdiff_t>(i));
                    continue;
                }
                ++i;  // cannot fit this step; retry next
                continue;
            }
            preempt(victim);  // tail (not yet gathered) -> free it
            continue;         // retry same index
        }
        toks.push_back(r.generated[r.n_fed - n_prompt]);
        seqs.push_back(&r.seq);
        ids.push_back(r.id);
        ++i;
    }
    if (!toks.empty()) {
        const std::uint32_t V = model_.hparams().n_vocab;
        const std::size_t need =
            static_cast<std::size_t>(toks.size()) * V;
        if (batched_logits_.size() < need) {
            batched_logits_.assign(need, 0.0f);
        }
        model_.forward_batch_decode(
            toks, cache_,
            std::span<kv::PagedKvCache::Seq* const>(seqs), ws_,
            std::span<float>(batched_logits_.data(), need));
        for (std::size_t j = 0; j < ids.size(); ++j) {
            Request& r = *requests_[ids[j]];
            if (r.logits.size() != V) {
                r.logits.assign(V, 0.0f);
            }
            std::copy(batched_logits_.begin() +
                          static_cast<std::ptrdiff_t>(j) * V,
                      batched_logits_.begin() +
                          static_cast<std::ptrdiff_t>(j + 1) * V,
                      r.logits.begin());
            ++r.n_fed;
        }
    }

    return !waiting_.empty() || !running_.empty();
}

}  // namespace locus::engine
