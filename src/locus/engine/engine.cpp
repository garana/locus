#include "locus/engine/engine.hpp"

#include <algorithm>
#include <cassert>
#include <random>

#include "locus/model/speculative.hpp"

namespace locus::engine {

Engine::Engine(const model::LlamaModel& m, tok::TokenId eos,
               Config cfg)
    : model_(m),
      eos_(eos),
      cfg_(cfg),
      cache_(m.make_cache(cfg.n_blocks, cfg.kv_type)),
      ws_(m.make_workspace()),
      logits_(m.hparams().n_vocab) {
    if (cfg_.prefix_cache) {
        prefix_cache_ = std::make_unique<PrefixCache>(
            cache_, cfg_.prefix_cache_slots);
    }
}

Engine::Engine(const model::LlamaModel& m, tok::TokenId eos)
    : Engine(m, eos, Config{}) {}

std::uint64_t Engine::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint,
    LogprobsOpt logprobs) {
    auto r = std::make_unique<Request>();
    r->id = requests_.size();
    r->prompt = std::move(prompt);
    r->max_new_tokens = max_new_tokens;
    r->sampling = sampling;
    r->constraint = std::move(constraint);
    r->logprobs_opt = logprobs;
    r->rng.seed(seed != 0 ? seed
                          : std::random_device{}() ^
                                (r->id * 0x9e3779b97f4a7c15ULL));
    requests_.push_back(std::move(r));
    waiting_.push_back(requests_.back()->id);
    return requests_.back()->id;
}

std::uint32_t Engine::blocks_for(std::uint32_t n_tokens) const {
    const std::uint32_t bt = cache_.geometry().block_tokens;
    return (n_tokens + bt - 1) / bt;
}

std::string Engine::admit_error(std::uint32_t n_prompt) const {
    const std::uint32_t n_ctx = model_.hparams().n_ctx;
    if (n_prompt >= n_ctx) {
        return "prompt length " + std::to_string(n_prompt) +
               " exceeds context window " + std::to_string(n_ctx);
    }
    const std::uint32_t want =
        blocks_for(n_prompt + cfg_.decode_headroom);
    if (want > cache_.total_blocks()) {
        return "prompt needs " + std::to_string(want) +
               " KV blocks but the pool holds only " +
               std::to_string(cache_.total_blocks());
    }
    return "";
}

bool Engine::has_runnable_work() const {
    if (!running_.empty()) {
        return true;  // running work always advances a step
    }
    if (waiting_.empty()) {
        return false;
    }
    // step() admits FCFS: it only looks at the queue front and breaks
    // if it does not fit, so runnability hinges on the front alone.
    // With nothing running the whole pool is reclaimable for it (free
    // blocks plus any evictable prefix-cache blocks == total_blocks);
    // a front that still does not fit is one admit_error should have
    // rejected -- park, do not spin.
    const Request& r = *requests_[waiting_.front()];
    const std::uint32_t want = blocks_for(
        static_cast<std::uint32_t>(r.prompt.size()) +
        static_cast<std::uint32_t>(r.generated.size()) +
        cfg_.decode_headroom);
    return want <= cache_.total_blocks();
}

bool Engine::step() {
    // Batched decode needs a model/backend that supports the
    // batched forward (Vulkan drives its own full forward and opts
    // out); fall back to the per-token scheduler otherwise so
    // default-on batched_decode is safe on every backend.
    // Speculative decoding runs on the per-token scheduler (it does
    // its own multi-token batched forward), so it takes priority over
    // the batched-decode path.
    if (cfg_.batched_decode && !cfg_.speculative &&
        model_.supports_batch()) {
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
            if (prefix_cache_) {
                prefix_cache_->evict_until_free(want);
            }
            if (want > cache_.free_blocks()) {
                break;  // FCFS: do not admit later arrivals first
            }
        }
        r.status = Status::kRunning;
        running_.push_back(r.id);
        waiting_.pop_front();
        try_adopt(r);
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

        // Speculative decode: when a just-sampled token is pending
        // (greedy, unconstrained), verify n-gram drafts in one
        // batched forward instead of a single per-token forward.
        if (cfg_.speculative && !prefilling && r.n_fed < n_total &&
            r.n_fed >= n_prompt && r.sampling.temperature <= 0.0f &&
            !r.constraint && model_.supports_batch() &&
            r.seq.n_tokens == r.n_fed) {
            if (spec_decode_step(r)) {
                return;  // one spec step per iteration
            }
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
    // Register the prompt's full-block prefix before releasing, so a
    // later request sharing it can adopt the (still-pinned) KV.
    if (prefix_cache_ && s == Status::kDone && !r.prompt.empty()) {
        prefix_cache_->insert(r.prompt, r.seq.blocks);
    }
    cache_.release(r.seq);
    r.status = s;
    r.error = std::move(error);
    r.n_fed = 0;
}

void Engine::try_adopt(Request& r) {
    if (!prefix_cache_ || r.n_fed != 0 || !r.generated.empty() ||
        !r.seq.blocks.empty()) {
        return;
    }
    const std::uint32_t n_prompt =
        static_cast<std::uint32_t>(r.prompt.size());
    if (n_prompt < 2) {
        return;
    }
    std::vector<kv::BlockId> blocks = prefix_cache_->match(r.prompt);
    const std::uint32_t bt = cache_.geometry().block_tokens;
    // Leave at least one prompt token to reprefill so the sampler
    // has logits for the last position.
    const std::uint32_t max_blocks = (n_prompt - 1) / bt;
    if (blocks.size() > max_blocks) {
        blocks.resize(max_blocks);
    }
    if (blocks.empty()) {
        return;
    }
    cache_.retain_prefix(r.seq, blocks);
    r.n_fed = static_cast<std::uint32_t>(blocks.size()) * bt;
    prefix_reused_tokens_ += r.n_fed;
}

bool Engine::spec_decode_step(Request& r) {
    const std::uint32_t n_prompt =
        static_cast<std::uint32_t>(r.prompt.size());
    const std::uint32_t base = r.n_fed;  // pending token's position
    const std::uint32_t V = model_.hparams().n_vocab;
    const std::uint32_t n_ctx = model_.hparams().n_ctx;

    // Context = prompt + generated up to and including the pending
    // token (generated[base - n_prompt]).
    std::vector<tok::TokenId> ctx = r.prompt;
    const std::size_t gen_upto = base - n_prompt + 1;
    ctx.insert(ctx.end(), r.generated.begin(),
               r.generated.begin() +
                   static_cast<std::ptrdiff_t>(gen_upto));
    const tok::TokenId pending = r.generated[base - n_prompt];

    const auto drafts = model::ngram_draft(ctx, cfg_.spec_ngram,
                                           cfg_.spec_draft);
    std::vector<tok::TokenId> batch;
    batch.push_back(pending);
    batch.insert(batch.end(), drafts.begin(), drafts.end());

    const std::uint32_t room = n_ctx > base ? n_ctx - base : 0;
    if (room == 0) {
        return false;
    }
    if (batch.size() > room) {
        batch.resize(room);
    }
    const std::uint32_t kb =
        static_cast<std::uint32_t>(batch.size());
    if (!cache_.ensure_capacity(r.seq, kb)) {
        return false;  // let the per-token path preempt
    }

    const std::size_t need = static_cast<std::size_t>(kb) * V;
    if (batched_logits_.size() < need) {
        batched_logits_.assign(need, 0.0f);
    }
    model_.forward_batch(batch, cache_, r.seq, ws_,
                         {batched_logits_.data(), need},
                         /*all_logits=*/true);
    ++spec_steps_;

    // Verify greedily: logits at position base+i-1 predict token
    // base+i; the pending token (batch[0]) is already committed.
    std::uint32_t acc = 0;
    bool terminal = false;
    for (std::uint32_t i = 1; i < kb; ++i) {
        const tok::TokenId predicted = model::argmax(
            {batched_logits_.data() +
                 static_cast<std::size_t>(i - 1) * V,
             V});
        const bool accepted = predicted == batch[i];
        const tok::TokenId t = accepted ? batch[i] : predicted;
        r.generated.push_back(t);
        if (on_token) {
            on_token(r, t);
        }
        if (accepted) {
            ++acc;
            ++spec_accepted_;
        }
        if (t == eos_ || r.generated.size() >= r.max_new_tokens) {
            terminal = true;
            break;
        }
        if (!accepted) {
            break;  // corrected -> discard the remaining drafts
        }
    }
    if (!terminal && acc == kb - 1) {
        // Every draft matched (or there were none): the bonus token.
        const tok::TokenId bonus = model::argmax(
            {batched_logits_.data() +
                 static_cast<std::size_t>(kb - 1) * V,
             V});
        r.generated.push_back(bonus);
        if (on_token) {
            on_token(r, bonus);
        }
        terminal = bonus == eos_ ||
                   r.generated.size() >= r.max_new_tokens;
    }

    // Commit pending + accepted drafts; roll back rejected KV.
    const std::uint32_t committed = base + 1 + acc;
    r.n_fed = committed;
    r.seq.n_tokens = committed;
    if (terminal) {
        finish(r, Status::kDone);
    }
    return true;
}

void Engine::run_to_completion() {
    // Stop when idle, or when the only backlog is a request too large
    // for the pool: step() would otherwise report "more work" forever
    // and spin. Such a request is left waiting (callers should reject
    // it up front via admit_error).
    while (!idle() && has_runnable_work()) {
        step();
    }
}

const Request* Engine::get(std::uint64_t id) const {
    return id < requests_.size() ? requests_[id].get() : nullptr;
}

void Engine::sample_from(Request& r, std::span<float> logits) {
    // sample() mutates logits (penalties/temperature); snapshot the
    // raw logits first so logprobs reflect the natural distribution.
    std::vector<float> raw;
    if (r.logprobs_opt.enabled) {
        raw.assign(logits.begin(), logits.end());
    }
    const auto next = model::sample(logits, r.sampling, r.generated,
                                    r.rng, r.constraint.get());
    if (r.constraint) {
        r.constraint->commit(next);
    }
    r.generated.push_back(next);
    if (r.logprobs_opt.enabled) {
        LogprobEntry e;
        e.token = next;
        model::logprobs_from(raw, next, r.logprobs_opt.top,
                             e.logprob, e.top);
        r.logprobs.push_back(std::move(e));
    }
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
            if (prefix_cache_) {
                prefix_cache_->evict_until_free(want);
            }
            if (want > cache_.free_blocks()) {
                break;
            }
        }
        r.status = Status::kRunning;
        running_.push_back(r.id);
        waiting_.pop_front();
        try_adopt(r);
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
