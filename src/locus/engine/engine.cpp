#include "locus/engine/engine.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>

#include "locus/model/speculative.hpp"

namespace locus::engine {

Engine::Engine(const model::TransformerModel& m, tok::TokenId eos,
               Config cfg)
    : model_(m),
      eos_(eos),
      cfg_(cfg),
      cache_(m.make_cache(cfg.n_blocks, cfg.kv_type)),
      ws_(m.make_workspace()),
      logits_(m.hparams().n_vocab),
      spec_cost_(m.active_backend().name == "vulkan") {
    if (cfg_.prefill_budget == 0) {
        cfg_.prefill_budget = m.prefill_chunk();
    }
    if (cfg_.disk_prefix) {
        const auto& hp = m.hparams();
        if (hp.idx_heads > 0 || !hp.ds41.ratio.empty()) {
            throw std::invalid_argument(
                "disk prefix cache: this model keeps per-sequence state "
                "outside the KV pool");
        }
        const DiskKey& k = cfg_.disk_prefix->key();
        const kv::PagedKvCache::Geometry& g = cache_.geometry();
        if (k.kv_type != g.kv_type || k.block_tokens != g.block_tokens ||
            k.kv_dim != g.kv_dim || k.n_layers != g.n_layers ||
            k.layer_base != g.layer_base ||
            cfg_.disk_prefix->block_bytes() != cache_.block_bytes()) {
            throw std::invalid_argument(
                "disk prefix cache: the store's key does not match the "
                "KV cache geometry");
        }
        // Device attention over a KV mirror (CUDA) gives other K/V
        // bits than the host's; only this cache knows if it has one.
        cfg_.disk_prefix->set_device_attention(
            cache_.mirror_data() != nullptr &&
            m.active_backend().ops.kv_attend != nullptr);
        cfg_.prefix_cache = true;
    }
    if (cfg_.prefix_cache) {
        prefix_cache_ = std::make_unique<PrefixCache>(
            cache_, cfg_.prefix_cache_slots);
    }
    if (cfg_.disk_prefix) {
        disk_worker_ = std::make_unique<DiskPrefixWorker>(
            cfg_.disk_prefix, cfg_.disk_write_backlog);
        prefix_cache_->on_evict =
            [this](std::span<const tok::TokenId> key,
                   const std::vector<kv::BlockId>& blocks) {
                write_back(key, blocks);
            };
    }
}

Engine::Engine(const model::TransformerModel& m, tok::TokenId eos)
    : Engine(m, eos, Config{}) {}

Engine::~Engine() {
    if (!disk_worker_) {
        return;
    }
    // Write-back covers only what RAM evicted; the entries still in RAM
    // are written here, directly from the pool (no copy: nothing runs
    // any more). Only the newest that fit the cap, written oldest first
    // so the newest end most recently used: writing more would evict
    // files just written.
    try {
        disk_worker_->drain();
        struct Entry {
            std::span<const tok::TokenId> key;
            const std::vector<kv::BlockId>* blocks;
        };
        std::vector<Entry> entries;
        prefix_cache_->for_each_oldest_first(
            [&](std::span<const tok::TokenId> key,
                const std::vector<kv::BlockId>& blocks) {
                entries.push_back({key, &blocks});
            });
        const std::uint64_t bb = cache_.block_bytes();
        std::uint64_t budget = cfg_.disk_prefix->cap_bytes();
        std::size_t first = entries.size();
        while (first > 0 &&
               entries[first - 1].blocks->size() * bb <= budget) {
            --first;
            budget -= entries[first].blocks->size() * bb;
        }
        for (std::size_t i = first; i < entries.size(); ++i) {
            const auto n =
                static_cast<std::uint32_t>(entries[i].blocks->size());
            if (n == 0 || cfg_.disk_prefix->find(entries[i].key) >= n) {
                continue;
            }
            std::vector<const std::uint8_t*> src;
            for (const kv::BlockId b : *entries[i].blocks) {
                src.push_back(cache_.block_data(b));
            }
            cfg_.disk_prefix->write(entries[i].key, n, src);
        }
    } catch (...) {
        // Out of memory at shutdown: the disk tier is only a cache.
    }
}

std::uint64_t Engine::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint,
    LogprobsOpt logprobs) {
    const std::uint64_t id = next_id_++;  // monotonic; never reused
    auto r = std::make_unique<Request>();
    r->id = id;
    r->prompt = std::move(prompt);
    r->max_new_tokens = max_new_tokens;
    r->sampling = sampling;
    r->constraint = std::move(constraint);
    r->logprobs_opt = logprobs;
    r->rng.seed(seed != 0 ? seed
                          : std::random_device{}() ^
                                (id * 0x9e3779b97f4a7c15ULL));
    r->t_submit = std::chrono::steady_clock::now();
    requests_.emplace(id, std::move(r));
    waiting_.push_back(id);
    return id;
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
    if (disk_worker_ && disk_worker_->has_done()) {
        return true;  // a finished read moves its request to running
    }
    if (waiting_.empty() ||
        running_.size() + loading_.size() >= cfg_.max_running) {
        return false;  // nothing to admit, or admission is capped
    }
    // step() admits FCFS: it only looks at the queue front and breaks
    // if it does not fit, so runnability hinges on the front alone.
    // With nothing running the whole pool is reclaimable for it (free
    // blocks plus any evictable prefix-cache blocks == total_blocks);
    // a front that still does not fit is one admit_error should have
    // rejected -- park, do not spin.
    const Request& r = *requests_.at(waiting_.front());
    const std::uint32_t want = blocks_for(
        static_cast<std::uint32_t>(r.prompt.size()) +
        static_cast<std::uint32_t>(r.generated.size()) +
        cfg_.decode_headroom);
    // Blocks of loading requests come back only when their reads end.
    std::uint32_t loading_blocks = 0;
    for (const std::uint64_t id : loading_) {
        loading_blocks += static_cast<std::uint32_t>(
            requests_.at(id)->seq.blocks.size());
    }
    for (const auto& [id, seq] : orphans_) {
        loading_blocks += static_cast<std::uint32_t>(seq.blocks.size());
    }
    return want + loading_blocks <= cache_.total_blocks();
}

bool Engine::step() {
    const auto t0 = std::chrono::steady_clock::now();
    step_tokens_ = 0;
    if (disk_worker_) {
        absorb_disk_loads();
    }
    const bool more = step_once();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0);
    metrics_.busy_ns.fetch_add(static_cast<std::uint64_t>(ns.count()),
                               std::memory_order_relaxed);
    metrics_.steps.fetch_add(1, std::memory_order_relaxed);
    metrics_.running.store(static_cast<std::uint32_t>(running_.size()),
                           std::memory_order_relaxed);
    metrics_.waiting.store(static_cast<std::uint32_t>(waiting_.size()),
                           std::memory_order_relaxed);
    // 0 once the engine has no work left, so an idle server does not
    // keep showing its last step as if it were busy.
    metrics_.batch_tokens.store(more ? step_tokens_ : 0,
                                std::memory_order_relaxed);
    count_block_states();
    return more;
}

void Engine::count_block_states() {
    // 0: not held by a running sequence; 1: decoding; 2: prefilling. A
    // block shared by both kinds counts once, as prefilling.
    block_state_.assign(cache_.total_blocks(), 0);
    for (const std::uint64_t id : running_) {
        const Request& r = *requests_.at(id);
        const std::uint8_t s = r.n_fed < r.prompt.size() ? 2 : 1;
        for (const kv::BlockId b : r.seq.blocks) {
            if (b < block_state_.size()) {
                block_state_[b] = std::max(block_state_[b], s);
            }
        }
    }
    // Blocks a disk read is filling count as prefilling: they hold the
    // prompt's KV, not the prefix cache's.
    const auto filling = [&](const kv::PagedKvCache::Seq& seq) {
        for (const kv::BlockId b : seq.blocks) {
            if (b < block_state_.size()) {
                block_state_[b] = 2;
            }
        }
    };
    for (const std::uint64_t id : loading_) {
        filling(requests_.at(id)->seq);
    }
    for (const auto& [id, seq] : orphans_) {
        filling(seq);
    }
    std::uint32_t prefilling = 0, decoding = 0;
    for (const std::uint8_t s : block_state_) {
        prefilling += s == 2;
        decoding += s == 1;
    }
    // The rest of the blocks in use belong only to the prefix cache.
    const std::uint32_t used = cache_.total_blocks() - cache_.free_blocks();
    const std::uint32_t held = prefilling + decoding;
    constexpr auto kRelaxed = std::memory_order_relaxed;
    metrics_.kv_blocks_prefilling.store(prefilling, kRelaxed);
    metrics_.kv_blocks_decoding.store(decoding, kRelaxed);
    metrics_.kv_blocks_cached.store(used > held ? used - held : 0, kRelaxed);
}

bool Engine::step_once() {
    // Batched decode needs a model/backend that supports a batched
    // decode step (Vulkan through its multi-sequence forward); fall
    // back to the per-token scheduler otherwise so default-on
    // batched_decode is safe on every backend.
    // Speculative decoding runs on the per-token scheduler (it does
    // its own multi-token batched forward), so it takes priority over
    // the batched-decode path.
    if (cfg_.batched_decode && !cfg_.speculative &&
        model_.supports_batch_decode()) {
        return step_batched();
    }
    // Admission: FCFS while the pool can cover prompt + headroom.
    while (!waiting_.empty() &&
           running_.size() + loading_.size() < cfg_.max_running) {
        if (!try_admit_front()) {
            break;  // FCFS: do not admit later arrivals first
        }
    }

    // One iteration: each running sequence advances; prefill takes its
    // share of the budget, decode always costs 1 token.
    share_prefill();
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_.at(running_[i]);
        advance(r, r.step_share);
        if (r.status != Status::kRunning) {
            running_.erase(running_.begin() +
                           static_cast<std::ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    return !idle();
}

void Engine::share_prefill() {
    // (remaining prompt tokens, request) of the prompts still
    // prefilling; any other running request is decoding.
    std::vector<std::pair<std::uint32_t, Request*>> want;
    bool decoding = false;
    for (const std::uint64_t id : running_) {
        Request& r = *requests_.at(id);
        r.step_share = 0;
        const auto n_prompt = static_cast<std::uint32_t>(r.prompt.size());
        if (r.n_fed < n_prompt) {
            want.emplace_back(n_prompt - r.n_fed, &r);
        } else {
            decoding = true;
        }
    }
    // With decodes running, every other step feeds no prompt, so half
    // the steps are decode-only.
    if (decoding && cfg_.decode_share && !want.empty()) {
        skip_prefill_ = !skip_prefill_;
        if (skip_prefill_) return;
    } else {
        skip_prefill_ = false;
    }
    // Each prompt is fed in the chunks it gets alone: the whole budget
    // or its whole remainder (batch invariance, i#403: a chunk's size
    // changes the sums). A prompt that waited prefill_max_wait steps
    // goes first, the longest waiter first; then the shortest
    // remainder. A prompt whose chunk no longer fits waits for the next
    // step. stable_sort keeps admission order among equals.
    const std::uint32_t max_wait = cfg_.prefill_max_wait;
    std::stable_sort(want.begin(), want.end(), [max_wait](const auto& a,
                                                          const auto& b) {
        const std::uint32_t wa = a.second->prefill_waits;
        const std::uint32_t wb = b.second->prefill_waits;
        if ((wa >= max_wait) != (wb >= max_wait)) return wa >= max_wait;
        if (wa >= max_wait) return wa > wb;
        return a.first < b.first;
    });
    const std::uint32_t chunk = cfg_.prefill_budget;
    std::uint32_t budget = chunk;
    for (auto& [remaining, r] : want) {
        const std::uint32_t need = std::min(remaining, chunk);
        if (need <= budget) {
            r->step_share = need;
            r->prefill_waits = 0;
            budget -= need;
        } else {
            ++r->prefill_waits;
        }
    }
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
        // Verify needs only forward_batch with every position's
        // logits, which the Vulkan backend has too.
        if (cfg_.speculative && !prefilling && r.n_fed < n_total &&
            r.n_fed >= n_prompt && model::is_greedy(r.sampling) &&
            !r.constraint && model_.supports_batch_prefill() &&
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
        // the per-token path, fewer weight reads. On a capacity miss
        // it falls through to the per-token path below, which reclaims
        // prefix-cache blocks (and preempts) one step at a time.
        // Deliberately NO reclaim_and_grow here: the per-token fallback
        // reclaims only as much as each token needs, not a whole-chunk
        // eviction up front. Trade-off: under a full pool a batched
        // prefill degrades to per-token here, losing the i#24/i#52
        // amortization -- a follow-up could reclaim for the chunk if
        // that regresses throughput under memory pressure.
        if (prefilling && cfg_.batched_prefill &&
            model_.supports_batch_prefill()) {
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
                step_tokens_ += nb;
                budget -= nb;
                continue;
            }
        }

        if (r.n_fed + 1 > model_.hparams().n_ctx) {
            finish(r, Status::kFailed, "context overflow");
            return;
        }
        if (!reclaim_and_grow(r, 1)) {
            // Pool still dry after reclaiming reusable prefix-cache
            // blocks: preempt the newest running sequence; if that is
            // us, we cannot make progress right now (or ever, if the
            // pool is just too small).
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
        if (cfg_.speculative && !prefilling) {
            // A plain decode step's time prices drafts against it
            // (model::choose_draft).
            const auto t_dec = std::chrono::steady_clock::now();
            model_.forward(input, cache_, r.seq, ws_, logits_);
            spec_cost_.observe(
                0, std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t_dec)
                       .count());
        } else {
            model_.forward(input, cache_, r.seq, ws_, logits_);
        }
        ++r.n_fed;
        ++step_tokens_;
        if (prefilling) {
            --budget;
        }
    }
}

bool Engine::try_admit_front() {
    Request& r = *requests_.at(waiting_.front());
    // Adopt any cached prefix BEFORE sizing the pool demand. Adoption
    // shares the cache entry's already-allocated blocks (retain_prefix
    // just bumps their refcount), so it consumes no free blocks, but
    // it pins the prefix into r.seq -- the eviction below then cannot
    // reclaim the very prefix we are about to reuse, and the block
    // demand `want` drops to the un-adopted tail. Without this order
    // (adopt after admission), a prompt whose pinned prefix leaves too
    // few free blocks for its own FULL length would evict that prefix
    // to admit itself, then find nothing to adopt -- a self-inflicted
    // cache miss. Readmitted victims (n_fed reset but output present)
    // do not adopt; they recompute in full.
    try_adopt(r);
    const std::uint32_t held =
        static_cast<std::uint32_t>(r.seq.blocks.size());
    // Readmitted victims recompute prompt + prior output, so admission
    // must cover their full committed length.
    const std::uint32_t need = blocks_for(
        static_cast<std::uint32_t>(r.prompt.size()) +
        static_cast<std::uint32_t>(r.generated.size()) +
        cfg_.decode_headroom);
    const std::uint32_t want = need > held ? need - held : 0;
    if (want > cache_.free_blocks()) {
        if (prefix_cache_) {
            prefix_cache_->evict_until_free(want);
        }
        if (want > cache_.free_blocks()) {
            return false;
        }
    }
    if (disk_worker_ && try_disk_load(r)) {
        return true;
    }
    r.status = Status::kRunning;
    if (r.t_admit == std::chrono::steady_clock::time_point{}) {
        r.t_admit = std::chrono::steady_clock::now();
    }
    running_.push_back(r.id);
    waiting_.pop_front();
    return true;
}

bool Engine::reclaim_and_grow(Request& r, std::uint32_t n_more) {
    if (cache_.ensure_capacity(r.seq, n_more)) {
        return true;
    }
    if (!prefix_cache_) {
        return false;
    }
    // Pool is physically full but evictable prefix-cache entries pin
    // reusable blocks. Drop LRU entries until the growth fits, then
    // retry. evict_until_free targets a free-block count, so size it
    // by the blocks this growth still needs.
    //
    // Caveat (pre-existing in evict_until_free, shared with the
    // admission path try_admit_front): an entry whose blocks are still
    // shared with a RUNNING sequence frees nothing when dropped.
    // evict_until_free drops LRU-first, and the oldest entries are the
    // least likely to be pinned by a running seq, so the no-gain case
    // needs essentially EVERY entry pinned -- a corner, not the common
    // case. When it does hit, the pool is cleared for no gain and we
    // fall through to preempt below. Correct and byte-neutral (a
    // dropped prefix just recomputes; finish() re-inserts it); a
    // follow-up could teach evict_until_free to skip fully-pinned
    // entries.
    prefix_cache_->evict_until_free(cache_.blocks_needed(r.seq, n_more));
    return cache_.ensure_capacity(r.seq, n_more);
}

void Engine::preempt(std::uint64_t victim_id) {
    Request& v = *requests_.at(victim_id);
    // With a disk tier, the victim's full blocks go to it first, keyed
    // by the tokens they hold (prompt, then output), so readmission
    // reads them back instead of recomputing (i#621).
    if (disk_worker_) {
        const std::uint32_t bt = cache_.geometry().block_tokens;
        const std::uint32_t n = std::min(
            v.seq.n_tokens / bt,
            static_cast<std::uint32_t>(v.seq.blocks.size()));
        if (n > 0) {
            const std::vector<kv::BlockId> blocks(
                v.seq.blocks.begin(), v.seq.blocks.begin() + n);
            if (write_back(context_of(v, n * bt), blocks)) {
                disk_swapped_tokens_ += n * bt;
            }
        }
    }
    cache_.release(v.seq);
    v.n_fed = 0;  // recompute (or a disk read) when readmitted
    // Discard the adopted-prefix accounting. Readmission starts from
    // nothing (a recompute, or the disk tier's blocks, which
    // absorb_disk_loads credits anew) -- try_adopt no-ops once any
    // output exists, and even a still-in-prefill victim re-adopts fresh
    // -- so this request no longer holds the prefix it was credited
    // for. Roll the credit out
    // of the global reuse metric and clear the per-request cache_read
    // count; a later re-adopt re-adds it. Without this, a preempted
    // request over-reports cache_read_input_tokens (it reused nothing)
    // and the /metrics reuse counter double-counts on re-adopt.
    prefix_reused_tokens_ -= v.reused_prefix_tokens;
    v.reused_prefix_tokens = 0;
    v.status = Status::kWaiting;
    running_.erase(
        std::find(running_.begin(), running_.end(), victim_id));
    // Keep FCFS order: victims go to the front of the wait queue.
    waiting_.push_front(victim_id);
}

void Engine::observe_done(const Request& r) {
    using Clock = std::chrono::steady_clock;
    constexpr Clock::time_point kUnset{};
    if (r.t_admit == kUnset || r.t_first == kUnset) {
        return;  // finished without being stepped (nothing to time)
    }
    const auto now = Clock::now();
    auto ns = [](Clock::time_point a, Clock::time_point b) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                .count());
    };
    constexpr auto kRelaxed = std::memory_order_relaxed;
    metrics_.queue.observe(ns(r.t_submit, r.t_admit));
    const std::uint64_t prefill = ns(r.t_admit, r.t_first);
    metrics_.prefill.observe(prefill);
    metrics_.ttft.observe(ns(r.t_submit, r.t_first));
    metrics_.request.observe(ns(r.t_submit, now));
    const auto n_prompt = static_cast<std::uint64_t>(r.prompt.size());
    metrics_.prefill_tokens.fetch_add(
        n_prompt > r.reused_prefix_tokens ? n_prompt - r.reused_prefix_tokens
                                          : 0,
        kRelaxed);
    if (r.generated.size() > 1) {
        const std::uint64_t decode = ns(r.t_first, now);
        const std::uint64_t n = r.generated.size() - 1;
        metrics_.decode_token.observe(decode / n);
        metrics_.decode_tokens.fetch_add(n, kRelaxed);
        metrics_.decode_ns.fetch_add(decode, kRelaxed);
    }
}

void Engine::finish(Request& r, Status s, std::string error) {
    // Register the prompt's full-block prefix before releasing, so a
    // later request sharing it can adopt the (still-pinned) KV.
    if (prefix_cache_ && s == Status::kDone && !r.prompt.empty()) {
        const std::uint32_t written =
            prefix_cache_->insert(r.prompt, r.seq.blocks);
        // Report only tokens cached BEYOND the reused prefix (the
        // reused portion was already resident) -- the cache_creation
        // count in the API usage block.
        r.cached_prefix_tokens = written > r.reused_prefix_tokens
                                     ? written - r.reused_prefix_tokens
                                     : 0;
    }
    if (s == Status::kDone) {
        observe_done(r);
    }
    cache_.release(r.seq);
    r.status = s;
    r.error = std::move(error);
    r.n_fed = 0;
    // Reclaim the heavy per-request buffers now the request is
    // terminal and will never be stepped again: r.logits (n_vocab
    // floats -- the largest buffer, pure sampling scratch) and
    // r.prompt (already copied into the prefix cache above; not part
    // of the client-visible result -- generated / logprobs stay for
    // view()). Without this a long-lived server retains them for every
    // finished request, so resident memory grows without bound with
    // request count. swap-with-empty frees the capacity (clear/resize
    // would keep it).
    std::vector<float>().swap(r.logits);
    std::vector<tok::TokenId>().swap(r.prompt);

    // Record the terminal record for oldest-first cap eviction and
    // enforce the backstop cap now, so an abandoned request (client
    // never release()s it) cannot let terminal records accumulate
    // without bound. Only terminals are ever evicted here.
    terminal_fifo_.push_back(r.id);
    while (terminal_fifo_.size() > cfg_.max_retained_terminal) {
        const std::uint64_t old = terminal_fifo_.front();
        terminal_fifo_.pop_front();
        requests_.erase(old);  // no-op if already released
    }
}

void Engine::release(std::uint64_t id) {
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return;  // unknown or already released
    }
    if (is_terminal(it->second->status)) {
        requests_.erase(it);  // terminal + consumed: drop the record
    }
    // A stale terminal_fifo_ entry for this id is skipped later.
}

void Engine::cancel(std::uint64_t id) {
    auto it = requests_.find(id);
    if (it == requests_.end() || is_terminal(it->second->status)) {
        return;  // unknown or already terminal
    }
    Request& r = *it->second;
    // Drop it from whichever queue holds it so running_/waiting_ (and
    // thus idle()/has_runnable_work) are accurate the instant cancel
    // returns, then finish as cancelled -- finish() frees the KV
    // blocks and, gating prefix insertion on kDone, never caches a
    // half-generated prompt. Safe to mutate the queues here because
    // the loop holds mu_ across step() and only yields it between
    // steps, so cancel() never races a step in flight.
    auto rit = std::find(running_.begin(), running_.end(), id);
    auto lit = std::find(loading_.begin(), loading_.end(), id);
    if (rit != running_.end()) {
        running_.erase(rit);
    } else if (lit != loading_.end()) {
        // The read still writes these blocks: hold them aside until it
        // ends (absorb_disk_loads releases them), so the pool is
        // unchanged once it does and no other request gets them early.
        loading_.erase(lit);
        orphans_.emplace(id, std::exchange(r.seq, {}));
    } else {
        auto wit = std::find(waiting_.begin(), waiting_.end(), id);
        if (wit != waiting_.end()) {
            waiting_.erase(wit);
        }
    }
    finish(r, Status::kCancelled, "cancelled");
}

std::size_t Engine::fail_running(const std::string& why) {
    const std::vector<std::uint64_t> ids = std::move(running_);
    running_.clear();
    for (const std::uint64_t id : ids) {
        auto it = requests_.find(id);
        if (it != requests_.end() && !is_terminal(it->second->status)) {
            finish(*it->second, Status::kFailed, why);
        }
    }
    return ids.size();
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
    r.reused_prefix_tokens = r.n_fed;  // served from cache (cache_read)
    prefix_reused_tokens_ += r.n_fed;
}

std::vector<tok::TokenId> Engine::context_of(const Request& r,
                                             std::uint32_t n) const {
    std::vector<tok::TokenId> out(
        r.prompt.begin(),
        r.prompt.begin() + std::min<std::size_t>(n, r.prompt.size()));
    if (n > r.prompt.size()) {
        out.insert(out.end(), r.generated.begin(),
                   r.generated.begin() + (n - r.prompt.size()));
    }
    return out;
}

bool Engine::try_disk_load(Request& r) {
    if (r.n_fed != r.reused_prefix_tokens) {
        return false;
    }
    // The tokens to cover: the prompt, and for a readmitted victim the
    // output so far, whose blocks preempt() wrote.
    const std::uint32_t n_total = static_cast<std::uint32_t>(
        r.prompt.size() + r.generated.size());
    const std::uint32_t bt = cache_.geometry().block_tokens;
    const std::vector<tok::TokenId> ctx = context_of(r, n_total);
    const std::uint32_t on_disk = cfg_.disk_prefix->find(ctx);
    // One token is always fed again, so the disk gives at most
    // n_total - 1 positions. Load only for at least one more block
    // than RAM gave: a block-aligned prompt cached in RAM (which keeps
    // its last block for the tail) stays a RAM hit.
    if (n_total < 2 ||
        std::min(on_disk * bt, n_total - 1) < r.n_fed + bt) {
        return false;
    }
    // Drop the shorter RAM prefix: the read fills private blocks from
    // position 0, so no shared block is ever written.
    cache_.release(r.seq);
    prefix_reused_tokens_ -= r.reused_prefix_tokens;
    r.reused_prefix_tokens = 0;
    r.n_fed = 0;
    if (!cache_.ensure_capacity(r.seq, on_disk * bt)) {
        prefix_cache_->evict_until_free(
            cache_.blocks_needed(r.seq, on_disk * bt));
        if (!cache_.ensure_capacity(r.seq, on_disk * bt)) {
            try_adopt(r);  // what RAM still holds, if anything
            return false;
        }
    }
    std::vector<std::uint8_t*> dst;
    for (const kv::BlockId b : r.seq.blocks) {
        dst.push_back(cache_.block_data(b));
    }
    disk_worker_->read(
        r.id,
        std::vector<tok::TokenId>(ctx.begin(), ctx.begin() + on_disk * bt),
        on_disk, std::move(dst));
    r.status = Status::kLoading;
    if (r.t_admit == std::chrono::steady_clock::time_point{}) {
        r.t_admit = std::chrono::steady_clock::now();
    }
    loading_.push_back(r.id);
    waiting_.pop_front();
    return true;
}

void Engine::absorb_disk_loads() {
    const std::uint32_t bt = cache_.geometry().block_tokens;
    for (const DiskPrefixWorker::Done& d : disk_worker_->take_done()) {
        if (auto o = orphans_.find(d.id); o != orphans_.end()) {
            cache_.release(o->second);  // cancelled while loading
            orphans_.erase(o);
            continue;
        }
        const auto it = std::find(loading_.begin(), loading_.end(), d.id);
        if (it == loading_.end()) {
            continue;
        }
        loading_.erase(it);
        Request& r = *requests_.at(d.id);
        if (d.ok) {
            for (const kv::BlockId b : r.seq.blocks) {
                cache_.sync_block(b);
            }
            // Keep one token to feed, so the sampler gets the last
            // position's logits; its KV row is written again with the
            // same bytes, in a private block.
            const auto n_prompt =
                static_cast<std::uint32_t>(r.prompt.size());
            const std::uint32_t n = std::min(
                static_cast<std::uint32_t>(r.seq.blocks.size()) * bt,
                n_prompt +
                    static_cast<std::uint32_t>(r.generated.size()) - 1);
            r.seq.n_tokens = n;
            r.n_fed = n;
            // Only prompt positions count as reused input (cache_read);
            // a victim's output positions are its own.
            const std::uint32_t reused = std::min(n, n_prompt);
            r.reused_prefix_tokens = reused;
            prefix_reused_tokens_ += reused;
            ++disk_hits_;
            disk_loaded_tokens_ += n;
        }
        // A failed read leaves n_fed at 0: prefill computes the prompt
        // over the blocks the read did not fill.
        r.status = Status::kRunning;
        running_.push_back(d.id);
    }
}

bool Engine::write_back(std::span<const tok::TokenId> key,
                        const std::vector<kv::BlockId>& blocks) {
    const auto n = static_cast<std::uint32_t>(blocks.size());
    const std::size_t bb = cache_.block_bytes();
    if (n == 0) {
        return false;
    }
    if (cfg_.disk_prefix->find(key) >= n) {
        return true;  // already on disk
    }
    if (!disk_worker_->would_accept(std::size_t{n} * bb)) {
        return false;  // no room queued: dropped
    }
    // Copied now: the blocks are freed and reused right after this.
    std::vector<std::uint8_t> bytes(std::size_t{n} * bb);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::memcpy(bytes.data() + i * bb, cache_.block_data(blocks[i]),
                    bb);
    }
    return disk_worker_->write(
        std::vector<tok::TokenId>(key.begin(), key.end()), n,
        std::move(bytes));
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

    // The last plain decode's one-token draft, checked against the
    // token it decoded, keeps the accept rate current while no draft
    // is verified.
    if (r.spec.has_guess) {
        model::observe_guess(r.spec, r.spec.guess == pending);
        r.spec.has_guess = false;
    }
    const std::uint32_t max_draft =
        cfg_.spec_adaptive
            ? model::choose_draft(r.spec, spec_cost_, cfg_.spec_draft)
            : cfg_.spec_draft;
    if (max_draft == 0) {
        const auto g = model::ngram_draft(ctx, cfg_.spec_ngram, 1);
        if (!g.empty()) {
            r.spec.has_guess = true;
            r.spec.guess = g.front();
        }
        return false;  // a verify would cost more than it saves
    }
    const auto drafts = model::ngram_draft(ctx, cfg_.spec_ngram,
                                           max_draft);
    if (drafts.empty()) {
        // Nothing to verify: the per-token forward is the same step
        // without a one-token batched forward's overhead.
        return false;
    }
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
    // The verify's cost covers the forward and the logit scans, the
    // work a draft adds over a plain decode step.
    const auto t_verify = std::chrono::steady_clock::now();
    model_.forward_batch(batch, cache_, r.seq, ws_,
                         {batched_logits_.data(), need},
                         /*all_logits=*/true);
    // pred[i] is the model's token after batch[i]: it equals the
    // draft batch[i + 1] while drafts pass, then corrects the first
    // one that fails, or is the bonus token after the last draft.
    std::vector<tok::TokenId> pred;
    pred.reserve(kb);
    for (std::uint32_t i = 0; i < kb; ++i) {
        pred.push_back(model::argmax(
            {batched_logits_.data() + static_cast<std::size_t>(i) * V,
             V}));
        if (i + 1 < kb && pred.back() != batch[i + 1]) {
            break;  // corrected: discard the remaining drafts
        }
    }
    spec_cost_.observe(kb - 1,
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t_verify)
                           .count());
    ++spec_steps_;
    step_tokens_ += static_cast<std::uint32_t>(batch.size());

    // The pending token (batch[0]) is already committed.
    std::uint32_t acc = 0;
    bool terminal = false;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        const tok::TokenId t = pred[i];
        r.generated.push_back(t);
        if (on_token) {
            on_token(r, t);
        }
        if (i + 1 < kb && t == batch[i + 1]) {
            ++acc;
            ++spec_accepted_;
        }
        if (t == eos_ || r.generated.size() >= r.max_new_tokens) {
            terminal = true;
            break;
        }
    }

    // A draft cut short by EOS or the token cap says nothing about
    // how well drafts match, so only a finished verify counts.
    const std::uint32_t drafted = kb - 1;
    spec_drafted_ += drafted;
    if (drafted >= spec_drafts_.size()) {
        spec_drafts_.resize(drafted + 1, 0);
    }
    ++spec_drafts_[drafted];
    if (!terminal && drafted > 0) {
        model::observe_draft(r.spec, drafted, acc);
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
    while (!idle() && (has_runnable_work() || loading())) {
        if (!has_runnable_work()) {
            std::this_thread::yield();  // only a disk read in flight
        }
        step();
    }
}

const Request* Engine::get(std::uint64_t id) const {
    const auto it = requests_.find(id);
    return it == requests_.end() ? nullptr : it->second.get();
}

void Engine::sample_from(Request& r, std::span<float> logits) {
    // Picking the token (with the logprobs snapshot and any grammar
    // constraint) is the sample stage (i#285).
    model::StageLap lap(model_.stage_metrics());
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
    lap.lap(model::StageMetrics::kNoLayer, model::Stage::kSample);
    r.generated.push_back(next);
    if (r.generated.size() == 1) {
        r.t_first = std::chrono::steady_clock::now();
    }
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
        // Same batched-prefill fast path as advance(): on a capacity
        // miss it falls through to the per-token loop below, which
        // reclaims. Deliberately no reclaim here -- see advance().
        if (cfg_.batched_prefill && model_.supports_batch_prefill()) {
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
                step_tokens_ += nb;
                budget -= nb;
                continue;
            }
        }
        if (r.n_fed + 1 > model_.hparams().n_ctx) {
            finish(r, Status::kFailed, "context overflow");
            return false;
        }
        if (!reclaim_and_grow(r, 1)) {
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
        ++step_tokens_;
        --budget;
    }
    return true;
}

bool Engine::step_batched() {
    // Admission: same FCFS policy as step().
    while (!waiting_.empty() &&
           running_.size() + loading_.size() < cfg_.max_running) {
        if (!try_admit_front()) {
            break;
        }
    }

    // Phase 1: prefill each running request still on its prompt, by
    // its share of the budget.
    share_prefill();
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_.at(running_[i]);
        if (r.n_fed <
            static_cast<std::uint32_t>(r.prompt.size())) {
            if (!prefill_only(r, r.step_share)) {
                running_.erase(running_.begin() +
                               static_cast<std::ptrdiff_t>(i));
                continue;
            }
        }
        ++i;
    }

    // Phase 2: sample from every request that has caught up.
    for (std::size_t i = 0; i < running_.size();) {
        Request& r = *requests_.at(running_[i]);
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
        Request& r = *requests_.at(running_[i]);
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
        if (!reclaim_and_grow(r, 1)) {
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
        step_tokens_ += static_cast<std::uint32_t>(toks.size());
        for (std::size_t j = 0; j < ids.size(); ++j) {
            Request& r = *requests_.at(ids[j]);
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

    return !idle();
}

}  // namespace locus::engine
