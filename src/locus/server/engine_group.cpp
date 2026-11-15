#include "locus/server/engine_group.hpp"

#include <stdexcept>
#include <utility>

#include "locus/backend/variants.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::server {

std::vector<EngineGroup::Slot> EngineGroup::plan(
    std::size_t n, const std::vector<int>& devices) {
    if (n == 0) {
        n = 1;
    }
    if (!devices.empty() && devices.size() != n) {
        throw std::invalid_argument(
            "engines: one device per engine, or none");
    }
    std::vector<Slot> slots(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!devices.empty()) {
            slots[i].vulkan_device = devices[i];
        } else {
            slots[i].threads = sys::ThreadPool::share_threads(i, n);
        }
    }
    return slots;
}

EngineGroup::EngineGroup(const model::TransformerModel& m,
                         tok::TokenId eos,
                         const engine::Engine::Config& cfg,
                         const std::vector<Slot>& slots) {
    const std::vector<Slot> one(1);
    const std::vector<Slot>& use = slots.empty() ? one : slots;
    bool binds = false;
    for (const Slot& s : use) {
        binds = binds || s.vulkan_device >= 0;
    }
    // The KV pool of an engine on a device is made on that device: bind
    // this thread to each in turn, then go back, so a later Vulkan call
    // here lands where it did before (as locus-stage's executors).
    const int main_device = binds ? backend::vulkan_thread_device() : -1;
    loops_.reserve(use.size());
    for (const Slot& s : use) {
        EngineLoop::Setup setup;
        setup.threads = s.threads;
        if (s.vulkan_device >= 0) {
            const int d = s.vulkan_device;
            backend::vulkan_bind_thread(d);
            setup.thread_init = [d] { backend::vulkan_bind_thread(d); };
        }
        loops_.push_back(
            std::make_unique<EngineLoop>(m, eos, cfg, std::move(setup)));
    }
    if (binds) {
        backend::vulkan_bind_thread(main_device);
    }
    load_.assign(loops_.size(), 0);
}

std::string EngineGroup::admit_error(std::uint32_t n_prompt) {
    return loops_.front()->admit_error(n_prompt);
}

std::uint64_t EngineGroup::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint,
    engine::LogprobsOpt logprobs) {
    std::size_t e = 0;
    {
        std::lock_guard<std::mutex> lk(live_mu_);
        for (std::size_t i = 1; i < loops_.size(); ++i) {
            if (load_[i] < load_[e]) {
                e = i;
            }
        }
        ++load_[e];
    }
    std::uint64_t local = 0;
    try {
        local = loops_[e]->submit(std::move(prompt), max_new_tokens,
                                  sampling, seed, std::move(constraint),
                                  logprobs);
    } catch (...) {
        // Not submitted: it must not count toward the engine's load.
        std::lock_guard<std::mutex> lk(live_mu_);
        --load_[e];
        throw;
    }
    const std::uint64_t id = local * loops_.size() + e;
    std::lock_guard<std::mutex> lk(live_mu_);
    live_.insert(id);
    return id;
}

EngineLoop::View EngineGroup::view(std::uint64_t id) {
    return loop_of(id).view(local_id(id));
}

EngineLoop::View EngineGroup::wait_progress(
    std::uint64_t id, std::size_t n_seen, std::chrono::milliseconds max_wait) {
    return loop_of(id).wait_progress(local_id(id), n_seen, max_wait);
}

EngineLoop::View EngineGroup::wait_done(std::uint64_t id) {
    return loop_of(id).wait_done(local_id(id));
}

EngineLoop::View EngineGroup::wait_done_or_cancel(
    std::uint64_t id, const std::function<bool()>& alive) {
    return loop_of(id).wait_done_or_cancel(local_id(id), alive);
}

void EngineGroup::cancel(std::uint64_t id) {
    loop_of(id).cancel(local_id(id));
}

void EngineGroup::follow(std::uint64_t id) {
    loop_of(id).follow(local_id(id));
}

bool EngineGroup::take(std::uint64_t id, EngineLoop::View& acc) {
    return loop_of(id).take(local_id(id), acc);
}

void EngineGroup::unfollow(std::uint64_t id) {
    loop_of(id).unfollow(local_id(id));
}

void EngineGroup::set_progress_hook(const EngineLoop::ProgressHook& hook) {
    const std::uint64_t n = loops_.size();
    for (std::uint64_t e = 0; e < n; ++e) {
        if (!hook) {
            loops_[e]->set_progress_hook({});
            continue;
        }
        // Engine e's local ids back to group ids (local * n + e); the
        // buffer is only touched by that engine's worker, under its mutex.
        loops_[e]->set_progress_hook(
            [hook, e, n, ids = std::vector<std::uint64_t>()](
                std::span<const std::uint64_t> local) mutable {
                ids.clear();
                for (const std::uint64_t l : local) {
                    ids.push_back(l * n + e);
                }
                hook(ids);
            });
    }
}

void EngineGroup::release(std::uint64_t id) {
    // EngineLoop::release keeps a request that is not terminal, so the
    // group keeps counting it then (a later release drops it).
    if (!loop_of(id).release(local_id(id))) return;
    std::lock_guard<std::mutex> lk(live_mu_);
    if (live_.erase(id) > 0) {
        --load_[engine_of(id)];
    }
}

EngineLoop::Stats EngineGroup::stats() {
    EngineLoop::Stats sum;
    for (const auto& l : loops_) {
        const EngineLoop::Stats s = l->stats();
        sum.free_blocks += s.free_blocks;
        sum.total_blocks += s.total_blocks;
        sum.prefix_reused_tokens += s.prefix_reused_tokens;
        sum.spec_accepted_tokens += s.spec_accepted_tokens;
        sum.spec_steps += s.spec_steps;
        sum.spec_drafted_tokens += s.spec_drafted_tokens;
        if (sum.spec_draft_counts.size() < s.spec_draft_counts.size()) {
            sum.spec_draft_counts.resize(s.spec_draft_counts.size());
        }
        for (std::size_t i = 0; i < s.spec_draft_counts.size(); ++i) {
            sum.spec_draft_counts[i] += s.spec_draft_counts[i];
        }
        sum.disk_hits += s.disk_hits;
        sum.disk_loaded_tokens += s.disk_loaded_tokens;
        sum.disk_swapped_tokens += s.disk_swapped_tokens;
        sum.disk_refused_tokens += s.disk_refused_tokens;
    }
    return sum;
}

}  // namespace locus::server
