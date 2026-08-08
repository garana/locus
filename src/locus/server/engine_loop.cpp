#include "locus/server/engine_loop.hpp"

namespace locus::server {

EngineLoop::EngineLoop(const model::LlamaModel& m,
                       tok::TokenId eos,
                       engine::Engine::Config cfg)
    : engine_(m, eos, cfg), worker_([this] { run(); }) {}

EngineLoop::~EngineLoop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    worker_.join();
}

std::string EngineLoop::admit_error(std::uint32_t n_prompt) {
    std::lock_guard<std::mutex> lk(mu_);
    return engine_.admit_error(n_prompt);
}

std::uint64_t EngineLoop::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint,
    engine::LogprobsOpt logprobs) {
    std::uint64_t id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        id = engine_.submit(std::move(prompt), max_new_tokens,
                            sampling, seed, std::move(constraint),
                            logprobs);
    }
    work_cv_.notify_all();
    return id;
}

void EngineLoop::run() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
        engine_.step();
        progress_cv_.notify_all();
        // Park unless there is work step() can make progress on now.
        // Gating on has_runnable_work (not just idle) keeps the
        // worker from busy-spinning while HOLDING mu_ on a backlog it
        // cannot admit -- the failure that wedged the whole server.
        if (!engine_.has_runnable_work()) {
            work_cv_.wait(lk, [this] {
                return stop_ || engine_.has_runnable_work();
            });
        }
    }
}

EngineLoop::View EngineLoop::snapshot_locked(
    std::uint64_t id) const {
    View v;
    if (const engine::Request* r = engine_.get(id)) {
        v.status = r->status;
        v.generated = r->generated;
        v.error = r->error;
        v.logprobs = r->logprobs;
    } else {
        v.status = engine::Status::kFailed;
        v.error = "unknown request id";
    }
    return v;
}

EngineLoop::View EngineLoop::view(std::uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    return snapshot_locked(id);
}

EngineLoop::View EngineLoop::wait_progress(std::uint64_t id,
                                           std::size_t n_seen) {
    std::unique_lock<std::mutex> lk(mu_);
    progress_cv_.wait(lk, [&] {
        View v = snapshot_locked(id);
        return v.generated.size() > n_seen ||
               v.status == engine::Status::kDone ||
               v.status == engine::Status::kFailed;
    });
    return snapshot_locked(id);
}

EngineLoop::Stats EngineLoop::stats() {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.free_blocks = engine_.free_blocks();
    s.total_blocks = engine_.total_blocks();
    s.prefix_reused_tokens = engine_.prefix_reused_tokens();
    s.spec_accepted_tokens = engine_.spec_accepted_tokens();
    s.spec_steps = engine_.spec_steps();
    return s;
}

EngineLoop::View EngineLoop::wait_done(std::uint64_t id) {
    std::unique_lock<std::mutex> lk(mu_);
    progress_cv_.wait(lk, [&] {
        View v = snapshot_locked(id);
        return v.status == engine::Status::kDone ||
               v.status == engine::Status::kFailed;
    });
    return snapshot_locked(id);
}

}  // namespace locus::server
