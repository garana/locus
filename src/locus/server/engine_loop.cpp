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

std::uint64_t EngineLoop::submit(
    std::vector<tok::TokenId> prompt, std::uint32_t max_new_tokens,
    model::SamplingParams sampling, std::uint64_t seed,
    std::unique_ptr<model::TokenConstraint> constraint) {
    std::uint64_t id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        id = engine_.submit(std::move(prompt), max_new_tokens,
                            sampling, seed, std::move(constraint));
    }
    work_cv_.notify_all();
    return id;
}

void EngineLoop::run() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
        const bool more = engine_.step();
        progress_cv_.notify_all();
        if (!more) {
            work_cv_.wait(lk, [this] {
                return stop_ || !engine_.idle();
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
