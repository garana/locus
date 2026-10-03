#include "locus/pipeline/stage_executor.hpp"

#include <unistd.h>

#include <exception>
#include <utility>

namespace locus::pipeline {

StageExecutor::StageExecutor(PipelineStage& stage, int wake_fd)
    : stage_(stage), wake_fd_(wake_fd) {
    thread_ = std::thread(&StageExecutor::run, this);
}

StageExecutor::~StageExecutor() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void StageExecutor::submit_step(int session, kv::PagedKvCache::Seq* seq,
                                Message in) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_.push_back(Job{session, false, seq, std::move(in)});
    }
    cv_.notify_one();
}

void StageExecutor::submit_release(int session,
                                   kv::PagedKvCache::Seq* seq) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_.push_back(Job{session, true, seq, Message{}});
    }
    cv_.notify_one();
}

void StageExecutor::drain(std::vector<Completion>& out) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& c : done_) {
        out.push_back(std::move(c));
    }
    done_.clear();
}

void StageExecutor::run() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) {
                return;
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }

        // Compute OUTSIDE the lock so submit()/drain() never block on a
        // forward pass. All stage mutation is on this one thread.
        Completion c;
        c.session = job.session;
        c.is_release = job.release;
        if (job.release) {
            stage_.reset(*job.seq);
        } else {
            try {
                c.out = stage_.step(*job.seq, job.in);
            } catch (const std::exception& e) {
                c.ok = false;
                c.err = e.what();
            } catch (...) {
                c.ok = false;
                c.err = "unknown error";
            }
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            done_.push_back(std::move(c));
        }
        // Wake the loop. One byte per completion; the loop drains all
        // ready completions per wake, so a coalesced read is fine.
        const char b = 1;
        ssize_t w = ::write(wake_fd_, &b, 1);
        (void)w;  // a full/broken wake pipe cannot be handled here; the
                  // loop's level-triggered drain recovers regardless.
    }
}

}  // namespace locus::pipeline
