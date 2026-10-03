#include "locus/pipeline/stage_executor.hpp"

#include <unistd.h>

#include <exception>
#include <unordered_set>
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
    // Fixed for the stage's backend; batching needs a backend that can
    // run a batched forward (CPU/CUDA, not Vulkan).
    const bool batch_ok = stage_.supports_batch();
    for (;;) {
        std::vector<Job> steps;
        Job rel;
        bool have_rel = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
            if (stop_ && jobs_.empty()) {
                return;
            }
            if (jobs_.front().release) {
                rel = std::move(jobs_.front());
                jobs_.pop_front();
                have_rel = true;
            } else {
                // Coalesce the leading run of step jobs into one batch,
                // stopping at the first release. This preserves FIFO
                // order (completions come back in submit order) and the
                // release-after-step ordering for any sequence, and it
                // never reorders a step across a release -- which could
                // change cache pressure (a step could find the pool full
                // where a preceding release would have freed blocks).
                //
                // Stop also at a SECOND job for a sequence already in the
                // batch: two steps for one session are consecutive
                // positions on one KV sequence, so they must run in order
                // (each advances the seq), not side by side in one batch.
                // Splitting them across batches keeps that order and
                // guarantees step_batch never sees a duplicate seq. In
                // normal serving this never triggers (the caller keeps
                // one step in flight per session); it makes the executor
                // robust if it does.
                std::unordered_set<kv::PagedKvCache::Seq*> seen;
                while (!jobs_.empty() && !jobs_.front().release &&
                       seen.find(jobs_.front().seq) == seen.end()) {
                    seen.insert(jobs_.front().seq);
                    steps.push_back(std::move(jobs_.front()));
                    jobs_.pop_front();
                }
            }
        }

        // Compute OUTSIDE the lock so submit()/drain() never block on a
        // forward pass. All stage mutation is on this one thread.
        std::vector<Completion> cs;
        if (have_rel) {
            stage_.reset(*rel.seq);
            Completion c;
            c.session = rel.session;
            c.is_release = true;
            cs.push_back(std::move(c));
        } else if (batch_ok) {
            std::vector<PipelineStage::BatchInput> items;
            items.reserve(steps.size());
            for (auto& j : steps) {
                items.push_back({j.seq, &j.in});
            }
            std::vector<PipelineStage::BatchOutput> res;
            try {
                stage_.step_batch(items, res);
            } catch (const std::exception& e) {
                // A whole-batch failure (not a per-entry one, which
                // step_batch isolates): fail every job so no session is
                // left without a completion.
                res.assign(steps.size(), PipelineStage::BatchOutput{});
                for (auto& r : res) {
                    r.ok = false;
                    r.err = e.what();
                }
            } catch (...) {
                res.assign(steps.size(), PipelineStage::BatchOutput{});
                for (auto& r : res) {
                    r.ok = false;
                    r.err = "unknown error";
                }
            }
            for (std::size_t i = 0; i < steps.size(); ++i) {
                Completion c;
                c.session = steps[i].session;
                c.ok = res[i].ok;
                c.out = std::move(res[i].out);
                c.err = std::move(res[i].err);
                cs.push_back(std::move(c));
            }
        } else {
            // Backend cannot batch (e.g. Vulkan): one step at a time,
            // same as before batching existed.
            for (auto& j : steps) {
                Completion c;
                c.session = j.session;
                try {
                    c.out = stage_.step(*j.seq, j.in);
                } catch (const std::exception& e) {
                    c.ok = false;
                    c.err = e.what();
                } catch (...) {
                    c.ok = false;
                    c.err = "unknown error";
                }
                cs.push_back(std::move(c));
            }
        }

        const std::size_t n = cs.size();
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& c : cs) {
                done_.push_back(std::move(c));
            }
        }
        // Wake the loop once per completion; it drains ALL ready
        // completions per wake, so a coalesced read is fine.
        for (std::size_t i = 0; i < n; ++i) {
            const char b = 1;
            ssize_t w = ::write(wake_fd_, &b, 1);
            (void)w;  // Result discarded: a full (EAGAIN) wake pipe
                      // already has a pending byte, so the loop still
                      // wakes and drains ALL completions; a broken pipe
                      // cannot be handled here. REQUIRES a non-blocking
                      // write end (see the ctor doc) -- a blocking one
                      // would stall the worker here if the loop were
                      // behind on draining.
        }
    }
}

}  // namespace locus::pipeline
