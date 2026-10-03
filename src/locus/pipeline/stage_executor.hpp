#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "locus/kv/paged_cache.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/stage.hpp"

namespace locus::pipeline {

/**
 * Runs a stage's forward work off the I/O loop thread (event-loop
 * serving model, DESIGN.md i#24 increment 1). The loop submits jobs and
 * reads completions; a single worker thread does the compute, so a slow
 * forward no longer blocks the loop from reading, accepting or flushing
 * other sessions.
 *
 * It owns ALL mutation of the stage: both step() (which grows/reads the
 * KV cache) and release() (which frees a sequence's blocks) touch the
 * stage's shared cache and workspace, which are not thread-safe. Running
 * both only on this one worker thread keeps that access single-threaded.
 * While an executor is running the caller MUST route every stage
 * mutation through here -- it must not call stage.step()/reset() itself
 * concurrently. The contract is about concurrency, not permanent
 * ownership: once the executor is DESTROYED (joined) the stage is
 * single-threaded again and direct step()/reset() is safe. The shutdown
 * path relies on this -- destroy the executor, then reset any remaining
 * sessions directly, rather than routing teardown releases through it.
 * (The guarantee is a convention, not enforced: both the loop and the
 * executor hold the same PipelineStage&.)
 *
 * One worker thread, so exactly one step runs at a time (increment 1
 * keeps today's one-forward-at-a-time behavior; batching is increment
 * 2). `session` is an opaque key the caller matches completions by; the
 * executor never interprets it.
 *
 * A `seq` passed to submit_*() must outlive the EXECUTOR, not merely
 * survive until its completion is drained: the destructor drains the
 * queued backlog (so queued releases -- and steps -- still run at
 * teardown), touching those seqs then. So the caller must not free a
 * session's seq, or clear its session map, before the executor is
 * destroyed. (Draining the backlog on stop is deliberate -- dropping
 * queued releases would leak KV blocks -- and bounded: one step in
 * flight per session caps the backlog at the session count.)
 *
 * Completion signalling: one byte is written to `wake_fd` whenever a
 * completion becomes available, so a poll loop can wait on it. It is
 * level-triggered in spirit -- one wake may cover several completions --
 * so drain() returns all ready completions and the caller loops, it does
 * not assume one completion per wake.
 */
class StageExecutor {
  public:
    /** One finished job. `is_release` true means a release() finished
     * (the seq's KV is freed, the caller may now free the session);
     * false means a step() finished, with `out` the output frame when
     * `ok`, or `ok`=false and `err` set if step() threw. */
    struct Completion {
        int session = -1;
        bool is_release = false;
        bool ok = true;
        Message out;
        std::string err;
    };

    /**
     * @param stage The stage to run; must outlive the executor.
     * @param wake_fd Write end of a pipe the caller polls; the executor
     *     writes one byte per completion. Must outlive the executor; the
     *     executor does not close it. It MUST be non-blocking: the
     *     executor discards the write result, so a blocking write end
     *     whose completions are not being drained would stall the worker
     *     inside write() and stop all job processing. (stage_main's
     *     make_wake_pipe sets both ends non-blocking -- reuse it.)
     */
    StageExecutor(PipelineStage& stage, int wake_fd);
    ~StageExecutor();
    StageExecutor(const StageExecutor&) = delete;
    StageExecutor& operator=(const StageExecutor&) = delete;

    /** Queues a forward step for `seq` on `in`; its Completion carries
     * the output frame. `seq` must stay valid until drained. */
    void submit_step(int session, kv::PagedKvCache::Seq* seq, Message in);

    /** Queues release of `seq`'s KV blocks; its Completion (is_release)
     * signals the caller may free the session. Ordered after any earlier
     * step for the same seq, so releasing a just-stepped session is
     * safe. */
    void submit_release(int session, kv::PagedKvCache::Seq* seq);

    /** Moves all ready completions onto the end of `out` (appends; does
     * not clear). Non-blocking. */
    void drain(std::vector<Completion>& out);

  private:
    struct Job {
        int session = -1;
        bool release = false;
        kv::PagedKvCache::Seq* seq = nullptr;
        Message in;
    };

    void run();

    PipelineStage& stage_;
    int wake_fd_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    std::vector<Completion> done_;
    bool stop_ = false;
    std::thread thread_;
};

}  // namespace locus::pipeline
