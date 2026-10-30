#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace locus::video {

/** One video job, as GET /v1/videos/{id} reports it. */
struct VideoJob {
    std::string id;
    std::string status = "queued";  ///< queued, in_progress, completed, failed
    int progress = 0;               ///< 0 to 100
    std::int64_t created_at = 0;    ///< Unix seconds
    std::int64_t completed_at = 0;  ///< 0 until done
    std::int64_t expires_at = 0;    ///< 0 until done
    std::string model, prompt, size, seconds;
    std::string error;  ///< the failure message
    std::string file;   ///< the MP4, once completed
};

/**
 * The video jobs of a server: a queue served by one worker thread, so
 * one clip is made at a time. Finished jobs (and their files) expire
 * ttl seconds after they finish.
 */
class VideoJobs {
  public:
    /**
     * Makes one clip: writes the MP4 to out_path, reports progress
     * (0 to 100) as it goes, and polls alive (false: stop soon).
     * Throws to fail the job (the message becomes its error).
     */
    using Runner = std::function<void(const std::string& out_path,
                                      const std::function<void(int)>& progress,
                                      const std::function<bool()>& alive)>;

    /**
     * @param dir Where the MP4 files go (created when missing).
     * @param max_pending Most jobs queued or in progress at once.
     * @param ttl Seconds a finished job and its file are kept.
     * @param now The clock (Unix seconds); tests pass their own.
     */
    VideoJobs(std::string dir, std::size_t max_pending, std::int64_t ttl,
              std::function<std::int64_t()> now = {});
    /** Stops the worker (a running job sees alive() false) and removes
     *  every file. */
    ~VideoJobs();
    VideoJobs(const VideoJobs&) = delete;
    VideoJobs& operator=(const VideoJobs&) = delete;

    /**
     * Queues a job. job's id, status, progress and times are set here.
     *
     * @returns The queued job, or nothing when max_pending jobs are
     *     already queued or running.
     */
    std::optional<VideoJob> submit(VideoJob job, Runner run);

    /** @returns The job, or nothing when unknown or expired. */
    std::optional<VideoJob> get(const std::string& id);

    /** Blocks until no job is queued or running (for tests). */
    void wait_idle();

  private:
    void worker();
    /** Drops finished jobs past their expiry; called with mu_ held. */
    void sweep();

    std::string dir_;
    std::size_t max_pending_;
    std::int64_t ttl_;
    std::function<std::int64_t()> now_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<std::string, VideoJob> jobs_;
    std::deque<std::pair<std::string, Runner>> queue_;
    std::size_t pending_ = 0;  ///< queued + in progress
    bool stop_ = false;
    std::thread thread_;
};

}  // namespace locus::video
