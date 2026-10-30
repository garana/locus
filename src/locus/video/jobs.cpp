#include "locus/video/jobs.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <random>

namespace locus::video {

namespace {

std::int64_t wall_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/** "video_" and 24 random hex digits, as OpenAI's ids look. */
std::string new_id() {
    static std::mutex mu;
    static std::mt19937_64 rng{std::random_device{}()};
    std::lock_guard<std::mutex> lk(mu);
    char buf[32];
    std::snprintf(buf, sizeof buf, "video_%016llx%08llx",
                  static_cast<unsigned long long>(rng()),
                  static_cast<unsigned long long>(rng() & 0xffffffffULL));
    return buf;
}

void remove_file(const std::string& path) {
    if (path.empty()) return;
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

}  // namespace

VideoJobs::VideoJobs(std::string dir, std::size_t max_pending,
                     std::int64_t ttl, std::function<std::int64_t()> now)
    : dir_(std::move(dir)),
      max_pending_(max_pending),
      ttl_(ttl),
      now_(now ? std::move(now) : wall_seconds) {
    std::filesystem::create_directories(dir_);
    thread_ = std::thread([this] { worker(); });
}

VideoJobs::~VideoJobs() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
    for (const auto& [id, job] : jobs_) remove_file(job.file);
    remove_file(dir_ + "/partial.mp4");
}

std::optional<VideoJob> VideoJobs::submit(VideoJob job, Runner run) {
    std::lock_guard<std::mutex> lk(mu_);
    sweep();
    if (pending_ >= max_pending_) return std::nullopt;
    job.id = new_id();
    job.status = "queued";
    job.progress = 0;
    job.created_at = now_();
    job.completed_at = job.expires_at = 0;
    job.error.clear();
    job.file.clear();
    jobs_[job.id] = job;
    queue_.emplace_back(job.id, std::move(run));
    ++pending_;
    cv_.notify_all();
    return job;
}

std::optional<VideoJob> VideoJobs::get(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    sweep();
    const auto it = jobs_.find(id);
    if (it == jobs_.end()) return std::nullopt;
    return it->second;
}

void VideoJobs::wait_idle() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return pending_ == 0 || stop_; });
}

void VideoJobs::sweep() {
    const std::int64_t t = now_();
    for (auto it = jobs_.begin(); it != jobs_.end();) {
        const VideoJob& j = it->second;
        if (j.expires_at != 0 && j.expires_at <= t) {
            remove_file(j.file);
            it = jobs_.erase(it);
        } else {
            ++it;
        }
    }
}

void VideoJobs::worker() {
    for (;;) {
        std::string id;
        Runner run;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            id = queue_.front().first;
            run = std::move(queue_.front().second);
            queue_.pop_front();
            jobs_[id].status = "in_progress";
        }
        // One file name while it is written, the job's once complete,
        // so a half-written file is never served.
        const std::string partial = dir_ + "/partial.mp4";
        const std::string final_path = dir_ + "/" + id + ".mp4";
        std::string error;
        bool cancelled = false;
        try {
            run(partial,
                [this, &id](int p) {
                    std::lock_guard<std::mutex> lk(mu_);
                    jobs_[id].progress = std::clamp(p, 0, 99);
                },
                [this] {
                    std::lock_guard<std::mutex> lk(mu_);
                    return !stop_;
                });
            std::lock_guard<std::mutex> lk(mu_);
            cancelled = stop_;
        } catch (const std::exception& e) {
            error = e.what();
        } catch (...) {
            error = "unknown error";
        }
        if (!cancelled && error.empty()) {
            std::error_code ec;
            std::filesystem::rename(partial, final_path, ec);
            if (ec) error = "video: cannot store the file: " + ec.message();
        } else {
            remove_file(partial);
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            VideoJob& j = jobs_[id];
            j.completed_at = now_();
            j.expires_at = j.completed_at + ttl_;
            if (cancelled) {
                j.status = "failed";
                j.error = "the server is shutting down";
            } else if (error.empty()) {
                j.status = "completed";
                j.progress = 100;
                j.file = final_path;
            } else {
                j.status = "failed";
                j.error = error;
            }
            --pending_;
        }
        cv_.notify_all();
    }
}

}  // namespace locus::video
