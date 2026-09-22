#include "locus/pipeline/stage_server.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace locus::pipeline {

namespace {

// Accepts one allowed input connection on listen_fd, logging (and
// dropping) peers outside `allow`. @returns the fd, or -1 if accept
// fails.
int accept_allowed(int listen_fd, const std::vector<Cidr>& allow) {
    for (;;) {
        std::string peer;
        const int fd = accept_one(listen_fd, &peer);
        if (fd < 0) {
            return -1;
        }
        if (!ip_allowed(peer, allow)) {
            std::fprintf(stderr,
                         "serve_stage: refused connection from %s "
                         "(not in --allow)\n",
                         peer.c_str());
            ::close(fd);
            continue;
        }
        return fd;
    }
}

// Connects to downstream_host:port, retrying on a fixed wait (no
// backoff). @returns the fd, or -1 if it gives up (reconnect_attempts).
int connect_downstream(const std::string& host, int port,
                       const StageConn& conn) {
    for (int attempt = 1;; ++attempt) {
        const int fd = connect_to(host, port, conn.connect_timeout_ms);
        if (fd >= 0) {
            return fd;
        }
        if (conn.reconnect_attempts > 0 &&
            attempt >= conn.reconnect_attempts) {
            std::fprintf(stderr,
                         "serve_stage: downstream %s:%d unreachable "
                         "after %d attempt(s)\n",
                         host.c_str(), port, attempt);
            return -1;
        }
        std::fprintf(stderr,
                     "serve_stage: downstream %s:%d connect failed, "
                     "retrying in %d ms\n",
                     host.c_str(), port, conn.reconnect_wait_ms);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(conn.reconnect_wait_ms));
    }
}

}  // namespace

bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn) {
    int served = 0;
    bool all_clean = true;  // any session ended abnormally?
    for (;;) {
        // Automatic reconnect: each session re-accepts a fresh input
        // and re-dials the downstream, so a dropped peer (detected by
        // keepalive, or a clean EOF) is recovered by serving the next
        // connection rather than exiting. Per the terminal-kTimeout
        // contract, a session that ends always tears its fds down (in
        // stage.run) and we re-accept -- never re-read a dead fd.
        const int in_fd = accept_allowed(listen_fd, allow);
        if (in_fd < 0) {
            ::close(listen_fd);
            return false;  // accept failed (listener broken)
        }
        set_keepalive(in_fd, conn.keepalive_idle_s,
                      conn.keepalive_intvl_s, conn.keepalive_count);
        if (conn.recv_timeout_ms > 0) {
            set_recv_timeout(in_fd, conn.recv_timeout_ms);
        }

        const int out_fd =
            connect_downstream(downstream_host, downstream_port, conn);
        if (out_fd < 0) {
            ::close(in_fd);
            ::close(listen_fd);
            return false;  // downstream unreachable within the budget
        }
        set_keepalive(out_fd, conn.keepalive_idle_s,
                      conn.keepalive_intvl_s, conn.keepalive_count);

        stage.reset();  // fresh KV state for this session's sequence
        if (!stage.run(in_fd, out_fd)) {  // closes in_fd and out_fd
            all_clean = false;
            std::fprintf(stderr,
                         "serve_stage: session ended abnormally; "
                         "re-accepting\n");
        }
        ++served;
        if (conn.serve_sessions > 0 && served >= conn.serve_sessions) {
            ::close(listen_fd);
            return all_clean;  // false if any session ended abnormally
        }
    }
}

}  // namespace locus::pipeline
