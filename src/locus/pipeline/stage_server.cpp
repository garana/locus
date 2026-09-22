#include "locus/pipeline/stage_server.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace locus::pipeline {

bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<Cidr>& allow,
                 const std::string& downstream_host,
                 int downstream_port, const StageConn& conn) {
    int in_fd = -1;
    for (;;) {
        std::string peer;
        const int fd = accept_one(listen_fd, &peer);
        if (fd < 0) {
            ::close(listen_fd);
            return false;
        }
        if (!ip_allowed(peer, allow)) {
            // Log the refused address (otherwise a mismatched allowlist
            // is invisible), then keep waiting for an allowed peer.
            std::fprintf(stderr,
                         "serve_stage: refused connection from %s "
                         "(not in --allow)\n",
                         peer.c_str());
            ::close(fd);
            continue;
        }
        in_fd = fd;
        break;
    }
    ::close(listen_fd);  // one input link per stage
    if (conn.recv_timeout_ms > 0) {
        set_recv_timeout(in_fd, conn.recv_timeout_ms);
    }

    // Connect downstream, retrying on a fixed wait (no backoff): a
    // stage may come up before its downstream. A successful connect is
    // the health signal, so there is no separate probe.
    int out_fd = -1;
    for (int attempt = 1;; ++attempt) {
        out_fd = connect_to(downstream_host, downstream_port,
                            conn.connect_timeout_ms);
        if (out_fd >= 0) {
            break;
        }
        if (conn.reconnect_attempts > 0 &&
            attempt >= conn.reconnect_attempts) {
            std::fprintf(stderr,
                         "serve_stage: downstream %s:%d unreachable "
                         "after %d attempt(s)\n",
                         downstream_host.c_str(), downstream_port,
                         attempt);
            ::close(in_fd);
            return false;
        }
        std::fprintf(stderr,
                     "serve_stage: downstream %s:%d connect failed, "
                     "retrying in %d ms\n",
                     downstream_host.c_str(), downstream_port,
                     conn.reconnect_wait_ms);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(conn.reconnect_wait_ms));
    }
    return stage.run(in_fd, out_fd);  // closes in_fd and out_fd
}

}  // namespace locus::pipeline
