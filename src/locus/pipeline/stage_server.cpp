#include "locus/pipeline/stage_server.hpp"

#include <unistd.h>

namespace locus::pipeline {

bool serve_stage(PipelineStage& stage, int listen_fd,
                 const std::vector<CidrV4>& allow,
                 const std::string& downstream_host,
                 int downstream_port) {
    int in_fd = -1;
    for (;;) {
        std::string peer;
        const int fd = accept_one(listen_fd, &peer);
        if (fd < 0) {
            ::close(listen_fd);
            return false;
        }
        if (!ip_allowed(peer, allow)) {
            ::close(fd);  // peer outside the allowed range; keep waiting
            continue;
        }
        in_fd = fd;
        break;
    }
    ::close(listen_fd);  // one input link per stage

    const int out_fd = connect_to(downstream_host, downstream_port);
    if (out_fd < 0) {
        ::close(in_fd);
        return false;
    }
    return stage.run(in_fd, out_fd);  // closes in_fd and out_fd
}

}  // namespace locus::pipeline
