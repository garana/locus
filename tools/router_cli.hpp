#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "cli_spec.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/router/router.hpp"

namespace locus_tools {

inline constexpr const char* kRouterUsage =
    "usage: locus-router [--listen HOST:PORT] [--refresh-ms N] "
    "INSTANCE...\n"
    "       locus-server --router [options] INSTANCE...\n"
    "\n"
    "Serves one URL in front of locus instances (each a locus-server,\n"
    "given as HOST:PORT) and proxies each request to an instance that\n"
    "serves its kind: text, image-to-text, image, video or speech, as\n"
    "the instance lists in its /v1/models. A request that names a\n"
    "model goes to an instance with that model when there is one.\n"
    "\n"
    "options:\n"
    "  --listen HOST:PORT  address to serve on (default 0.0.0.0:8080)\n"
    "  --refresh-ms N      how often each instance's /v1/models is\n"
    "                      read again (default 10000)\n"
    "  -h, --help          show this help and exit\n";

/**
 * Parses the router's arguments (argv[first..]) into `cfg`.
 * @returns "" on success, else an error message.
 */
inline std::string parse_router_args(int argc, char** argv, int first,
                                     locus::router::RouterConfig& cfg,
                                     bool& help) {
    for (int i = first; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "-h" || a == "--help") {
            help = true;
        } else if (a == "--listen" && i + 1 < argc) {
            if (!locus::pipeline::parse_hostport(argv[++i], cfg.host,
                                                 cfg.port)) {
                return std::string("bad --listen address: ") + argv[i];
            }
            if (cfg.host.empty()) cfg.host = "0.0.0.0";
        } else if (a == "--refresh-ms" && i + 1 < argc) {
            int n = 0;
            if (!parse_nonneg_int(argv[++i], n) || n == 0) {
                return std::string("bad --refresh-ms: ") + argv[i];
            }
            cfg.refresh_ms = n;
        } else if (!a.empty() && a.front() == '-') {
            return "unknown option: " + std::string(a);
        } else {
            cfg.instances.emplace_back(a);
        }
    }
    if (!help && cfg.instances.empty()) {
        return "no instances given";
    }
    return "";
}

/** Runs the router until it is stopped. @returns the exit status. */
inline int run_router(int argc, char** argv, int first) {
    locus::router::RouterConfig cfg;
    bool help = false;
    const std::string err = parse_router_args(argc, argv, first, cfg, help);
    if (help) {
        std::fputs(kRouterUsage, stdout);
        return 0;
    }
    if (!err.empty()) {
        std::fprintf(stderr, "locus-router: %s\n%s", err.c_str(),
                     kRouterUsage);
        return 2;
    }
    try {
        locus::router::Router r(cfg);
        const int port = r.start();
        if (port < 0) {
            std::fprintf(stderr, "locus-router: cannot listen on %s:%d\n",
                         cfg.host.c_str(), cfg.port);
            return 1;
        }
        for (const auto& in : r.instances()) {
            std::string kinds;
            for (const auto k : in.kinds) {
                kinds += (kinds.empty() ? "" : ",") +
                         std::string(locus::router::kind_name(k));
            }
            std::fprintf(stderr, "instance %s:%d %s %s\n", in.host.c_str(),
                         in.port, in.healthy ? "up" : "down",
                         kinds.c_str());
        }
        std::fprintf(stderr, "locus-router listening on :%d\n", port);
        r.wait();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-router: %s\n", e.what());
        return 2;
    }
    return 0;
}

}  // namespace locus_tools
