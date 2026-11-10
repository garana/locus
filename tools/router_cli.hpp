#pragma once

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
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
    "given as HOST:PORT, or as srv:NAME for the targets of NAME's DNS\n"
    "SRV records, looked up again on every refresh, for example\n"
    "srv:_locus._tcp.example.com, or as scan:A.B.C.D/N:PORT for the\n"
    "servers that answer on PORT in that IPv4 range, N from 22\n"
    "to 32, scanned about once a minute) and proxies each request to an\n"
    "instance that serves its kind: text, image-to-text, image, video\n"
    "or speech, as the instance lists in its /v1/models. A request\n"
    "that names a model goes to an instance with that model when\n"
    "there is one. A scan admits any server that answers there, locus\n"
    "or not, and sends it requests (and, with --upstream-auth client,\n"
    "the clients' keys): scan only a range whose hosts you trust.\n"
    "\n"
    "options:\n"
    "  --listen HOST:PORT  address to serve on (default 0.0.0.0:8080)\n"
    "  --refresh-ms N      how often each instance's /v1/models is\n"
    "                      read again (default 10000)\n"
    "  --auth-helper PATH  check client keys with this auth helper\n"
    "                      (locus-auth-helper is a reference one): every\n"
    "                      request but /health needs a key it allows\n"
    "  --auth-helper-arg A an argument for the helper (repeatable)\n"
    "  --upstream-auth M   the credential sent to the instances: client\n"
    "                      (the client's own, default), none, or key\n"
    "  --instance-keys F   a file of \"HOST:PORT KEY\" lines (\"* KEY\"\n"
    "                      for any other instance given as HOST:PORT;\n"
    "                      one found by SRV or registration needs its\n"
    "                      own line), so keys stay off the command\n"
    "                      line: the key each instance gets with\n"
    "                      --upstream-auth key, and with its\n"
    "                      /v1/models poll in any mode\n"
    "  --register-keys F   let instances register themselves (POST\n"
    "                      /locus/register, from locus-server\n"
    "                      --register) with a key from file F, one per\n"
    "                      line; without it registration is off. Guard\n"
    "                      these keys like client keys: requests (and\n"
    "                      with --upstream-auth client, the clients'\n"
    "                      keys) go to the address a holder registers\n"
    "  -h, --help          show this help and exit\n";

/**
 * Parses the router's arguments (argv[first..]) into `cfg`.
 * @returns "" on success, else an error message.
 */
inline std::string parse_router_args(int argc, char** argv, int first,
                                     locus::router::RouterConfig& cfg,
                                     bool& help) {
    std::string helper;
    std::vector<std::string> helper_args;
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
        } else if (a == "--auth-helper" && i + 1 < argc) {
            helper = argv[++i];
        } else if (a == "--auth-helper-arg" && i + 1 < argc) {
            helper_args.emplace_back(argv[++i]);
        } else if (a == "--upstream-auth" && i + 1 < argc) {
            const std::string_view m = argv[++i];
            using UA = locus::router::RouterConfig::UpstreamAuth;
            if (m == "client") {
                cfg.upstream_auth = UA::kClient;
            } else if (m == "none") {
                cfg.upstream_auth = UA::kNone;
            } else if (m == "key") {
                cfg.upstream_auth = UA::kKey;
            } else {
                return "bad --upstream-auth: " + std::string(m) +
                       " (client, none or key)";
            }
        } else if (a == "--instance-keys" && i + 1 < argc) {
            std::ifstream f(argv[++i]);
            if (!f) {
                return std::string("cannot read --instance-keys ") + argv[i];
            }
            std::stringstream ss;
            ss << f.rdbuf();
            try {
                cfg.instance_keys = locus::router::parse_instance_keys(ss.str());
            } catch (const std::exception& e) {
                return e.what();
            }
        } else if (a == "--register-keys" && i + 1 < argc) {
            std::ifstream f(argv[++i]);
            if (!f) {
                return std::string("cannot read --register-keys ") + argv[i];
            }
            std::stringstream ss;
            ss << f.rdbuf();
            cfg.register_keys = locus::router::parse_register_keys(ss.str());
            if (cfg.register_keys.empty()) {
                return std::string("no keys in --register-keys ") + argv[i];
            }
        } else if (!a.empty() && a.front() == '-') {
            return "unknown option: " + std::string(a);
        } else {
            cfg.instances.emplace_back(a);
        }
    }
    // Instances that register themselves may be the only ones.
    if (!help && cfg.instances.empty() && cfg.register_keys.empty()) {
        return "no instances given (and no --register-keys)";
    }
    if (helper.empty() && !helper_args.empty()) {
        return "--auth-helper-arg given without --auth-helper";
    }
    if (!helper.empty()) {
        cfg.auth_helper_argv = {helper};
        cfg.auth_helper_argv.insert(cfg.auth_helper_argv.end(),
                                    helper_args.begin(), helper_args.end());
    }
    if (cfg.upstream_auth ==
            locus::router::RouterConfig::UpstreamAuth::kKey &&
        cfg.instance_keys.empty()) {
        return "--upstream-auth key needs --instance-keys";
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
