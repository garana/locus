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
    "to 32, scanned about once a minute, or as mdns:SERVICE for the\n"
    "instances a DNS-SD browse for SERVICE.local finds on the LAN, for\n"
    "example mdns:_locus._tcp, browsed on every refresh; advertise one\n"
    "with dns-sd -R NAME _locus._tcp local PORT on macOS or\n"
    "avahi-publish-service NAME _locus._tcp PORT on Linux) and proxies\n"
    "each request to an instance that serves its kind: text,\n"
    "image-to-text, image, video, speech or rerank, as the instance\n"
    "lists in its /v1/models. A request\n"
    "that names a model goes to an instance with that model when\n"
    "there is one. A scan or an mDNS browse admits any server that\n"
    "answers, locus or not, and sends it requests (and, with\n"
    "--upstream-auth client, the clients' keys): use them only on a\n"
    "network whose hosts you trust.\n"
    "\n"
    "options:\n"
    "  --listen HOST:PORT  address to serve on (default :8080: every\n"
    "                      interface, IPv6 and IPv4; 0.0.0.0:PORT for\n"
    "                      IPv4 only)\n"
    "  --refresh-ms N      how often each instance's /v1/models is\n"
    "                      read again (default 10000)\n"
    "  --media-timeout S   longest silence from an instance on an image,\n"
    "                      speech or video request, in seconds (default\n"
    "                      1800; 0 = none)\n"
    "  --text-timeout S    the same for every other request (default\n"
    "                      1800; 0 = none)\n"
    "  --auth-helper PATH  check client keys with this auth helper\n"
    "                      (locus-auth-helper is a reference one): every\n"
    "                      request but /health needs a key it allows\n"
    "                      (a binary or a script that starts with #!)\n"
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
        } else if (a == "--refresh-ms" && i + 1 < argc) {
            int n = 0;
            if (!parse_nonneg_int(argv[++i], n) || n == 0) {
                return std::string("bad --refresh-ms: ") + argv[i];
            }
            cfg.refresh_ms = n;
        } else if (a == "--media-timeout" && i + 1 < argc) {
            int n = 0;
            if (!parse_nonneg_int(argv[++i], n) || n > 86400) {
                return std::string("bad --media-timeout: ") + argv[i];
            }
            cfg.media_read_timeout_ms = n * 1000;
        } else if (a == "--text-timeout" && i + 1 < argc) {
            int n = 0;
            if (!parse_nonneg_int(argv[++i], n) || n > 86400) {
                return std::string("bad --text-timeout: ") + argv[i];
            }
            cfg.text_read_timeout_ms = n * 1000;
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
    if (const auto w = locus::router::discovery_auth_warning(cfg);
        !w.empty()) {
        std::fprintf(stderr, "locus-router: %s\n", w.c_str());
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
        // The bound host too, so a loopback-only router reads as one; an
        // IPv6 address in brackets, as locus-server prints it.
        if (cfg.host.empty()) {
            std::fprintf(stderr,
                         "locus-router listening on every interface, port "
                         "%d\n",
                         port);
        } else {
            const bool v6 = cfg.host.find(':') != std::string::npos;
            std::fprintf(stderr, "locus-router listening on %s%s%s:%d\n",
                         v6 ? "[" : "", cfg.host.c_str(), v6 ? "]" : "",
                         port);
        }
        r.wait();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-router: %s\n", e.what());
        return 2;
    }
    return 0;
}

}  // namespace locus_tools
