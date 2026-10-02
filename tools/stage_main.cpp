#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <set>
#include <string>
#include <vector>

#include "locus/config/config_file.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/llama.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/pipeline/stage_server.hpp"
#include "stage_config.hpp"

namespace {

const char* kUsage =
    "usage: locus-stage --model <m.gguf> --layers A:B "
    "--listen HOST:PORT --downstream HOST:PORT... [--allow CIDR]...\n"
    "\n"
    "Runs one pipeline stage (multi-server, DESIGN.md R15+): accepts an\n"
    "input connection on --listen, runs the model's layers [A,B), and\n"
    "sends its output to a downstream. --listen HOST may be empty for\n"
    "the wildcard address (\":PORT\"). --downstream is repeatable: give\n"
    "it more than once to name a pool of interchangeable replicas (no\n"
    "load balancer -- successive sessions round-robin across the pool,\n"
    "and each connect uses the first replica that accepts, so a dead\n"
    "one is skipped). A refused replica is skipped for free, but a\n"
    "filtered (blackholed) one listed before a live one costs up to\n"
    "--connect-timeout per session. --allow restricts which peer\n"
    "addresses may connect and is repeatable (e.g. --allow 10.0.0.0/8);\n"
    "with no --allow, any peer may connect.\n"
    "\n"
    "--config FILE reads a key=value file: every flag below has a key\n"
    "(the flag name without the dashes, e.g. connect-timeout). CLI flags\n"
    "are read first, then the file overrides/adds to them (the file\n"
    "wins); downstream and allow may repeat, and if the file gives\n"
    "either, its lines replace the CLI's. A '#' starts a comment only\n"
    "at the beginning of a line (it is kept verbatim inside a value).\n"
    "\n"
    "Flags (and the matching config keys):\n";

// Parses "A:B" into a half-open layer range, requiring A < B.
bool parse_layers(const std::string& s, std::uint32_t& a,
                  std::uint32_t& b) {
    const auto c = s.find(':');
    if (c == std::string::npos) {
        return false;
    }
    char* e1 = nullptr;
    char* e2 = nullptr;
    const long la = std::strtol(s.substr(0, c).c_str(), &e1, 10);
    const std::string bs = s.substr(c + 1);
    const long lb = std::strtol(bs.c_str(), &e2, 10);
    if (*e1 != '\0' || *e2 != '\0' || la < 0 || lb < 0 || la >= lb) {
        return false;
    }
    a = static_cast<std::uint32_t>(la);
    b = static_cast<std::uint32_t>(lb);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const auto spec = locus_tools::stage_spec();
    locus_tools::StageOptions opt;
    std::string config_path;
    std::set<std::string> seen;  // directives set via CLI or config

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            std::printf("%s%s", kUsage, spec.help().c_str());
            return 0;
        }
        if (a == "--config") {
            config_path = next("--config");
            continue;
        }
        if (a.rfind("--", 0) != 0) {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
        const std::string name = a.substr(2);
        const auto* d = spec.find(name);
        if (d == nullptr) {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
        const std::string value =
            d->takes_value() ? next(a.c_str()) : std::string();
        try {
            spec.apply(*d, opt, value);  // validates kInt
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 2;
        }
        seen.insert(name);
    }

    // CLI-then-file precedence: the config file overrides/adds to what
    // the flags set (the file wins). i#19, DESIGN.md R15+.
    if (!config_path.empty()) {
        try {
            const auto cfg =
                locus::config::ConfigFile::parse(config_path);
            spec.apply_config(cfg, opt, seen);
        } catch (const locus::config::ConfigFile::Error& e) {
            std::fprintf(stderr, "config: %s\n", e.message.c_str());
            return 2;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "config: %s\n", e.what());
            return 2;
        }
    }

    if (const std::string miss = spec.first_missing_required(seen);
        !miss.empty()) {
        std::fprintf(stderr,
                     "missing required option: %s (--%s or config key "
                     "%s)\n",
                     miss.c_str(), miss.c_str(), miss.c_str());
        return 2;
    }

    std::uint32_t la = 0, lb = 0;
    if (!parse_layers(opt.layers, la, lb)) {
        std::fprintf(stderr, "bad --layers (want A:B with A < B)\n");
        return 2;
    }
    std::string lh;
    int lp = 0;
    if (!locus::pipeline::parse_hostport(opt.listen, lh, lp)) {
        std::fprintf(stderr, "bad --listen (want HOST:PORT)\n");
        return 2;
    }
    std::vector<locus::pipeline::HostPort> pool;
    std::string pool_desc;  // for the startup log line
    for (const auto& ds : opt.downstream) {
        std::string dh;
        int dp = 0;
        if (!locus::pipeline::parse_hostport(ds, dh, dp)) {
            std::fprintf(stderr,
                         "bad --downstream (want HOST:PORT): %s\n",
                         ds.c_str());
            return 2;
        }
        pool.push_back({dh, dp});
        if (!pool_desc.empty()) {
            pool_desc += ", ";
        }
        pool_desc += ds;
    }
    std::vector<locus::pipeline::Cidr> allow;
    for (const auto& s : opt.allow) {
        const auto c = locus::pipeline::Cidr::parse(s);
        if (!c) {
            std::fprintf(stderr, "bad --allow CIDR: %s\n", s.c_str());
            return 2;
        }
        allow.push_back(*c);
    }

    try {
        auto g = locus::gguf::GgufFile::open(opt.model);
        // Slice-only loading: this process wires up only layers [la, lb)
        // (its share of the model), so cluster memory is ~1x the model.
        // hparams().n_layers stays the full count, so the range check
        // below still validates against the whole model.
        auto model = locus::model::LlamaModel::load(g, la, lb);
        if (lb > model.hparams().n_layers) {
            std::fprintf(stderr,
                         "--layers end %u exceeds model n_layers %u\n",
                         lb, model.hparams().n_layers);
            return 2;
        }
        locus::pipeline::PipelineStage stage(model, la, lb);
        const int lfd = locus::pipeline::listen_on(lh, lp, nullptr);
        if (lfd < 0) {
            std::fprintf(stderr, "listen on %s failed\n",
                         opt.listen.c_str());
            return 1;
        }
        std::fprintf(stderr,
                     "locus-stage: layers [%u,%u) on %s -> %s\n", la,
                     lb, opt.listen.c_str(), pool_desc.c_str());
        locus::pipeline::StageConn conn;
        conn.connect_timeout_ms = opt.connect_timeout;
        conn.reconnect_wait_ms = opt.reconnect_wait;
        conn.reconnect_attempts = opt.reconnect_attempts;
        conn.recv_timeout_ms = opt.read_timeout;
        conn.keepalive_idle_s = opt.keepalive_idle;
        conn.keepalive_intvl_s = opt.keepalive_interval;
        conn.keepalive_count = opt.keepalive_count;
        conn.serve_sessions = opt.sessions;
        const bool ok = locus::pipeline::serve_stage(stage, lfd, allow,
                                                     pool, conn);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-stage: %s\n", e.what());
        return 1;
    }
}
