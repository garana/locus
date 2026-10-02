#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <csignal>
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

// SIGHUP reload plumbing (i#19). The handler does only async-signal-
// safe work: set a flag and write one byte to the wake pipe. serve_stage
// watches the pipe in its event loop (sys::Poller) and does the actual
// reload between sessions. g_wake_w is the pipe write end.
volatile std::sig_atomic_t g_reload = 0;
volatile std::sig_atomic_t g_wake_w = -1;  // read in the handler too

void on_sighup(int /*sig*/) {
    g_reload = 1;
    if (g_wake_w >= 0) {
        const char b = 1;
        const ssize_t n = ::write(g_wake_w, &b, 1);  // signal-safe
        (void)n;  // a full pipe (EAGAIN) is fine: a wake is already queued
    }
}

void install_sighup() {
    struct sigaction sa;
    sa.sa_handler = on_sighup;
    sigemptyset(&sa.sa_mask);  // macro on some platforms; no :: qualifier
    sa.sa_flags = 0;
    sigaction(SIGHUP, &sa, nullptr);
}

// Creates the self-pipe the SIGHUP handler pokes and serve_stage polls.
// Both ends are non-blocking: the handler's write must never block, and
// serve_stage drains the read end without blocking. @returns the read
// end, or -1 on failure (reload then simply has no wake fd and falls
// back to applying on the next peer).
int make_wake_pipe() {
    int fds[2];
    if (::pipe(fds) != 0) {
        return -1;
    }
    for (const int fd : fds) {
        const int fl = ::fcntl(fd, F_GETFL, 0);
        if (fl >= 0) {
            ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        }
    }
    g_wake_w = fds[1];
    return fds[0];
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

    // The CLI-only options, kept immutable as the reload baseline: each
    // SIGHUP reload starts from a copy of this and re-overlays the file,
    // so a key removed from the file reverts to its CLI value instead of
    // sticking (i#19 part 2).
    const locus_tools::StageOptions cli_baseline = opt;

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

    // One validation+conversion path, shared with reload.
    locus_tools::StageRuntime rt;
    if (const std::string err = locus_tools::build_runtime(opt, rt);
        !err.empty()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    std::string pool_desc;  // for the startup log line
    for (const auto& ds : opt.downstream) {
        if (!pool_desc.empty()) {
            pool_desc += ", ";
        }
        pool_desc += ds;
    }

    try {
        auto g = locus::gguf::GgufFile::open(opt.model);
        // Slice-only loading: this process wires up only layers
        // [begin, end) (its share of the model), so cluster memory is
        // ~1x the model. hparams().n_layers stays the full count, so
        // the range check still validates against the whole model.
        auto model = locus::model::LlamaModel::load(g, rt.layer_begin,
                                                    rt.layer_end);
        if (rt.layer_end > model.hparams().n_layers) {
            std::fprintf(stderr,
                         "--layers end %u exceeds model n_layers %u\n",
                         rt.layer_end, model.hparams().n_layers);
            return 2;
        }
        locus::pipeline::PipelineStage stage(model, rt.layer_begin,
                                             rt.layer_end);
        const int lfd = locus::pipeline::listen_on(rt.listen_host,
                                                   rt.listen_port,
                                                   nullptr);
        if (lfd < 0) {
            std::fprintf(stderr, "listen on %s failed\n",
                         opt.listen.c_str());
            return 1;
        }
        std::fprintf(stderr,
                     "locus-stage: layers [%u,%u) on %s -> %s\n",
                     rt.layer_begin, rt.layer_end, opt.listen.c_str(),
                     pool_desc.c_str());

        // Cached hostname resolution for downstream names (i#40): the
        // Resolver turns a name into a cached IP so a dial avoids a
        // per-connection getaddrinfo; a numeric-IP downstream passes
        // through untouched. Pre-warm the configured pool HERE, before
        // serving, so the first dial hits the cache rather than blocking
        // on DNS on the serve loop -- a blocking lookup is fine now
        // because nothing is being served yet. (A reload adds new names
        // with prime(), which is non-blocking, because reload.apply runs
        // on the serve loop; see below.)
        locus::pipeline::Resolver resolver(
            locus_tools::resolver_options(opt));
        for (const auto& hp : rt.pool) {
            resolver.resolve(hp.host);
        }
        rt.conn.resolver = &resolver;

        // Hot reload (SIGHUP, i#19): re-read the config from the
        // immutable CLI baseline and apply the reloadable subset
        // (allowlist, pool, connection policy). model, layers and the
        // listen bind are fixed for a running stage -- a file that
        // changes them is warned about and otherwise ignored. A reload
        // that fails to parse/validate is logged and the running config
        // is kept.
        locus::pipeline::StageReload reload;
        if (!config_path.empty()) {
            // Create the wake pipe (sets g_wake_w) BEFORE installing the
            // handler, so a SIGHUP can never run with g_wake_w == -1.
            reload.wake_fd = make_wake_pipe();
            reload.flag = &g_reload;
            install_sighup();
            reload.apply =
                [&](std::vector<locus::pipeline::Cidr>& a,
                    std::vector<locus::pipeline::HostPort>& p,
                    locus::pipeline::StageConn& c) {
                    // A reload must never kill a running stage, whatever
                    // the file or parser throws (ConfigFile::Error, a
                    // bad-int runtime_error, anything): catch everything
                    // and keep the current config.
                    try {
                        locus_tools::StageOptions fresh = cli_baseline;
                        const auto cfg =
                            locus::config::ConfigFile::parse(config_path);
                        std::set<std::string> rseen;
                        spec.apply_config(cfg, fresh, rseen);
                        locus_tools::StageRuntime rt2;
                        if (const std::string err =
                                locus_tools::build_runtime(fresh, rt2);
                            !err.empty()) {
                            std::fprintf(stderr,
                                         "reload: %s; keeping current "
                                         "config\n",
                                         err.c_str());
                            return;
                        }
                        // Fixed for a running stage: warn and ignore.
                        // sessions is the lifetime budget compared
                        // against a process-start counter, so reloading
                        // it could exit a long-up stage immediately.
                        if (fresh.model != opt.model) {
                            std::fprintf(stderr,
                                         "reload: model cannot change on "
                                         "a running stage; ignoring\n");
                        }
                        if (fresh.layers != opt.layers) {
                            std::fprintf(stderr,
                                         "reload: layers cannot change "
                                         "on a running stage; ignoring\n");
                        }
                        if (fresh.listen != opt.listen) {
                            std::fprintf(stderr,
                                         "reload: listen cannot change "
                                         "on a running stage; ignoring\n");
                        }
                        if (fresh.sessions != opt.sessions) {
                            std::fprintf(stderr,
                                         "reload: sessions cannot change "
                                         "on a running stage; ignoring\n");
                        }
                        a = rt2.allow;
                        p = rt2.pool;
                        c = rt2.conn;
                        c.serve_sessions = rt.conn.serve_sessions;  // keep
                        c.resolver = &resolver;  // rt2.conn cleared it
                        // Warm any newly-added names WITHOUT blocking:
                        // this runs on the serve loop, so prime() records
                        // intent and the refresh thread fills it a tick
                        // later. A blocking resolve() here would freeze
                        // every active session for a DNS timeout.
                        //
                        // Operator note: a reload that renames the WHOLE
                        // pool leaves every replica primed-but-unfilled
                        // until the next refresh tick (~1 s), during which
                        // the next dial finds nothing live and retries per
                        // the reconnect policy. With reconnect_attempts=0
                        // (the default) that is a ~1 s stall; with a small
                        // reconnect_attempts and reconnect_wait a one-shot
                        // full rename can exhaust the budget and the stage
                        // gives up before the names warm.
                        for (const auto& hp : p) {
                            resolver.prime(hp.host);
                        }
                        std::fprintf(stderr,
                                     "reload: applied %s (allow=%zu "
                                     "pool=%zu)\n",
                                     config_path.c_str(), a.size(),
                                     p.size());
                    } catch (const std::exception& e) {
                        std::fprintf(stderr,
                                     "reload: %s; keeping current "
                                     "config\n",
                                     e.what());
                    } catch (...) {
                        std::fprintf(stderr,
                                     "reload: unknown error; keeping "
                                     "current config\n");
                    }
                };
        }

        const bool ok = locus::pipeline::serve_stage(
            stage, lfd, rt.allow, rt.pool, rt.conn, reload);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "locus-stage: %s\n", e.what());
        return 1;
    }
}
