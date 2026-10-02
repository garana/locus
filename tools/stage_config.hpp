#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "cli_spec.hpp"
#include "locus/pipeline/net.hpp"
#include "locus/pipeline/resolver.hpp"
#include "locus/pipeline/stage_server.hpp"

namespace locus_tools {

/**
 * All of locus-stage's options. The CLI fills it first, then a config
 * file overlays it (CLI-then-file, the file wins; i#19, DESIGN.md
 * R15+). Field defaults match the CLI defaults. The field <-> flag/key
 * mapping lives in stage_spec(), the single source of truth for both
 * the CLI parser and the config-file loader.
 *
 * File-wins inverts the usual "CLI overrides the file" convention on
 * purpose: the config file is the live source of truth that SIGHUP
 * re-reads (i#19 part 2), and the CLI is only the bootstrap. If the CLI
 * won, a reload could never change anything also passed on the command
 * line. Do not "fix" this to the conventional order -- it would
 * silently break reload.
 */
struct StageOptions {
    std::string model;                    /**< --model (required) */
    std::string layers;                   /**< --layers A:B (required) */
    std::string listen;                   /**< --listen HOST:PORT (req) */
    std::vector<std::string> downstream;  /**< --downstream pool (req) */
    std::vector<std::string> allow;       /**< --allow CIDR (repeatable)*/
    int connect_timeout = 5000;           /**< --connect-timeout ms */
    int reconnect_wait = 1000;            /**< --reconnect-wait ms */
    int reconnect_attempts = 0;           /**< --reconnect-attempts */
    int read_timeout = 0;                 /**< --read-timeout ms */
    int keepalive_idle = 5;               /**< --keepalive-idle s */
    int keepalive_interval = 2;           /**< --keepalive-interval s */
    int keepalive_count = 3;              /**< --keepalive-count */
    int sessions = 0;                     /**< --sessions (0=forever) */
    int resolve_ttl = 30;                 /**< --resolve-ttl s (name
                                           *   cache; fallback TTL until
                                           *   record TTL is used) */
    int resolve_max_ttl = 300;            /**< --resolve-max-ttl s cap */
    int resolve_refresh_percent = 75;     /**< --resolve-refresh-percent
                                           *   of the TTL */
};

/**
 * The locus-stage directive table: every CLI flag and its identically
 * named config-file key, with arity/type and whether it is required,
 * bound to a StageOptions field. Format validation of the string
 * fields (A:B layers, HOST:PORT, CIDR) stays in the tool, after the
 * spec has filled them.
 */
inline Spec<StageOptions> stage_spec() {
    using D = Directive<StageOptions>;
    using O = StageOptions;
    return Spec<StageOptions>({
        D::string_("model", &O::model, "gguf model file", true),
        D::string_("layers", &O::layers,
                   "layer range A:B (half-open, A<B)", true),
        D::string_("listen", &O::listen,
                   "bind address HOST:PORT (empty HOST = wildcard)",
                   true),
        D::string_list("downstream", &O::downstream,
                       "downstream HOST:PORT; repeat for a replica pool",
                       true),
        D::string_list("allow", &O::allow,
                       "allowed peer CIDR; repeatable (default: any)"),
        D::integer("connect-timeout", &O::connect_timeout,
                   "downstream connect timeout ms (0 = OS default)"),
        D::integer("reconnect-wait", &O::reconnect_wait,
                   "fixed wait between connect attempts ms (no backoff)"),
        D::integer("reconnect-attempts", &O::reconnect_attempts,
                   "give up after N attempts (0 = retry forever)"),
        D::integer("read-timeout", &O::read_timeout,
                   "input read timeout ms (0 = block)"),
        D::integer("keepalive-idle", &O::keepalive_idle,
                   "idle s before first keepalive probe (0 = disable)"),
        D::integer("keepalive-interval", &O::keepalive_interval,
                   "seconds between keepalive probes"),
        D::integer("keepalive-count", &O::keepalive_count,
                   "unacked keepalive probes before drop"),
        D::integer("sessions", &O::sessions,
                   "serve N sessions then exit (0 = forever)"),
        D::integer("resolve-ttl", &O::resolve_ttl,
                   "downstream-name cache TTL s (fallback until the "
                   "record TTL is used)"),
        D::integer("resolve-max-ttl", &O::resolve_max_ttl,
                   "cap on the downstream-name cache TTL s"),
        D::integer("resolve-refresh-percent", &O::resolve_refresh_percent,
                   "refresh a cached name at this percent of its TTL"),
    });
}

/** Builds the resolver policy from the resolve-* options. These are
 * startup-fixed (the Resolver is long-lived): a SIGHUP reload does not
 * re-tune an already-running resolver. */
inline locus::pipeline::Resolver::Options resolver_options(
    const StageOptions& opt) {
    locus::pipeline::Resolver::Options ro;
    ro.default_ttl_s = opt.resolve_ttl;
    ro.max_ttl_s = opt.resolve_max_ttl;
    ro.refresh_frac =
        static_cast<double>(opt.resolve_refresh_percent) / 100.0;
    return ro;
}

/** Parses "A:B" into a half-open layer range, requiring A < B. */
inline bool parse_layers(const std::string& s, std::uint32_t& a,
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

/**
 * The runtime values a StageOptions resolves to: the layer range and
 * listen address (fixed for a running stage) and the allowlist, pool
 * and connection policy (the reloadable subset). build_runtime is the
 * one conversion+validation path, shared by startup and SIGHUP reload.
 */
struct StageRuntime {
    std::uint32_t layer_begin = 0;
    std::uint32_t layer_end = 0;
    std::string listen_host;
    int listen_port = 0;
    std::vector<locus::pipeline::Cidr> allow;
    std::vector<locus::pipeline::HostPort> pool;
    locus::pipeline::StageConn conn;
};

/**
 * Validates and converts `opt` into runtime values. @returns an empty
 * string on success (out filled), else a human-readable error; the
 * caller decides whether that is fatal (startup) or logged and skipped
 * (reload).
 */
inline std::string build_runtime(const StageOptions& opt,
                                 StageRuntime& out) {
    if (!parse_layers(opt.layers, out.layer_begin, out.layer_end)) {
        return "bad layers (want A:B with A < B): " + opt.layers;
    }
    if (!locus::pipeline::parse_hostport(opt.listen, out.listen_host,
                                         out.listen_port)) {
        return "bad listen (want HOST:PORT): " + opt.listen;
    }
    out.pool.clear();
    for (const auto& ds : opt.downstream) {
        std::string h;
        int p = 0;
        if (!locus::pipeline::parse_hostport(ds, h, p)) {
            return "bad downstream (want HOST:PORT): " + ds;
        }
        out.pool.push_back({h, p});
    }
    out.allow.clear();
    for (const auto& s : opt.allow) {
        const auto c = locus::pipeline::Cidr::parse(s);
        if (!c) {
            return "bad allow CIDR: " + s;
        }
        out.allow.push_back(*c);
    }
    out.conn.connect_timeout_ms = opt.connect_timeout;
    out.conn.reconnect_wait_ms = opt.reconnect_wait;
    out.conn.reconnect_attempts = opt.reconnect_attempts;
    out.conn.recv_timeout_ms = opt.read_timeout;
    out.conn.keepalive_idle_s = opt.keepalive_idle;
    out.conn.keepalive_intvl_s = opt.keepalive_interval;
    out.conn.keepalive_count = opt.keepalive_count;
    out.conn.serve_sessions = opt.sessions;
    return "";
}

}  // namespace locus_tools
