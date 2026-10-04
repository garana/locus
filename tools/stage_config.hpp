#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
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
    int kv_blocks = 0;                    /**< --kv-blocks per worker
                                           *   (0 = model default) */
    int executors = 1;                    /**< --executors: CPU batching
                                           *   lanes (sessions pinned one
                                           *   per lane for KV locality) */
    int max_batch = 16;                   /**< --max-batch: sessions
                                           *   coalesced per executor
                                           *   forward */
    std::string backend;                  /**< --backend NAME: math
                                           *   backend for this stage
                                           *   (empty = auto default) */
    std::vector<std::string> device;      /**< --device N: CUDA device
                                           *   ordinal(s). One entry binds
                                           *   the whole process; >1 is
                                           *   per-executor binding (inc
                                           *   4b, not yet supported) */
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
        D::integer("kv-blocks", &O::kv_blocks,
                   "per-worker KV cache blocks (0 = model default)"),
        D::integer("executors", &O::executors,
                   "CPU batching executors; sessions pinned least-loaded "
                   "for KV locality (default 1)"),
        D::integer("max-batch", &O::max_batch,
                   "sessions coalesced into one batched forward per "
                   "executor (default 16)"),
        D::string_("backend", &O::backend,
                   "math backend for this stage (see locus-run "
                   "--backends); empty = auto-select"),
        D::string_list("device", &O::device,
                       "CUDA device ordinal; requires --backend cuda. "
                       "One entry binds the whole process to that device. "
                       "Under a GPU backend, executors are per-device "
                       "lanes, not per-core -- use one executor per "
                       "device unless measured otherwise"),
    });
}

/**
 * Resolved device binding for a stage (i#24 inc 4). `device` < 0 means
 * no explicit device was requested, so the stage runs on the backend's
 * default device.
 */
struct DeviceBinding {
    std::string backend;  /**< backend name to resolve ("" = auto) */
    int device = -1;      /**< CUDA device ordinal, or -1 for none */
};

/**
 * Validates --backend/--device and resolves them to a DeviceBinding.
 * In inc 4a a single --device binds the WHOLE process to that CUDA
 * ordinal. Multiple --device entries (one per executor) are inc 4b and
 * rejected here, but the eventual rule (count is 1 or == --executors) is
 * enforced NOW so 4b fills the flag surface in rather than changing its
 * arity. A --device with a non-cuda backend, a non-integer ordinal, or a
 * negative ordinal is an error.
 *
 * `out` is overwritten at entry (backend copied from opt, device reset
 * to -1), so on an error return it holds opt.backend and device -1, not
 * the caller's prior values.
 * @param opt parsed options.
 * @param n_exec the resolved executor count (--executors, >= 1).
 * @param out device/backend binding; device >= 0 only on success.
 * @returns "" on success, else a human-readable error.
 */
inline std::string resolve_device_binding(const StageOptions& opt,
                                          int n_exec, DeviceBinding& out) {
    out.backend = opt.backend;
    out.device = -1;
    if (opt.device.empty()) {
        return "";  // no explicit device: the backend's default
    }
    // --device only means anything for CUDA. Check this first so the
    // diagnostic names the real problem regardless of how many entries
    // were given (not a confusing arity/4b message for a non-cuda
    // backend).
    if (opt.backend != "cuda") {
        return "--device requires --backend cuda";
    }
    if (opt.device.size() > 1) {
        if (static_cast<int>(opt.device.size()) != n_exec) {
            return "multiple --device entries must match --executors (" +
                   std::to_string(n_exec) + "), got " +
                   std::to_string(opt.device.size());
        }
        return "per-executor device binding (multiple --device) is inc "
               "4b and not yet supported; pass a single --device";
    }
    // One guarded non-negative-int parse (parse_nonneg_int, cli_spec.hpp)
    // shared with Spec kInt and parse_layers -- it rejects junk, a
    // negative ordinal, and an out-of-int value (strtol returns a long,
    // so on LP64 a value above INT_MAX would pass a < 0 test and then
    // truncate on the cast, which here would skip cuda_set_device yet
    // still bind the backend: a silent wrong device).
    const std::string& s = opt.device.front();
    int dev = 0;
    if (!parse_nonneg_int(s, dev)) {
        return "bad --device (want a non-negative integer): " + s;
    }
    out.device = dev;
    return "";
}

/**
 * Guards a planned KV footprint against available RAM. `per_worker_bytes`
 * is one worker's committed KV pool (LlamaModel::kv_pool_bytes); the
 * model's weights are mmap'd (page cache, kernel-evictable) so they are
 * not counted here. required = per_worker_bytes * max(workers, 1).
 *
 * @param avail_bytes total physical RAM (sys::total_ram_bytes()); 0 means
 *     unknown, in which case this returns "" -- a guard must not refuse
 *     to start just because it could not measure RAM.
 * @param headroom fraction of RAM the KV pools may use, leaving the rest
 *     for the resident weight pages, per-worker workspaces and the OS.
 * @returns an error string if the pools would exceed the budget, else "".
 *
 * This guards against an out-of-memory kill, NOT against page-cache
 * starvation: every KV byte committed is a page the mmap'd weight window
 * cannot keep resident, so a config that passes (say 79% of RAM) can
 * still thrash on a streaming model whose throughput depends on the hot
 * weights staying cached. "Fits" is not "performs" -- passing this check
 * means the stage will start, not that it will be fast.
 */
inline std::string check_kv_memory(std::size_t per_worker_bytes,
                                   int workers, std::uint64_t avail_bytes,
                                   double headroom = 0.8) {
    if (avail_bytes == 0) {
        return "";  // RAM unknown: do not block startup
    }
    const std::uint64_t n = workers < 1 ? 1 : static_cast<std::uint64_t>(
                                                  workers);
    const std::uint64_t required =
        static_cast<std::uint64_t>(per_worker_bytes) * n;
    const std::uint64_t budget =
        static_cast<std::uint64_t>(static_cast<double>(avail_bytes) *
                                   headroom);
    if (required <= budget) {
        return "";
    }
    const auto mib = [](std::uint64_t b) {
        return b / (1024 * 1024);
    };
    return "KV cache needs ~" + std::to_string(mib(required)) + " MiB (" +
           std::to_string(n) + " worker(s) x ~" +
           std::to_string(mib(per_worker_bytes)) +
           " MiB), over the ~" + std::to_string(mib(budget)) +
           " MiB budget (" + std::to_string(mib(avail_bytes)) +
           " MiB RAM); reduce --kv-blocks or --workers";
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

/** Parses "A:B" into a half-open layer range, requiring A < B. Both
 * halves go through parse_nonneg_int (cli_spec.hpp), the shared guarded
 * parse, so an out-of-range bound is REJECTED rather than truncating
 * long -> uint32_t and silently serving a different slice (e.g.
 * "4294967296:..." narrowing to 0). */
inline bool parse_layers(const std::string& s, std::uint32_t& a,
                         std::uint32_t& b) {
    const auto c = s.find(':');
    if (c == std::string::npos) {
        return false;
    }
    int la = 0;
    int lb = 0;
    if (!parse_nonneg_int(s.substr(0, c), la) ||
        !parse_nonneg_int(s.substr(c + 1), lb) || la >= lb) {
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
