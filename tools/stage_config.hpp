#pragma once

#include <string>
#include <vector>

#include "cli_spec.hpp"

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
    });
}

}  // namespace locus_tools
