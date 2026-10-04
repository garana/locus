#include <exception>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "catch_amalgamated.hpp"
#include "cli_spec.hpp"
#include "locus/config/config_file.hpp"
#include "stage_config.hpp"

using locus::config::ConfigFile;
using locus_tools::StageOptions;

// ---- the key=value config parser ----

TEST_CASE("ConfigFile parses key=value with whitespace and comments",
          "[config]") {
    const auto cfg = ConfigFile::parse_string(
        "# a comment\n"
        "model = m.gguf\n"
        "  layers=0:4  \n"
        "\n"
        "   # indented comment\n"
        "listen = :8000\n");
    REQUIRE(cfg.get("model") == "m.gguf");
    REQUIRE(cfg.get("layers") == "0:4");    // trimmed both sides
    REQUIRE(cfg.get("listen") == ":8000");  // value keeps its colon
    REQUIRE_FALSE(cfg.has("absent"));
    REQUIRE(cfg.get("absent") == std::nullopt);
}

TEST_CASE("ConfigFile keeps repeated keys in order; get() is last-wins",
          "[config]") {
    const auto cfg = ConfigFile::parse_string(
        "downstream = a:1\n"
        "downstream = b:2\n"
        "downstream = c:3\n"
        "sessions = 1\n"
        "sessions = 2\n");
    REQUIRE(cfg.values("downstream") ==
            std::vector<std::string>{"a:1", "b:2", "c:3"});
    REQUIRE(cfg.get("sessions") == "2");  // later line overrides earlier
    REQUIRE(cfg.keys() ==
            std::vector<std::string>{"downstream", "sessions"});
}

TEST_CASE("ConfigFile rejects a malformed line", "[config]") {
    REQUIRE_THROWS_AS(ConfigFile::parse_string("model = m\nnonsense\n"),
                      ConfigFile::Error);
    REQUIRE_THROWS_AS(ConfigFile::parse_string("= novalue\n"),
                      ConfigFile::Error);
}

// Regression guard: Error MUST be catchable as std::exception. The
// SIGHUP reload path catches only std::exception; before this, Error
// was a bare struct, so a malformed config on reload escaped to
// std::terminate and killed the running stage.
TEST_CASE("ConfigFile::Error is a std::exception (reload must catch it)",
          "[config]") {
    static_assert(
        std::is_base_of_v<std::exception, ConfigFile::Error>,
        "ConfigFile::Error must derive from std::exception so the "
        "reload path's catch(const std::exception&) catches it");
    bool caught_as_std = false;
    try {
        ConfigFile::parse_string("nonsense\n");  // malformed line
    } catch (const std::exception& e) {
        caught_as_std = true;
        REQUIRE(std::string(e.what()).find("expected key = value") !=
                std::string::npos);
    }
    REQUIRE(caught_as_std);
}

TEST_CASE("ConfigFile::parse throws on a missing file", "[config]") {
    REQUIRE_THROWS_AS(ConfigFile::parse("/no/such/locus-config-xyz"),
                      ConfigFile::Error);
}

TEST_CASE("ConfigFile::parse throws on a non-regular path", "[config]") {
    // A directory opens but reads nothing; it must error, not look like
    // an empty config (which would read as a silent no-op on reload).
    REQUIRE_THROWS_AS(ConfigFile::parse("/"), ConfigFile::Error);
}

TEST_CASE("ConfigFile pins CRLF, '=' in value, and inline '#'",
          "[config]") {
    // A CRLF line: the trailing \r is trimmed, so the int still parses.
    REQUIRE(ConfigFile::parse_string("sessions = 7\r\n").get("sessions") ==
            "7");
    // Split on the FIRST '=' only; a '=' inside the value survives.
    REQUIRE(ConfigFile::parse_string("model = /tmp/a=b.gguf\n")
                .get("model") == "/tmp/a=b.gguf");
    // '#' is a comment only at line start; inline it stays in the value
    // (so no quoting rules are needed for values containing '#').
    REQUIRE(ConfigFile::parse_string("allow = 10.0.0.0/8 # office\n")
                .get("allow") == "10.0.0.0/8 # office");
}

// ---- the directive spec (one table drives CLI + config) ----

TEST_CASE("stage_spec exposes the required directives", "[config]") {
    const auto spec = locus_tools::stage_spec();
    REQUIRE(spec.find("model") != nullptr);
    REQUIRE(spec.find("model")->required());
    REQUIRE(spec.find("layers")->required());
    REQUIRE(spec.find("listen")->required());
    REQUIRE(spec.find("downstream")->required());
    REQUIRE_FALSE(spec.find("allow")->required());
    REQUIRE_FALSE(spec.find("connect-timeout")->required());
    REQUIRE(spec.find("nope") == nullptr);
    // --help text mentions a flag and its config key share a name.
    REQUIRE(spec.help().find("--connect-timeout") != std::string::npos);
}

TEST_CASE("spec.apply sets fields by type and validates ints",
          "[config]") {
    const auto spec = locus_tools::stage_spec();
    StageOptions opt;
    spec.apply(*spec.find("model"), opt, "m.gguf");
    spec.apply(*spec.find("connect-timeout"), opt, "250");
    spec.apply(*spec.find("downstream"), opt, "a:1");
    spec.apply(*spec.find("downstream"), opt, "b:2");  // kList appends
    REQUIRE(opt.model == "m.gguf");
    REQUIRE(opt.connect_timeout == 250);
    REQUIRE(opt.downstream == std::vector<std::string>{"a:1", "b:2"});
    REQUIRE_THROWS_AS(
        spec.apply(*spec.find("connect-timeout"), opt, "notanint"),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        spec.apply(*spec.find("sessions"), opt, "-5"),
        std::runtime_error);
}

TEST_CASE("stage_spec parses --executors and --max-batch (i#24 inc 3)",
          "[config]") {
    const auto spec = locus_tools::stage_spec();
    StageOptions opt;
    REQUIRE(opt.executors == 1);  // defaults
    REQUIRE(opt.max_batch == 16);
    REQUIRE(spec.find("executors") != nullptr);
    REQUIRE(spec.find("max-batch") != nullptr);
    spec.apply(*spec.find("executors"), opt, "4");
    spec.apply(*spec.find("max-batch"), opt, "8");
    REQUIRE(opt.executors == 4);
    REQUIRE(opt.max_batch == 8);
    REQUIRE_THROWS_AS(spec.apply(*spec.find("executors"), opt, "-1"),
                      std::runtime_error);
}

TEST_CASE("spec.apply_config: the file wins over CLI, and adds",
          "[config]") {
    const auto spec = locus_tools::stage_spec();
    StageOptions opt;  // stand in for what the CLI left
    opt.model = "cli.gguf";
    opt.layers = "0:2";
    opt.listen = ":9000";
    opt.downstream = {"cli-a:1"};
    opt.connect_timeout = 5000;
    std::set<std::string> seen{"model", "layers", "listen", "downstream"};

    const auto cfg = ConfigFile::parse_string(
        "model = file.gguf\n"      // overrides CLI
        "connect-timeout = 250\n"  // overrides CLI
        "keepalive-idle = 30\n"    // adds (CLI left the default)
        "downstream = f-a:1\n"     // repeatable: replaces CLI's list
        "downstream = f-b:2\n");
    spec.apply_config(cfg, opt, seen);

    REQUIRE(opt.model == "file.gguf");    // file won
    REQUIRE(opt.connect_timeout == 250);  // file won
    REQUIRE(opt.keepalive_idle == 30);    // file added
    REQUIRE(opt.layers == "0:2");         // untouched (not in file)
    REQUIRE(opt.listen == ":9000");       // untouched
    REQUIRE(opt.downstream ==
            std::vector<std::string>{"f-a:1", "f-b:2"});  // replaced
}

TEST_CASE("spec.apply_config: a file silent on a list keeps CLI's list",
          "[config]") {
    const auto spec = locus_tools::stage_spec();
    StageOptions opt;
    opt.downstream = {"cli-a:1", "cli-b:2"};
    std::set<std::string> seen{"downstream"};
    spec.apply_config(ConfigFile::parse_string("sessions = 3\n"), opt,
                      seen);
    REQUIRE(opt.downstream ==
            std::vector<std::string>{"cli-a:1", "cli-b:2"});
    REQUIRE(opt.sessions == 3);
}

TEST_CASE("spec.apply_config rejects unknown key / bad integer",
          "[config]") {
    const auto spec = locus_tools::stage_spec();
    StageOptions opt;
    std::set<std::string> seen;
    REQUIRE_THROWS_AS(
        spec.apply_config(ConfigFile::parse_string("bogus-key = x\n"),
                          opt, seen),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        spec.apply_config(
            ConfigFile::parse_string("read-timeout = notanint\n"), opt,
            seen),
        std::runtime_error);
}

TEST_CASE("spec.first_missing_required tracks what was seen",
          "[config]") {
    const auto spec = locus_tools::stage_spec();
    std::set<std::string> seen;
    // Nothing seen: a required directive is reported.
    const std::string miss = spec.first_missing_required(seen);
    REQUIRE_FALSE(miss.empty());
    REQUIRE((miss == "model" || miss == "layers" || miss == "listen" ||
             miss == "downstream"));
    // All required seen (allow/timeouts optional): nothing missing.
    seen = {"model", "layers", "listen", "downstream"};
    REQUIRE(spec.first_missing_required(seen).empty());
}

// ---- build_runtime: the shared startup/reload conversion ----

TEST_CASE("build_runtime validates and converts StageOptions",
          "[config]") {
    StageOptions opt;
    opt.layers = "2:5";
    opt.listen = "127.0.0.1:8000";
    opt.downstream = {"a:1", "[::1]:2"};  // bracketed IPv6 in the pool
    opt.allow = {"10.0.0.0/8"};
    opt.connect_timeout = 250;
    opt.keepalive_idle = 9;
    opt.sessions = 3;

    locus_tools::StageRuntime rt;
    REQUIRE(locus_tools::build_runtime(opt, rt).empty());
    REQUIRE(rt.layer_begin == 2);
    REQUIRE(rt.layer_end == 5);
    REQUIRE(rt.listen_host == "127.0.0.1");
    REQUIRE(rt.listen_port == 8000);
    REQUIRE(rt.pool.size() == 2);
    REQUIRE(rt.pool[0].host == "a");
    REQUIRE(rt.pool[0].port == 1);
    REQUIRE(rt.pool[1].host == "::1");  // brackets stripped
    REQUIRE(rt.pool[1].port == 2);
    REQUIRE(rt.allow.size() == 1);
    REQUIRE(rt.conn.connect_timeout_ms == 250);
    REQUIRE(rt.conn.keepalive_idle_s == 9);
    REQUIRE(rt.conn.serve_sessions == 3);

    // Bad formats return a message (not a throw); the caller decides.
    locus_tools::StageRuntime scratch;
    StageOptions bad = opt;
    bad.layers = "5:2";  // A >= B
    REQUIRE_FALSE(locus_tools::build_runtime(bad, scratch).empty());
    bad = opt;
    bad.listen = "noport";
    REQUIRE_FALSE(locus_tools::build_runtime(bad, scratch).empty());
    bad = opt;
    bad.allow = {"not-a-cidr"};
    REQUIRE_FALSE(locus_tools::build_runtime(bad, scratch).empty());
    bad = opt;
    bad.downstream = {"missing-port"};
    REQUIRE_FALSE(locus_tools::build_runtime(bad, scratch).empty());
}

// i#40: the resolve-* knobs map onto Resolver::Options, with the refresh
// percent converted to a fraction (the user-facing "75%" -> 0.75).
TEST_CASE("resolver_options maps the resolve knobs (percent -> fraction)",
          "[config]") {
    locus_tools::StageOptions opt;  // defaults
    REQUIRE(opt.resolve_ttl == 30);
    REQUIRE(opt.resolve_max_ttl == 300);
    REQUIRE(opt.resolve_refresh_percent == 75);

    auto ro = locus_tools::resolver_options(opt);
    REQUIRE(ro.default_ttl_s == 30);
    REQUIRE(ro.max_ttl_s == 300);
    REQUIRE(ro.refresh_frac > 0.749);
    REQUIRE(ro.refresh_frac < 0.751);  // 75% -> 0.75

    opt.resolve_refresh_percent = 50;
    const double f = locus_tools::resolver_options(opt).refresh_frac;
    REQUIRE(f > 0.499);
    REQUIRE(f < 0.501);  // 50% -> 0.50
}

// The resolve-* flags are in the directive table and parse as ints.
TEST_CASE("stage_spec parses the resolve-* knobs", "[config]") {
    const auto spec = locus_tools::stage_spec();
    locus_tools::StageOptions opt;
    REQUIRE(spec.find("resolve-ttl") != nullptr);
    REQUIRE(spec.find("resolve-max-ttl") != nullptr);
    REQUIRE(spec.find("resolve-refresh-percent") != nullptr);
    spec.apply(*spec.find("resolve-ttl"), opt, "12");
    spec.apply(*spec.find("resolve-max-ttl"), opt, "120");
    spec.apply(*spec.find("resolve-refresh-percent"), opt, "80");
    REQUIRE(opt.resolve_ttl == 12);
    REQUIRE(opt.resolve_max_ttl == 120);
    REQUIRE(opt.resolve_refresh_percent == 80);
}

// i#40/i#24: the memory guard refuses a KV footprint that would exceed
// available RAM (per-worker bytes x workers vs avail x headroom), and
// no-ops when RAM is unknown.
TEST_CASE("check_kv_memory guards KV against available RAM", "[config]") {
    using locus_tools::check_kv_memory;
    constexpr std::uint64_t MiB = 1024 * 1024;
    constexpr std::uint64_t GiB = 1024 * MiB;

    // 700 MiB x 1 under 0.8 x 1 GiB (~819 MiB): fits.
    REQUIRE(check_kv_memory(700 * MiB, 1, GiB).empty());
    // 900 MiB x 1: over the ~819 MiB budget.
    REQUIRE_FALSE(check_kv_memory(900 * MiB, 1, GiB).empty());
    // Worker count multiplies: 300 MiB x 2 fits, 500 MiB x 2 does not.
    REQUIRE(check_kv_memory(300 * MiB, 2, GiB).empty());
    REQUIRE_FALSE(check_kv_memory(500 * MiB, 2, GiB).empty());
    // Unknown RAM (0) never blocks, even for an absurd request.
    REQUIRE(check_kv_memory(100 * GiB, 8, 0).empty());
    // workers < 1 is treated as 1 (not 0 bytes).
    REQUIRE_FALSE(check_kv_memory(900 * MiB, 0, GiB).empty());
    // The error message names the knobs to turn.
    const std::string err = check_kv_memory(900 * MiB, 1, GiB);
    REQUIRE(err.find("--kv-blocks") != std::string::npos);
    REQUIRE(err.find("--workers") != std::string::npos);
}

// ---- device-select seam (i#24 inc 4) ----

TEST_CASE("resolve_device_binding: no device is the backend default",
          "[config][device]") {
    using locus_tools::DeviceBinding;
    using locus_tools::resolve_device_binding;
    StageOptions opt;
    DeviceBinding db;
    // No --device, no --backend: nothing bound, default device.
    REQUIRE(resolve_device_binding(opt, 1, db).empty());
    REQUIRE(db.device == -1);
    REQUIRE(db.backend.empty());
    // --backend without --device: backend carried, still default device.
    opt.backend = "cuda";
    REQUIRE(resolve_device_binding(opt, 1, db).empty());
    REQUIRE(db.device == -1);
    REQUIRE(db.backend == "cuda");
}

TEST_CASE("resolve_device_binding: a single CUDA device binds the process",
          "[config][device]") {
    using locus_tools::DeviceBinding;
    using locus_tools::resolve_device_binding;
    StageOptions opt;
    opt.backend = "cuda";
    opt.device = {"2"};
    DeviceBinding db;
    REQUIRE(resolve_device_binding(opt, 4, db).empty());
    REQUIRE(db.device == 2);
    REQUIRE(db.backend == "cuda");
}

TEST_CASE("resolve_device_binding: --device requires --backend cuda",
          "[config][device]") {
    using locus_tools::DeviceBinding;
    using locus_tools::resolve_device_binding;
    DeviceBinding db;
    // --device with no backend.
    StageOptions a;
    a.device = {"0"};
    REQUIRE(resolve_device_binding(a, 1, db).find("--backend cuda") !=
            std::string::npos);
    // --device with a non-cuda backend.
    StageOptions b;
    b.backend = "sse4";
    b.device = {"0"};
    REQUIRE(resolve_device_binding(b, 1, db).find("--backend cuda") !=
            std::string::npos);
}

TEST_CASE("resolve_device_binding: a non-integer / negative device fails",
          "[config][device]") {
    using locus_tools::DeviceBinding;
    using locus_tools::resolve_device_binding;
    DeviceBinding db;
    // Includes out-of-int values: strtol parses them into a positive
    // long that would truncate to a negative/wrong int on the cast, so
    // they must be rejected, not silently bound (i#24 fail-loud).
    for (const char* bad : {"x", "1.5", "", "-1", "2147483648",
                            "4294967298", "99999999999999999999"}) {
        StageOptions opt;
        opt.backend = "cuda";
        opt.device = {bad};
        REQUIRE(resolve_device_binding(opt, 1, db).find("bad --device") !=
                std::string::npos);
    }
}

TEST_CASE("resolve_device_binding: multiple --device is inc 4b, gated now",
          "[config][device]") {
    using locus_tools::DeviceBinding;
    using locus_tools::resolve_device_binding;
    DeviceBinding db;
    StageOptions opt;
    opt.backend = "cuda";
    // Count must be 1 or == --executors: 3 devices vs 4 executors errors
    // on the arity, naming --executors.
    opt.device = {"0", "1", "2"};
    REQUIRE(resolve_device_binding(opt, 4, db).find("--executors") !=
            std::string::npos);
    // Count == --executors (the eventual 4b per-executor shape) is
    // accepted by the arity rule but rejected as not-yet-implemented, so
    // the flag surface is fixed before 4b fills it in.
    REQUIRE(resolve_device_binding(opt, 3, db).find("4b") !=
            std::string::npos);
}
