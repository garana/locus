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
