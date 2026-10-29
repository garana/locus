#include <cstdint>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/engine/metrics.hpp"
#include "locus/server/metrics_text.hpp"

using locus::engine::Histogram;

TEST_CASE("histogram buckets are cumulative with le bounds",
          "[metrics]") {
    const double bounds[] = {0.001, 0.01, 0.1};
    Histogram h(bounds);
    h.observe(500'000);     // 0.5 ms: first bucket
    h.observe(1'000'000);   // exactly 1 ms: le="0.001" includes it
    h.observe(5'000'000);   // 5 ms
    h.observe(2'000'000'000);  // 2 s: only +Inf
    const auto s = h.snapshot();
    REQUIRE(s.bounds == std::vector<double>{0.001, 0.01, 0.1});
    REQUIRE(s.cumulative == std::vector<std::uint64_t>{2, 3, 3, 4});
    REQUIRE(s.count == 4);
    REQUIRE(s.sum == Catch::Approx(2.0065));
}

TEST_CASE("histogram rejects too many bounds", "[metrics]") {
    const std::vector<double> many(Histogram::kMaxBounds + 1, 1.0);
    REQUIRE_THROWS_AS(Histogram(many), std::invalid_argument);
}

TEST_CASE("metrics text renders labels and histogram series",
          "[metrics]") {
    using locus::server::MetricsText;
    const double bounds[] = {0.5};
    Histogram h(bounds);
    h.observe(250'000'000);
    MetricsText mt({{"model", "a\"b\\c"}, {"backend", "neon"}});
    mt.counter("x_total", "X.", std::uint64_t{3});
    mt.counter("y_seconds_total", "Y.", 1.5);
    mt.histogram("z_seconds", "Z.", h.snapshot());
    mt.info("i", "I.", {{"arch", "llama"}});
    const std::string lb = R"(model="a\"b\\c",backend="neon")";
    REQUIRE(mt.text() ==
            "# HELP x_total X.\n# TYPE x_total counter\n"
            "x_total{" + lb + "} 3\n"
            "# HELP y_seconds_total Y.\n# TYPE y_seconds_total counter\n"
            "y_seconds_total{" + lb + "} 1.5\n"
            "# HELP z_seconds Z.\n# TYPE z_seconds histogram\n"
            "z_seconds_bucket{" + lb + ",le=\"0.5\"} 1\n"
            "z_seconds_bucket{" + lb + ",le=\"+Inf\"} 1\n"
            "z_seconds_sum{" + lb + "} 0.25\n"
            "z_seconds_count{" + lb + "} 1\n"
            "# HELP i I.\n# TYPE i gauge\n"
            "i{" + lb + ",arch=\"llama\"} 1\n");
    REQUIRE(MetricsText::escape("a\nb") == "a\\nb");
}

// i#285: one family header, then labelled series after the fixed labels.
TEST_CASE("metrics text renders labelled families", "[metrics]") {
    using locus::server::MetricsText;
    const double bounds[] = {1024};
    Histogram bytes(bounds, 1.0);  // observations in bytes, rendered as is
    bytes.observe(100);
    bytes.observe(5000);
    // One label: spelled as a vector, since GCC finds {{"k", "v"}}
    // ambiguous with MetricsText's copy and move constructors.
    MetricsText mt(std::vector<MetricsText::Label>{{"model", "m"}});
    mt.family("r_total", "R.", "counter");
    mt.value("r_total", MetricsText::label("route", "/a\"b"),
             std::uint64_t{2});
    mt.family("b_bytes", "B.", "histogram");
    mt.histogram_series("b_bytes", MetricsText::label("route", "/x"),
                        bytes.snapshot());
    REQUIRE(mt.text() ==
            "# HELP r_total R.\n# TYPE r_total counter\n"
            "r_total{model=\"m\",route=\"/a\\\"b\"} 2\n"
            "# HELP b_bytes B.\n# TYPE b_bytes histogram\n"
            "b_bytes_bucket{model=\"m\",route=\"/x\",le=\"1024\"} 1\n"
            "b_bytes_bucket{model=\"m\",route=\"/x\",le=\"+Inf\"} 2\n"
            "b_bytes_sum{model=\"m\",route=\"/x\"} 5100\n"
            "b_bytes_count{model=\"m\",route=\"/x\"} 2\n");
}
