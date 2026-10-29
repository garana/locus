#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include "catch_amalgamated.hpp"
#include "locus/model/stage_metrics.hpp"
#include "locus/sys/features.hpp"

using locus::model::Stage;
using locus::model::StageLap;
using locus::model::StageMetrics;

// i#285: per (layer, stage) sums, the no-layer row, and the stage total.
TEST_CASE("stage metrics sum per layer and stage", "[metrics]") {
    StageMetrics m(3);
    REQUIRE(m.layers() == 3);
    m.add(0, Stage::kAttention, 10);
    m.add(2, Stage::kAttention, 5);
    m.add(2, Stage::kAttention, 1);
    m.add(1, Stage::kFfn, 7);
    m.add(StageMetrics::kNoLayer, Stage::kHead, 4);
    m.add(99, Stage::kNorm, 2);  // past the count: the no-layer row
    REQUIRE(m.ns(0, Stage::kAttention) == 10);
    REQUIRE(m.ns(2, Stage::kAttention) == 6);
    REQUIRE(m.ns(1, Stage::kAttention) == 0);
    REQUIRE(m.stage_ns(Stage::kAttention) == 16);
    REQUIRE(m.stage_ns(Stage::kFfn) == 7);
    REQUIRE(m.ns(StageMetrics::kNoLayer, Stage::kHead) == 4);
    REQUIRE(m.ns(StageMetrics::kNoLayer, Stage::kNorm) == 2);
    REQUIRE(m.stage_ns(Stage::kNorm) == 2);
    REQUIRE(std::string(locus::model::stage_name(Stage::kMoe)) == "moe");
    REQUIRE(std::string(locus::model::stage_name(Stage::kSample)) ==
            "sample");
}

// Each lap charges the time since the previous one; restart() drops the
// time so far; null metrics do nothing.
TEST_CASE("stage lap charges the time between laps", "[metrics]") {
    using namespace std::chrono_literals;
    StageMetrics m(1);
    StageLap lap(&m);
    std::this_thread::sleep_for(20ms);
    lap.lap(0, Stage::kAttention);
    lap.lap(0, Stage::kFfn);  // right after: almost nothing
    std::this_thread::sleep_for(20ms);
    lap.restart();
    lap.lap(0, Stage::kNorm);
    REQUIRE(m.ns(0, Stage::kAttention) >= 20'000'000);
    REQUIRE(m.ns(0, Stage::kFfn) < 5'000'000);
    REQUIRE(m.ns(0, Stage::kNorm) < 5'000'000);
    StageLap none(nullptr);
    none.lap(0, Stage::kHead);
    none.restart();
    REQUIRE(m.stage_ns(Stage::kHead) == 0);
}

// The hardware label: a name on every box the tests run on (macOS's
// brand string, x86 Linux's model name, an ARM board's device-tree
// model), with no surrounding spaces.
TEST_CASE("cpu name is found and trimmed", "[metrics]") {
    const std::string n = locus::sys::cpu_name();
    INFO(n);
    REQUIRE(!n.empty());
    REQUIRE(n.front() != ' ');
    REQUIRE(n.back() != ' ');
    REQUIRE(n.find('\0') == std::string::npos);
}
