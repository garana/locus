#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/model/expert_metrics.hpp"
#include "locus/sys/meminfo.hpp"
#include "locus/sys/mmap_file.hpp"

using locus::model::ExpertCalls;
using locus::model::expert_api;

namespace {

/** @returns The calls of one (layer, expert, api), 0 when absent. */
std::uint64_t calls_of(const std::vector<ExpertCalls::Series>& s,
                       std::uint32_t layer, std::uint32_t expert,
                       std::string_view api) {
    for (const auto& c : s) {
        if (c.layer == layer && c.expert == expert && c.api == api) {
            return c.calls;
        }
    }
    return 0;
}

}  // namespace

TEST_CASE("expert calls count per layer, expert and api", "[metrics]") {
    ExpertCalls::reset();
    REQUIRE(ExpertCalls::snapshot().empty());

    ExpertCalls::record(1, 3, 4, 8, true, "cuda");
    ExpertCalls::record(1, 3, 4, 8, true, "cuda");
    ExpertCalls::record(2, 7, 4, 8, false, "sse4");
    // Past the table's size: dropped, not a crash.
    ExpertCalls::record(4, 0, 4, 8, true, "cuda");
    ExpertCalls::record(0, 8, 4, 8, true, "cuda");

    const auto s = ExpertCalls::snapshot();
    REQUIRE(s.size() == 2);
    REQUIRE(calls_of(s, 1, 3, "cuda") == 2);
    REQUIRE(calls_of(s, 2, 7, "sse4") == 1);
    for (const auto& c : s) {
        REQUIRE(std::string(c.device) == (c.api == "cuda" ? "gpu" : "cpu"));
    }

    ExpertCalls::reset();
    REQUIRE(ExpertCalls::snapshot().empty());
}

// i#1219: a run adds to its cell's runs and time and to the tokens
// histogram of this thread's phase; a batched run over several tokens
// is one run; Vulkan and Metal runs are not timed; disk bytes sum over
// the apis that picked the expert.
TEST_CASE("expert runs, time, tokens and disk bytes", "[metrics]") {
    using locus::model::RunPhase;
    using locus::model::RunPhaseScope;
    ExpertCalls::reset();
    REQUIRE(locus::model::run_phase() == RunPhase::kPrefill);
    ExpertCalls::record_run(1, 3, 4, 8, false, "neon", 5, 1000);
    {
        const RunPhaseScope ph(RunPhase::kDecode);
        REQUIRE(locus::model::run_phase() == RunPhase::kDecode);
        ExpertCalls::record_run(1, 3, 4, 8, false, "neon", 1, 300);
        ExpertCalls::record_run(1, 3, 4, 8, true, "vulkan", 1, 300);
        ExpertCalls::record_run(1, 3, 4, 8, false, "metal", 1, 300);
    }
    REQUIRE(locus::model::run_phase() == RunPhase::kPrefill);
    const auto runs = ExpertCalls::run_snapshot();
    REQUIRE(runs.size() == 1);
    REQUIRE(runs[0].layer == 1);
    REQUIRE(runs[0].expert == 3);
    REQUIRE(runs[0].api == "neon");
    REQUIRE(runs[0].runs == 2);
    REQUIRE(runs[0].ns == 1300);
    const auto rt = ExpertCalls::run_tokens();
    REQUIRE(rt.size() == 2);
    for (const auto& t : rt) {
        REQUIRE(t.api == "neon");
        REQUIRE(t.tokens.count == 1);
        // Bounds 1, 2, 4, 8: 5 tokens are in le="8" (index 3), not 4.
        if (t.phase == RunPhase::kPrefill) {
            REQUIRE(t.tokens.sum == 5);
            REQUIRE(t.tokens.cumulative[2] == 0);
            REQUIRE(t.tokens.cumulative[3] == 1);
        } else {
            REQUIRE(t.phase == RunPhase::kDecode);
            REQUIRE(t.tokens.cumulative[0] == 1);
        }
    }
    ExpertCalls::record_disk(2, 7, 4, 8, false, "neon", 4096);
    ExpertCalls::record_disk(2, 7, 4, 8, true, "cuda", 8192);
    ExpertCalls::record_disk(2, 6, 4, 8, true, "cuda", 0);  // no series
    const auto disk = ExpertCalls::disk_snapshot();
    REQUIRE(disk.size() == 1);
    REQUIRE(disk[0].layer == 2);
    REQUIRE(disk[0].expert == 7);
    REQUIRE(disk[0].bytes == 12288);
    ExpertCalls::reset();
    REQUIRE(ExpertCalls::run_snapshot().empty());
}

// i#1219: mincore over a range: a fresh anonymous mapping is not in
// RAM until touched; touching half its pages leaves the other half.
TEST_CASE("nonresident bytes follow the pages touched", "[metrics]") {
    const std::size_t page =
        static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t n = 8 * page;
    void* p = ::mmap(nullptr, n, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(p != MAP_FAILED);
    auto* b = static_cast<unsigned char*>(p);
    REQUIRE(locus::sys::nonresident_bytes(b, n) == n);
    for (std::size_t i = 0; i < 4; ++i) b[i * page] = 1;
    REQUIRE(locus::sys::nonresident_bytes(b, n) == 4 * page);
    // A range inside one touched page counts that page only.
    REQUIRE(locus::sys::nonresident_bytes(b + 10, 100) == 0);
    REQUIRE(locus::sys::nonresident_bytes(b + 5 * page + 10, 100) == page);
    ::munmap(p, n);
    REQUIRE(locus::sys::nonresident_bytes(nullptr, 10) == 0);
}

// i#1219: the process's storage reads: a number that only grows.
TEST_CASE("storage read bytes are reported and do not go back",
          "[metrics]") {
    const auto a = locus::sys::storage_read_bytes();
    const auto b = locus::sys::storage_read_bytes();
    REQUIRE(b >= a);
}

TEST_CASE("expert api label follows the backend", "[metrics]") {
    bool gpu = true;
    REQUIRE(expert_api("sse4", gpu) == "sse4");
    REQUIRE_FALSE(gpu);
    REQUIRE(expert_api("cuda", gpu) == "cuda");
    REQUIRE(gpu);
    gpu = false;
    const std::string_view vk = expert_api("vulkan", gpu);
    REQUIRE(gpu);
    REQUIRE((vk == "vulkan" || vk == "moltenvk"));
}
