#include <cstdint>

#include "catch_amalgamated.hpp"
#include "locus/sys/meminfo.hpp"

// On the hosts this runs on (Linux, macOS) total RAM is knowable and
// larger than any sane lower bound. (On an unsupported platform the
// function returns 0 by contract, which this would flag -- intentionally,
// since the memory guard silently no-ops there and that is worth knowing.)
TEST_CASE("total_ram_bytes reports a plausible amount", "[meminfo]") {
    const std::uint64_t ram = locus::sys::total_ram_bytes();
    REQUIRE(ram > 64ull * 1024 * 1024);  // > 64 MiB: not zero/tiny
    REQUIRE(ram < 16ull * 1024 * 1024 * 1024 * 1024);  // < 16 TiB: sane
}
