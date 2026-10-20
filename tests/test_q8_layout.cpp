#include <cstring>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/vulkan/q8_layout.hpp"

TEST_CASE("q8_0_split_rows puts each row's scales before its quants",
          "[vulkan]") {
    // 3 rows x 64 cols: 2 blocks per row, 68 bytes per row. Byte
    // value encodes (row, block, offset in block).
    const std::uint32_t rows = 3, cols = 64, nb = 2;
    std::vector<std::byte> src(rows * nb * 34);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t b = 0; b < nb; ++b) {
            for (std::uint32_t i = 0; i < 34; ++i) {
                src[(r * nb + b) * 34 + i] =
                    static_cast<std::byte>(r * 100 + b * 40 + i);
            }
        }
    }
    std::vector<std::byte> dst(src.size());
    locus::backend::vk::q8_0_split_rows(src.data(), dst.data(), rows,
                                        cols);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::byte* row = dst.data() + r * nb * 34;
        for (std::uint32_t b = 0; b < nb; ++b) {
            // Scale bytes of block b at 2b; its 32 quants at
            // 2 nb + 32 b.
            REQUIRE(row[2 * b] == src[(r * nb + b) * 34]);
            REQUIRE(row[2 * b + 1] == src[(r * nb + b) * 34 + 1]);
            REQUIRE(std::memcmp(row + 2 * nb + 32 * b,
                                src.data() + (r * nb + b) * 34 + 2,
                                32) == 0);
        }
    }
}
