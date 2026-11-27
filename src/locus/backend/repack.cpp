#include "locus/backend/repack.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <shared_mutex>

namespace locus::backend {

namespace {

constexpr std::size_t kBlk = 34;  // f16 scale + 32 int8

/** One registered matrix: its mapped rows and their packed copy. */
struct Entry {
    const std::byte* src;
    std::size_t src_bytes;
    std::size_t row_bytes;
    const std::byte* packed;
};

/** The registered matrices, sorted by src. */
struct Table {
    std::shared_mutex mu;
    std::vector<Entry> entries;
};

Table& table() {
    static Table t;
    return t;
}

}  // namespace

bool packs_q8_0(const Mat& w) {
    return w.type == gguf::TensorType::kQ8_0 && w.data != nullptr &&
           w.rows > 0 && w.cols > 0 && w.cols % 32 == 0;
}

void pack_q8_0_quad(const Mat& w, std::uint32_t r0, std::byte* out) {
    const std::size_t nb = w.cols / 32;
    const std::byte* rows[4];
    for (std::uint32_t j = 0; j < 4; ++j) {
        const std::uint32_t r = std::min(r0 + j, w.rows - 1);
        rows[j] = w.data + std::size_t{r} * nb * kBlk;
    }
    for (std::size_t b = 0; b < nb; ++b) {
        std::byte* o = out + b * kPackedQ8_0Block;
        for (std::size_t j = 0; j < 4; ++j) {
            std::memcpy(o + 2 * j, rows[j] + b * kBlk, 2);
        }
        for (std::size_t k = 0; k < 8; ++k) {
            for (std::size_t j = 0; j < 4; ++j) {
                std::memcpy(o + 8 + 16 * k + 4 * j,
                            rows[j] + b * kBlk + 2 + 4 * k, 4);
            }
        }
    }
}

PackedQ8_0::PackedQ8_0(const Mat& w)
    : src_(w.data), src_bytes_(std::size_t{w.rows} * mat_row_bytes(w)) {
    const std::size_t row_bytes = mat_row_bytes(w);
    const std::uint32_t quads = (w.rows + 3) / 4;
    bytes_.resize(std::size_t{quads} * 4 * row_bytes);
    for (std::uint32_t q = 0; q < quads; ++q) {
        pack_q8_0_quad(w, 4 * q,
                       bytes_.data() + std::size_t{q} * 4 * row_bytes);
    }
    Table& t = table();
    std::unique_lock lk(t.mu);
    const Entry e{src_, src_bytes_, row_bytes, bytes_.data()};
    t.entries.insert(
        std::upper_bound(t.entries.begin(), t.entries.end(), e,
                         [](const Entry& a, const Entry& b) {
                             return a.src < b.src;
                         }),
        e);
}

PackedQ8_0::~PackedQ8_0() {
    Table& t = table();
    std::unique_lock lk(t.mu);
    std::erase_if(t.entries, [&](const Entry& e) {
        return e.packed == bytes_.data();
    });
}

const std::byte* find_packed_q8_0(const Mat& w) {
    if (!packs_q8_0(w)) {
        return nullptr;
    }
    Table& t = table();
    std::shared_lock lk(t.mu);
    if (t.entries.empty()) {
        return nullptr;
    }
    // The last entry starting at or before w.data.
    auto it = std::upper_bound(
        t.entries.begin(), t.entries.end(), w.data,
        [](const std::byte* p, const Entry& e) { return p < e.src; });
    if (it == t.entries.begin()) {
        return nullptr;
    }
    const Entry& e = *--it;
    const auto off = static_cast<std::size_t>(w.data - e.src);
    const std::size_t row_bytes = mat_row_bytes(w);
    if (row_bytes != e.row_bytes || off % (4 * row_bytes) != 0 ||
        off + std::size_t{w.rows} * row_bytes > e.src_bytes) {
        return nullptr;
    }
    return e.packed + off;
}

}  // namespace locus::backend
