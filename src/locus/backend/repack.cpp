#include "locus/backend/repack.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace locus::backend {

namespace {

constexpr std::size_t kBlkQ8 = 34;  // f16 scale + 32 int8
constexpr std::size_t kBlkQ4 = 18;  // f16 scale + 16 bytes of nibbles

/** One registered matrix: its mapped rows and their packed copy. */
struct Entry {
    const std::byte* src;
    std::size_t src_bytes;
    std::size_t row_bytes;
    gguf::TensorType type;
    const std::byte* packed;
};

/** An immutable list of the registered matrices, sorted by src. */
using Snapshot = std::vector<Entry>;

/**
 * The registry: find_packed runs on every matvec call of every thread,
 * so it takes no lock (a shared_mutex there cost the M2 Pro 9% of its
 * neon decode, i#1243): it loads `current` and searches it. A register
 * or unregister of a set (a model's load, its use_backend or its
 * teardown) builds one new snapshot under `writer` and swaps it in;
 * the old ones stay alive to the end, so a search never reads freed
 * memory: 40 bytes an entry, about 4.5 KB a snapshot for Llama 3.2 1B's
 * 113 matrices, one snapshot each load and teardown. Never destroyed,
 * so no exit-order race.
 */
struct Registry {
    std::atomic<const Snapshot*> current{nullptr};
    std::mutex writer;
    std::vector<std::unique_ptr<Snapshot>> all;

    /** Swaps in edit(copy of the current list); callers hold writer. */
    template <typename Edit>
    void publish(Edit&& edit) {
        const Snapshot* cur = current.load(std::memory_order_relaxed);
        auto next = std::make_unique<Snapshot>(cur ? *cur : Snapshot{});
        edit(*next);
        current.store(next->empty() ? nullptr : next.get(),
                      std::memory_order_release);
        all.push_back(std::move(next));
    }
};

Registry& registry() {
    static Registry* r = new Registry;
    return *r;
}

/** The four rows r0 .. r0 + 3 (a row past the last repeats the last). */
void quad_rows(const Mat& w, std::uint32_t r0, std::size_t row_bytes,
               const std::byte* rows[4]) {
    for (std::uint32_t j = 0; j < 4; ++j) {
        const std::uint32_t r = std::min(r0 + j, w.rows - 1);
        rows[j] = w.data + std::size_t{r} * row_bytes;
    }
}

}  // namespace

bool packs_block32(const Mat& w) {
    return (w.type == gguf::TensorType::kQ8_0 ||
            w.type == gguf::TensorType::kQ4_0) &&
           w.data != nullptr && w.rows > 0 && w.cols > 0 &&
           w.cols % 32 == 0;
}

void pack_q8_0_quad(const Mat& w, std::uint32_t r0, std::byte* out) {
    const std::size_t nb = w.cols / 32;
    const std::byte* rows[4];
    quad_rows(w, r0, nb * kBlkQ8, rows);
    for (std::size_t b = 0; b < nb; ++b) {
        std::byte* o = out + b * kPackedQ8_0Block;
        for (std::size_t j = 0; j < 4; ++j) {
            std::memcpy(o + 2 * j, rows[j] + b * kBlkQ8, 2);
        }
        for (std::size_t k = 0; k < 8; ++k) {
            for (std::size_t j = 0; j < 4; ++j) {
                std::memcpy(o + 8 + 16 * k + 4 * j,
                            rows[j] + b * kBlkQ8 + 2 + 4 * k, 4);
            }
        }
    }
}

void pack_q4_0_quad(const Mat& w, std::uint32_t r0, std::byte* out) {
    const std::size_t nb = w.cols / 32;
    const std::byte* rows[4];
    quad_rows(w, r0, nb * kBlkQ4, rows);
    for (std::size_t b = 0; b < nb; ++b) {
        std::byte* o = out + b * kPackedQ4_0Block;
        for (std::size_t j = 0; j < 4; ++j) {
            std::memcpy(o + 2 * j, rows[j] + b * kBlkQ4, 2);
        }
        for (std::size_t k = 0; k < 4; ++k) {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t i = 0; i < 4; ++i) {
                    o[8 + 16 * k + 4 * j + i] =
                        rows[j][b * kBlkQ4 + 2 + 4 * k + i] ^
                        std::byte{0x88};
                }
            }
        }
    }
}

PackedBlock32::PackedBlock32(const Mat& w)
    : type_(w.type),
      src_(w.data),
      src_bytes_(std::size_t{w.rows} * mat_row_bytes(w)),
      row_bytes_(mat_row_bytes(w)) {
    const std::uint32_t quads = (w.rows + 3) / 4;
    bytes_.resize(std::size_t{quads} * 4 * row_bytes_);
    const bool q8 = w.type == gguf::TensorType::kQ8_0;
    for (std::uint32_t q = 0; q < quads; ++q) {
        std::byte* out = bytes_.data() + std::size_t{q} * 4 * row_bytes_;
        if (q8) {
            pack_q8_0_quad(w, 4 * q, out);
        } else {
            pack_q4_0_quad(w, 4 * q, out);
        }
    }
}

PackedBlock32::~PackedBlock32() {
    if (registered_) {
        PackedBlock32* self = this;
        unregister_packed({&self, 1});
    }
}

void register_packed(std::span<PackedBlock32* const> set) {
    Registry& r = registry();
    std::lock_guard lk(r.writer);
    r.publish([&](Snapshot& s) {
        for (PackedBlock32* p : set) {
            if (p->registered_) {
                continue;
            }
            p->registered_ = true;
            s.push_back({p->src_, p->src_bytes_, p->row_bytes_, p->type_,
                         p->bytes_.data()});
        }
        std::sort(s.begin(), s.end(), [](const Entry& a, const Entry& b) {
            return a.src < b.src;
        });
    });
}

void unregister_packed(std::span<PackedBlock32* const> set) {
    Registry& r = registry();
    std::lock_guard lk(r.writer);
    r.publish([&](Snapshot& s) {
        for (PackedBlock32* p : set) {
            if (!p->registered_) {
                continue;
            }
            p->registered_ = false;
            std::erase_if(s, [&](const Entry& e) {
                return e.packed == p->bytes_.data();
            });
        }
    });
}

const std::byte* find_packed(const Mat& w) {
    if (!packs_block32(w)) {
        return nullptr;
    }
    const Snapshot* s =
        registry().current.load(std::memory_order_acquire);
    if (s == nullptr) {
        return nullptr;  // no copy registered: one load, no search
    }
    // The last entry starting at or before w.data.
    auto it = std::upper_bound(
        s->begin(), s->end(), w.data,
        [](const std::byte* p, const Entry& e) { return p < e.src; });
    if (it == s->begin()) {
        return nullptr;
    }
    const Entry& e = *--it;
    const auto off = static_cast<std::size_t>(w.data - e.src);
    const std::size_t row_bytes = mat_row_bytes(w);
    if (w.type != e.type || row_bytes != e.row_bytes ||
        off % (4 * row_bytes) != 0 ||
        off + std::size_t{w.rows} * row_bytes > e.src_bytes) {
        return nullptr;
    }
    return e.packed + off;
}

}  // namespace locus::backend
