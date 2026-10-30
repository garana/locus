#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "json.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/t5_encoder.hpp"

using locus::image::T5Encoder;
using locus::tok::TokenId;

namespace {

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";
const std::string kModel = kDir + "models/t5-xxl-enc-f16.gguf";

// The synthetic model of scripts/make-t5-goldens.py synth.
constexpr std::uint32_t kVocab = 32, kWidth = 16, kHeads = 4, kHeadDim = 4,
                        kFfn = 24, kBlocks = 2, kBuckets = 8, kMaxDist = 16,
                        kTokens = 20;

struct Tensor {
    std::string name;
    std::vector<std::uint64_t> shape;  ///< torch order: [out, in]
};

/** The script's tensor list, in its LCG order. */
std::vector<Tensor> synth_tensors() {
    const std::uint64_t w = kWidth, inner = kHeads * kHeadDim;
    std::vector<Tensor> t = {
        {"token_embd.weight", {kVocab, w}},
        {"enc.blk.0.attn_rel_b.weight", {kBuckets, kHeads}}};
    for (std::uint32_t n = 0; n < kBlocks; ++n) {
        const std::string b = "enc.blk." + std::to_string(n) + ".";
        t.push_back({b + "attn_norm.weight", {w}});
        t.push_back({b + "attn_q.weight", {inner, w}});
        t.push_back({b + "attn_k.weight", {inner, w}});
        t.push_back({b + "attn_v.weight", {inner, w}});
        t.push_back({b + "attn_o.weight", {w, inner}});
        t.push_back({b + "ffn_norm.weight", {w}});
        t.push_back({b + "ffn_gate.weight", {kFfn, w}});
        t.push_back({b + "ffn_up.weight", {kFfn, w}});
        t.push_back({b + "ffn_down.weight", {w, kFfn}});
    }
    t.push_back({"enc.output_norm.weight", {w}});
    return t;
}

/**
 * A t5encoder GGUF image with the script's F32 weights.
 *
 * @param heads, head_dim The values written to the keys (the tensors
 *     keep the script's shapes).
 * @param bf16 Every tensor as BF16 (rounded to nearest even), like
 *     FLUX's T5 converted, instead of F32.
 */
std::vector<std::byte> synth_t5(std::uint32_t heads = kHeads,
                                std::uint32_t head_dim = kHeadDim,
                                bool bf16 = false) {
    const auto tensors = synth_tensors();
    std::vector<std::vector<float>> data;
    std::uint32_t lcg = 146;
    for (const auto& t : tensors) {
        std::size_t n = 1;
        for (auto d : t.shape) n *= d;
        std::vector<float> v(n);
        for (auto& x : v) {
            lcg = lcg * 1664525u + 1013904223u;
            x = (static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f) * 0.5f;
        }
        data.push_back(std::move(v));
    }
    GgufBuilder g;
    g.header(tensors.size(), 9);
    g.kv_string("general.architecture", "t5encoder");
    g.kv_u32("t5encoder.embedding_length", kWidth);
    g.kv_u32("t5encoder.feed_forward_length", kFfn);
    g.kv_u32("t5encoder.block_count", kBlocks);
    g.kv_u32("t5encoder.attention.head_count", heads);
    g.kv_u32("t5encoder.attention.key_length", head_dim);
    g.kv_u32("t5encoder.attention.relative_buckets_count", kBuckets);
    g.kv_u32("t5encoder.attention.relative_max_distance", kMaxDist);
    g.kv_f32("t5encoder.attention.layer_norm_rms_epsilon", 1e-6f);
    std::uint64_t offset = 0;
    const std::size_t elem = bf16 ? 2 : 4;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& s = tensors[i].shape;
        const std::vector<std::uint64_t> dims(s.rbegin(), s.rend());
        g.tensor(tensors[i].name, dims, bf16 ? 30 /* BF16 */ : 0 /* F32 */,
                 offset);
        offset = (offset + data[i].size() * elem + 31) & ~31ull;
    }
    g.pad();
    for (const auto& v : data) {
        for (float x : v) {
            if (!bf16) {
                g.f32(x);
                continue;
            }
            std::uint32_t u;
            std::memcpy(&u, &x, 4);
            u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
            g.u8(static_cast<std::uint8_t>((u >> 16) & 0xff));
            g.u8(static_cast<std::uint8_t>(u >> 24));
        }
        while (g.bytes().size() % 32 != 0) g.zeros(1);
    }
    auto span = g.bytes();
    return {span.begin(), span.end()};
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::vector<char> b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

/**
 * @returns The largest difference between two rows of `width` floats,
 *     over the reference row's RMS.
 */
double row_error(const float* got, const float* ref, std::uint32_t width) {
    double ss = 0.0, worst = 0.0;
    for (std::uint32_t i = 0; i < width; ++i) {
        ss += double{ref[i]} * ref[i];
        worst = std::max(worst, std::fabs(double{got[i]} - ref[i]));
    }
    return worst / std::sqrt(ss / width);
}

}  // namespace

// transformers' bucket table for T5 (32 buckets, max distance 128), as
// docs/models/pixart-alpha.md section 3.4 lists it; the edges at 16, 32
// and 64 are where the log formula gives an exact integer.
TEST_CASE("t5 relative position buckets match the reference table",
          "[t5][image]") {
    const auto b = [](std::uint32_t n) -> std::uint32_t {
        if (n < 8) return n;
        if (n < 12) return 8;
        if (n < 16) return 9;
        if (n < 23) return 10;
        if (n < 32) return 11;
        if (n < 46) return 12;
        if (n < 64) return 13;
        if (n < 91) return 14;
        return 15;
    };
    for (std::int32_t n = 0; n < 512; ++n) {
        INFO("distance " << n);
        const std::uint32_t want = b(static_cast<std::uint32_t>(n));
        REQUIRE(T5Encoder::relative_bucket(-n, 32, 128) == want);
        if (n > 0) {
            REQUIRE(T5Encoder::relative_bucket(n, 32, 128) == 16 + want);
        }
    }
}

// VALUES, on every box: the synthetic encoder against torch's
// T5EncoderModel with the same weights (scripts/make-t5-goldens.py
// synth). Pins the rms norms, the bias per distance bucket (20 tokens
// reach past max_distance 16), the unscaled attention and the gated
// tanh GELU.
TEST_CASE("t5 encoder matches torch on a synthetic model", "[t5][image]") {
    const auto img = synth_t5();
    auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = T5Encoder::load(g);
    REQUIRE(enc.width() == kWidth);
    REQUIRE(enc.vocab() == kVocab);
    std::vector<TokenId> ids;
    for (std::uint32_t i = 0; i + 1 < kTokens; ++i) {
        ids.push_back(static_cast<TokenId>((i * 7 + 3) % kVocab));
    }
    ids.push_back(1);
    const auto& op = locus::backend::best_backend().ops;
    const auto got = enc.encode(ids, op);
    const auto ref = read_f32(kDir + "golden/t5_synth.f32");
    REQUIRE(ref.size() == std::size_t{kTokens} * kWidth);
    REQUIRE(got.size() == ref.size());
    for (std::uint32_t t = 0; t < kTokens; ++t) {
        INFO("token " << t);
        REQUIRE(row_error(&got[std::size_t{t} * kWidth],
                          &ref[std::size_t{t} * kWidth], kWidth) <= 1e-5);
    }
    const std::vector<TokenId> bad{static_cast<TokenId>(kVocab)};
    REQUIRE_THROWS_AS(enc.encode(bad, op), std::invalid_argument);
    REQUIRE_THROWS_AS(enc.encode({}, op), std::invalid_argument);
}

// FLUX's T5 converts to BF16 (the norms and the relative bias as F32 in
// the converter, here all BF16 to cover the loader's widening too): the
// same model with BF16-rounded weights stays within the rounding of the
// F32 golden.
TEST_CASE("t5 encoder reads BF16 weights", "[t5][image]") {
    const auto img = synth_t5(kHeads, kHeadDim, true);
    auto g = locus::gguf::GgufFile::parse(img);
    const auto enc = T5Encoder::load(g);
    std::vector<TokenId> ids;
    for (std::uint32_t i = 0; i + 1 < kTokens; ++i) {
        ids.push_back(static_cast<TokenId>((i * 7 + 3) % kVocab));
    }
    ids.push_back(1);
    const auto got = enc.encode(ids, locus::backend::best_backend().ops);
    const auto ref = read_f32(kDir + "golden/t5_synth.f32");
    REQUIRE(got.size() == ref.size());
    for (std::uint32_t t = 0; t < kTokens; ++t) {
        INFO("token " << t);
        REQUIRE(row_error(&got[std::size_t{t} * kWidth],
                          &ref[std::size_t{t} * kWidth], kWidth) <= 5e-2);
    }
}

// heads x head_dim = 2^32 wraps to 0 in 32 bits; the loader must refuse
// it before sizing any buffer, and refuse heads that do not match q.
TEST_CASE("t5 encoder rejects inconsistent head sizes", "[t5][image]") {
    const auto wrap = synth_t5(1u << 16, 1u << 16);
    auto g1 = locus::gguf::GgufFile::parse(wrap);
    REQUIRE_THROWS_AS(T5Encoder::load(g1), locus::gguf::Error);
    const auto wide = synth_t5(kHeads * 2, kHeadDim);
    auto g2 = locus::gguf::GgufFile::parse(wide);
    REQUIRE_THROWS_AS(T5Encoder::load(g2), locus::gguf::Error);
}

// PixArt's T5 v1.1 XXL (F16) against the torch F32 run of the same
// weights (scripts/make-t5-goldens.py real): the golden prompt, the
// empty negative prompt and, with LOCUS_HEAVY_TESTS, a 107-token
// caption (about 75 s more on an M2 Pro).
TEST_CASE("t5 xxl encoder matches the reference", "[t5][image][e2e]") {
    if (!std::filesystem::exists(kModel)) {
        SKIP("t5-xxl-enc-f16.gguf absent; see scripts/convert-t5-encoder.py");
    }
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto enc = T5Encoder::load(g);
    REQUIRE(enc.width() == 4096);
    std::ifstream jf(kDir + "golden/t5_encoder.json");
    REQUIRE(jf.good());
    const auto index = nlohmann::json::parse(jf);
    const auto ref = read_f32(kDir + "golden/t5_encoder.f32");
    const bool heavy = std::getenv("LOCUS_HEAVY_TESTS") != nullptr;
    // The CPU backend, and the GPU when a Vulkan device is usable.
    std::vector<std::pair<const char*, const locus::backend::Ops*>> legs = {
        {"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vulkan_backend_usable()) {
        legs.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    for (const auto& [leg, op] : legs) {
        std::size_t at = 0;
        for (const auto& c : index["cases"]) {
            const auto name = c["name"].get<std::string>();
            const auto ids = c["ids"].get<std::vector<TokenId>>();
            const auto rows = c["rows"].get<std::vector<std::uint32_t>>();
            const std::size_t next = at + rows.size() * 4096;
            REQUIRE(next <= ref.size());
            if (name == "long" && !heavy) {
                at = next;
                continue;
            }
            INFO(leg << " case " << name);
            const auto t0 = std::chrono::steady_clock::now();
            const auto got = enc.encode(ids, *op);
            const double secs =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
            double worst = 0.0;
            for (std::size_t r = 0; r < rows.size(); ++r) {
                worst = std::max(
                    worst, row_error(&got[std::size_t{rows[r]} * 4096],
                                     &ref[at + r * 4096], 4096));
            }
            // Measured on macOS arm64 (CPU): 4.3e-4 prompt, 2.5e-4
            // empty, 7.7e-4 long (sum order over 24 blocks; a wrong
            // bucket or norm is near 1).
            WARN("t5 " << leg << " " << name << ": max diff / row rms "
                       << worst << ", " << secs << " s");
            CHECK(worst <= 2e-3);
            at = next;
        }
    }
}
