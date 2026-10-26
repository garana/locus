#include "locus/model/codec.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/tok/tokenizer.hpp"

using locus::model::OrpheusCodec;
using locus::model::OrpheusCodecSampler;
using locus::model::orpheus_deband;
using locus::tok::TokenId;

namespace {

constexpr TokenId kBase = 128266;  // <custom_token_10>, Orpheus band.

/** Builds the 7 flat tokens of one frame from its per-position codes. */
std::vector<TokenId> frame(const int (&code)[7]) {
    std::vector<TokenId> out;
    for (int i = 0; i < 7; ++i) {
        out.push_back(kBase + code[i] + i * OrpheusCodec::kCodebook);
    }
    return out;
}

}  // namespace

TEST_CASE("orpheus de-band splits a frame by SNAC stride", "[codec]") {
    // Position 0 -> coarse; 1,4 -> mid; 2,3,5,6 -> fine.
    const int c[7] = {948, 1988, 1575, 543, 1988, 3389, 219};
    const auto toks = frame(c);
    const auto out = orpheus_deband(toks, kBase);

    REQUIRE(out.coarse == std::vector<int>{948});
    REQUIRE(out.mid == std::vector<int>{1988, 1988});
    REQUIRE(out.fine == std::vector<int>{1575, 543, 3389, 219});
}

TEST_CASE("orpheus de-band concatenates frames per stream", "[codec]") {
    const int a[7] = {1, 2, 3, 4, 5, 6, 7};
    const int b[7] = {10, 20, 30, 40, 50, 60, 70};
    std::vector<TokenId> toks = frame(a);
    const auto bt = frame(b);
    toks.insert(toks.end(), bt.begin(), bt.end());

    const auto out = orpheus_deband(toks, kBase);
    REQUIRE(out.coarse == std::vector<int>{1, 10});
    REQUIRE(out.mid == std::vector<int>{2, 5, 20, 50});
    REQUIRE(out.fine == std::vector<int>{3, 4, 6, 7, 30, 40, 60, 70});
}

TEST_CASE("orpheus de-band rejects bad input", "[codec]") {
    SECTION("length not a multiple of 7") {
        std::vector<TokenId> toks = {kBase, kBase + 1};
        REQUIRE_THROWS_AS(orpheus_deband(toks, kBase),
                          std::invalid_argument);
    }
    SECTION("code out of band (below base)") {
        std::vector<TokenId> toks(7, kBase);
        toks[0] = kBase - 1;  // decodes to -1
        REQUIRE_THROWS_AS(orpheus_deband(toks, kBase),
                          std::invalid_argument);
    }
    SECTION("code out of band (past its sub-band)") {
        const int c[7] = {4096, 0, 0, 0, 0, 0, 0};  // position 0 == 4096
        const auto toks = frame(c);
        REQUIRE_THROWS_AS(orpheus_deband(toks, kBase),
                          std::invalid_argument);
    }
}

TEST_CASE("orpheus sampler buffers audio, stops on non-audio",
          "[codec]") {
    OrpheusCodecSampler s(kBase);
    const int c[7] = {948, 1988, 1575, 543, 1988, 3389, 219};
    const auto f = frame(c);

    for (auto t : f) {
        REQUIRE(s.push(t) == OrpheusCodecSampler::Status::kAudio);
    }
    REQUIRE(s.frames() == 1);
    REQUIRE_FALSE(s.has_partial());
    REQUIRE(s.stop_token() == -1);

    // A partial frame, then a terminator.
    REQUIRE(s.push(f[0]) == OrpheusCodecSampler::Status::kAudio);
    REQUIRE(s.has_partial());
    REQUIRE(s.push(128258) == OrpheusCodecSampler::Status::kStop);
    REQUIRE(s.stop_token() == 128258);

    // codes() returns only the complete frame.
    const auto out = s.codes();
    REQUIRE(out.coarse == std::vector<int>{948});
    REQUIRE(out.mid == std::vector<int>{1988, 1988});
    REQUIRE(out.fine == std::vector<int>{1575, 543, 3389, 219});
}

TEST_CASE("orpheus sampler codes_lenient drops a desynced frame",
          "[codec]") {
    OrpheusCodecSampler s(kBase);
    const int good[7] = {948, 1988, 1575, 543, 1988, 3389, 219};
    for (auto t : frame(good)) {
        REQUIRE(s.push(t) == OrpheusCodecSampler::Status::kAudio);
    }
    // A second frame of all-band-base ids: position 1 decodes to -4096,
    // so the frame is out of range (a desync).
    for (int i = 0; i < 7; ++i) {
        REQUIRE(s.push(kBase) == OrpheusCodecSampler::Status::kAudio);
    }
    REQUIRE(s.frames() == 2);
    REQUIRE_THROWS_AS(s.codes(), std::invalid_argument);  // strict.

    const auto lenient = s.codes_lenient();  // drops the bad frame.
    REQUIRE(lenient.coarse == std::vector<int>{948});
    REQUIRE(lenient.mid == std::vector<int>{1988, 1988});
    REQUIRE(lenient.fine == std::vector<int>{1575, 543, 3389, 219});
}

namespace {

std::string golden_dir() {
    return std::string(LOCUS_SOURCE_DIR) + "/tests/golden";
}

std::string orpheus_model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/orpheus-3b-0.1-ft-Q4_K_M.gguf";
}

std::vector<TokenId> read_ids(const std::string& path) {
    std::ifstream f(path);
    std::vector<TokenId> out;
    long v;
    while (f >> v) {
        out.push_back(static_cast<TokenId>(v));
    }
    return out;
}

std::vector<std::vector<int>> read_streams(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::vector<int>> out;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::vector<int> row;
        int v;
        while (ls >> v) {
            row.push_back(v);
        }
        out.push_back(row);
    }
    return out;
}

}  // namespace

// Deterministic, no model: de-banding the golden's own flat audio
// tokens must reproduce the golden's three code streams exactly. This
// pins the sampler to the reference de-band without any quantization
// or tokenization sensitivity.
TEST_CASE("orpheus de-band matches the golden codes", "[codec][golden]") {
    const auto toks_path = golden_dir() + "/orpheus_tokens.txt";
    const auto codes_path = golden_dir() + "/orpheus_codes.txt";
    if (!std::filesystem::exists(toks_path) ||
        !std::filesystem::exists(codes_path)) {
        SKIP("golden absent; run .local-dev/golden/run.sh");
    }
    auto flat = read_ids(toks_path);
    // The flat stream may carry a trailing non-audio stop token; drop
    // any tokens past the last whole frame of audio-band ids.
    std::size_t n_audio = 0;
    const TokenId hi =
        kBase + OrpheusCodec::kCodesPerFrame * OrpheusCodec::kCodebook;
    while (n_audio < flat.size() && flat[n_audio] >= kBase &&
           flat[n_audio] < hi) {
        ++n_audio;
    }
    flat.resize(n_audio - n_audio % OrpheusCodec::kCodesPerFrame);

    const auto got = orpheus_deband(flat, kBase);
    const auto want = read_streams(codes_path);
    REQUIRE(want.size() == 3);
    REQUIRE(got.coarse == want[0]);
    REQUIRE(got.mid == want[1]);
    REQUIRE(got.fine == want[2]);
}

// Model-gated: resolve the band base from the real vocab and drive the
// model greedily, feeding the codec sampler. The first frame must be
// token-exact with the golden (identical prompt + greedy argmax).
// Full-sequence exactness is quant-activation bound (locus f32 vs the
// reference's Q8_K intermediates on this coarse Q4_K_M), so the hard
// check is pinned to the first frame; the rest is required only to stay
// in-band. See the de-band-the-golden case above for the exact oracle.
TEST_CASE("orpheus model drives the codec sampler", "[codec][e2e]") {
    const auto path = orpheus_model_path();
    if (!std::filesystem::exists(path)) {
        SKIP("orpheus model absent");
    }
    auto g = locus::gguf::GgufFile::open(path);
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::tokenizer_from_gguf(g);

    REQUIRE(locus::model::orpheus_band_base(*tok) == kBase);

    // The reference prompt wrapping: START_ID, then the HF-style
    // tokenization (BOS + text), then the four end markers.
    constexpr TokenId kStart = 128259;
    const TokenId kEnd[4] = {128009, 128260, 128261, 128257};
    auto body = tok->encode(
        "tara: The quick brown fox jumps over the lazy dog.", true);
    std::vector<TokenId> ids;
    ids.push_back(kStart);
    ids.insert(ids.end(), body.begin(), body.end());
    for (auto e : kEnd) {
        ids.push_back(e);
    }

    auto cache = model.make_cache();
    auto ws = model.make_workspace();
    std::vector<float> logits(model.hparams().n_vocab);

    // First frame against the golden, teacher-forced: before each
    // golden code is fed, it must sit within 0.25 of the top logit.
    // Greedy decode follows the golden on arm64, but x86 SSE4's
    // default Q8 activations tip a near-tie in the first frame, as
    // with muse prompt 2 and deepseek41; this keeps the default path
    // checked on every lane.
    auto golden_flat = read_ids(golden_dir() + "/orpheus_tokens.txt");
    if (!golden_flat.empty()) {
        locus::kv::PagedKvCache::Seq tf;
        for (auto id : ids) {
            REQUIRE(cache.ensure_capacity(tf, 1));
            model.forward(id, cache, tf, ws, logits);
        }
        for (int i = 0; i < OrpheusCodec::kCodesPerFrame; ++i) {
            const float top =
                *std::max_element(logits.begin(), logits.end());
            INFO("code " << i);
            REQUIRE(logits[static_cast<std::size_t>(golden_flat[i])] >=
                    top - 0.25f);
            REQUIRE(cache.ensure_capacity(tf, 1));
            model.forward(golden_flat[i], cache, tf, ws, logits);
        }
        cache.release(tf);
    }

    // Then the sampler drives greedy decode to at least one frame.
    locus::kv::PagedKvCache::Seq seq;
    for (auto id : ids) {
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(id, cache, seq, ws, logits);
    }
    OrpheusCodecSampler sampler(kBase);
    std::vector<TokenId> gen;
    for (int i = 0; i < OrpheusCodec::kCodesPerFrame * 96; ++i) {
        const auto next = locus::model::argmax(logits);
        if (sampler.push(next) == OrpheusCodecSampler::Status::kStop) {
            break;
        }
        gen.push_back(next);
        REQUIRE(cache.ensure_capacity(seq, 1));
        model.forward(next, cache, seq, ws, logits);
    }
    cache.release(seq);

    REQUIRE(sampler.frames() >= 1);
}
