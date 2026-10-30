#include "locus/image/janus_vision.hpp"

#include <stdexcept>

#include "locus/image/janus_chat.hpp"
#include "locus/tok/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "gguf_builder.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "png_parse.hpp"

namespace {

// Keep in sync with scripts/make-janus-goldens.py (vision).
constexpr std::uint32_t kRows = 16, kRowStep = 36;

const std::string kDir = std::string(LOCUS_SOURCE_DIR) + "/tests/";

std::vector<std::byte> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> c((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    std::vector<std::byte> b(c.size());
    std::memcpy(b.data(), c.data(), c.size());
    return b;
}

std::vector<std::int32_t> ints(const std::string& line) {
    std::vector<std::int32_t> v;
    std::istringstream ls(line);
    for (std::int32_t x; ls >> x;) v.push_back(x);
    return v;
}

double rms(std::span<const float> x) {
    double s = 0.0;
    for (float v : x) s += double{v} * v;
    return std::sqrt(s / static_cast<double>(x.size()));
}

/**
 * Compares the sampled rows of got (n x width, token-major) with ref
 * (kRows x width): REQUIREs the error's RMS within rel of the
 * reference's RMS, and reports the worst element.
 */
void check_rows(std::span<const float> got, std::span<const float> ref,
                std::uint32_t width, double rel) {
    double err = 0.0, worst = 0.0;
    for (std::uint32_t r = 0; r < kRows; ++r) {
        for (std::uint32_t c = 0; c < width; ++c) {
            const double d =
                double{got[std::size_t{r} * kRowStep * width + c]} -
                ref[std::size_t{r} * width + c];
            err += d * d;
            worst = std::max(worst, std::abs(d));
        }
    }
    err = std::sqrt(err / (double{kRows} * width));
    WARN("width " << width << ": error rms " << err << ", worst " << worst
                  << ", reference rms " << rms(ref));
    REQUIRE(err <= rel * rms(ref));
}

struct Golden {
    std::vector<float> vis, ali;  // kRows x 1024, kRows x 2048
    std::vector<std::int32_t> prompt, answer;
    double vis_rms = 0.0, ali_rms = 0.0;
};

Golden read_golden() {
    Golden g;
    const auto f = read_bytes(kDir + "golden/janus_vision.f32");
    REQUIRE(f.size() == std::size_t{kRows} * (1024 + 2048) * 4);
    g.vis.resize(std::size_t{kRows} * 1024);
    g.ali.resize(std::size_t{kRows} * 2048);
    std::memcpy(g.vis.data(), f.data(), g.vis.size() * 4);
    std::memcpy(g.ali.data(), f.data() + g.vis.size() * 4, g.ali.size() * 4);
    std::ifstream t(kDir + "golden/janus_understand.txt");
    std::string l1, l2, l3;
    REQUIRE(std::getline(t, l1));
    REQUIRE(std::getline(t, l2));
    REQUIRE(std::getline(t, l3));
    g.prompt = ints(l1);
    g.answer = ints(l2);
    std::istringstream(l3) >> g.vis_rms >> g.ali_rms;
    return g;
}

/** The reference image (384 x 384) as pixels in [-1, 1]. */
std::vector<float> golden_pixels(const locus::image::JanusVision& v) {
    const auto png = png_test::parse_stored_png(
        read_bytes(kDir + "golden/janus_apple.png"));
    REQUIRE(png.w == v.image_size());
    REQUIRE(png.h == v.image_size());
    return v.to_pixels(png.rgb);
}

const std::string kModel = kDir + "models/janus-pro-1b-bf16.gguf";

bool have_model() {
    if (!std::filesystem::exists(kModel)) return false;
    return locus::gguf::GgufFile::open(kModel).find_tensor(
               "vit.patch_embd.weight") != nullptr;
}

}  // namespace

// The reference's apply_sft_template_for_multi_turn_prompts output for
// this conversation with Janus-Pro's roles (printed from the Janus
// processor), with the default system prompt and with none: stripped
// turns, "\n\n" after a user turn, eos after an assistant turn,
// "<|Assistant|>:" at the end.
TEST_CASE("janus chat text matches the reference template",
          "[janus][image][vision]") {
    using locus::image::ChatTurn;
    const std::vector<ChatTurn> turns = {
        {"user", "<image_placeholder>\nWhat is on the table?"},
        {"assistant", "An apple."},
        {"user", "  What color is it?  "}};
    // <｜end▁of▁sentence｜>, the eos token's text.
    const std::string eos = "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81"
                            "sentence\xef\xbd\x9c>";
    const std::string body =
        "<|User|>: <image_placeholder>\nWhat is on the table?\n\n"
        "<|Assistant|>: An apple." + eos +
        "<|User|>: What color is it?\n\n<|Assistant|>:";
    REQUIRE(locus::image::janus_chat_text(turns,
                                          locus::image::kJanusSystemPrompt,
                                          eos) ==
            std::string(locus::image::kJanusSystemPrompt) + "\n\n" + body);
    REQUIRE(locus::image::janus_chat_text(turns, "", eos) == body);
    // The reply always follows a user turn.
    REQUIRE_THROWS_AS(locus::image::janus_chat_text(
                          std::vector<ChatTurn>{{"assistant", "x"}}, "", eos),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(locus::image::janus_chat_text({}, "", eos),
                      std::invalid_argument);
}

// The prompt ids the reference fed the model for the golden question,
// from the chat turns: system prompt, template, BOS, and the image
// placeholder expanded to 576 slots between begin and end of image.
TEST_CASE("janus chat ids match the reference prompt",
          "[janus][image][vision]") {
    if (!have_model()) {
        SKIP("janus-pro-1b-bf16.gguf with vit.* absent");
    }
    const Golden ref = read_golden();
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto tok = locus::tok::tokenizer_from_gguf(g);
    const auto v = locus::image::JanusVision::load(
        g, static_cast<std::uint32_t>(*g.get_uint("llama.embedding_length")));
    const std::vector<locus::image::ChatTurn> turns = {
        {"user", "<image_placeholder>\nWhat is on the table?"}};
    const std::string text = locus::image::janus_chat_text(
        turns, locus::image::kJanusSystemPrompt, tok->piece(tok->eos_id()));
    std::size_t images = 0;
    const auto ids = locus::image::janus_chat_ids(*tok, text, v, images);
    REQUIRE(images == 1);
    REQUIRE(ids == std::vector<locus::tok::TokenId>(ref.prompt.begin(),
                                                    ref.prompt.end()));
}

// i#241: the SigLIP tower and the aligner on the reference image,
// against the torch f32 reference (scripts/make-janus-goldens.py
// vision), on 16 sampled patches and on the RMS of the whole output.
TEST_CASE("janus vision encoder matches the reference features",
          "[janus][image][vision]") {
    if (!have_model()) {
        SKIP("janus-pro-1b-bf16.gguf with vit.* absent; see "
             "scripts/convert-janus.py");
    }
    const Golden ref = read_golden();
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto v = locus::image::JanusVision::load(
        g, static_cast<std::uint32_t>(*g.get_uint("llama.embedding_length")));
    REQUIRE(v.tokens() == 576);
    std::vector<std::pair<const char*, const locus::backend::Ops*>> legs = {
        {"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vulkan_backend_usable()) {
        // The whole tower on the GPU, activations resident.
        legs.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    for (const auto& [name, op] : legs) {
        DYNAMIC_SECTION(name) {
            const auto feat = v.features(golden_pixels(v), *op);
            check_rows(feat, ref.vis, 1024, 1e-4);
            REQUIRE(rms(feat) == Catch::Approx(ref.vis_rms).epsilon(1e-4));
            const auto ali = v.align(feat, *op);
            check_rows(ali, ref.ali, 2048, 1e-4);
            REQUIRE(rms(ali) == Catch::Approx(ref.ali_rms).epsilon(1e-4));
        }
    }
}

// i#241: the reference's answer to "What is on the table?" about the
// reference image, teacher-forced: the image enters as the aligned
// embeddings at the 576 <image_placeholder> positions (forward_layers'
// layer-0 input embedding), and each reference token must be locus's
// top token or within kMargin of it.
TEST_CASE("janus answers a question about the reference image",
          "[janus][image][vision][e2e]") {
    if (!have_model()) {
        SKIP("janus-pro-1b-bf16.gguf with vit.* absent");
    }
    if (std::getenv("LOCUS_HEAVY_TESTS") == nullptr) {
        SKIP("a 630-token prompt takes over a minute on a CPU; set "
             "LOCUS_HEAVY_TESTS=1");
    }
    constexpr float kMargin = 0.25f;
    const Golden ref = read_golden();
    auto g = locus::gguf::GgufFile::open(kModel);
    auto m = locus::model::TransformerModel::load(g);
    const auto* neon = locus::backend::find_backend("neon");
    if (neon != nullptr && neon->available) m.use_backend(*neon);
    const auto& op = m.active_backend().ops;
    const auto v = locus::image::JanusVision::load(g, m.hparams().n_embd);
    const auto emb = v.align(v.features(golden_pixels(v), op), op);

    const auto& hp = m.hparams();
    auto cache = m.make_cache();
    auto ws = m.make_workspace();
    std::vector<float> logits(hp.n_vocab);
    locus::kv::PagedKvCache::Seq seq;
    const std::uint32_t total =
        static_cast<std::uint32_t>(ref.prompt.size() + ref.answer.size());
    REQUIRE(cache.ensure_capacity(seq, total));
    std::size_t image_row = 0;
    const auto feed = [&](std::int32_t id) {
        if (id == v.placeholder()) {
            REQUIRE(image_row < v.tokens());
            m.forward_layers(
                0,
                std::span<const float>(emb).subspan(image_row * hp.n_embd,
                                                    hp.n_embd),
                0, hp.n_layers, cache, seq, ws, logits);
            ++image_row;
        } else {
            m.forward_layers(id, {}, 0, hp.n_layers, cache, seq, ws, logits);
        }
    };
    for (const std::int32_t id : ref.prompt) feed(id);
    REQUIRE(image_row == v.tokens());

    std::size_t exact = 0;
    for (std::size_t i = 0; i < ref.answer.size(); ++i) {
        const auto top = std::max_element(logits.begin(), logits.end());
        const float want = logits[static_cast<std::size_t>(ref.answer[i])];
        INFO("answer token " << i << ": reference " << ref.answer[i]
                             << " logit " << want << ", locus top "
                             << (top - logits.begin()) << " logit " << *top);
        REQUIRE(*top - want <= kMargin);
        exact += (top - logits.begin()) == ref.answer[i];
        if (i + 1 < ref.answer.size()) feed(ref.answer[i]);
    }
    INFO(exact << " of " << ref.answer.size() << " answer tokens are argmax");
    CHECK(exact == ref.answer.size());
    cache.release(seq);
}

// The tower on the reference image: the best CPU backend against the
// Vulkan image ops.
TEST_CASE("bench: janus vision tower", "[.][bench][janus]") {
    if (!have_model()) SKIP("janus-pro-1b-bf16.gguf with vit.* absent");
    auto g = locus::gguf::GgufFile::open(kModel);
    const auto v = locus::image::JanusVision::load(
        g, static_cast<std::uint32_t>(*g.get_uint("llama.embedding_length")));
    const auto px = golden_pixels(v);
    std::vector<std::pair<const char*, const locus::backend::Ops*>> legs = {
        {"cpu", &locus::backend::best_backend().ops}};
    if (locus::backend::vulkan_backend_usable()) {
        legs.push_back({"vulkan", &locus::backend::vulkan_image_ops()});
    }
    for (const auto& [name, op] : legs) {
        for (int rep = 0; rep < 3; ++rep) {
            using clk = std::chrono::steady_clock;
            const auto t0 = clk::now();
            const auto feat = v.features(px, *op);
            const auto t1 = clk::now();
            const auto ali = v.align(feat, *op);
            const auto t2 = clk::now();
            const auto sec = [](clk::duration d) {
                return std::chrono::duration<double>(d).count();
            };
            WARN(name << ": " << sec(t2 - t0) << " s (features "
                      << sec(t1 - t0) << ", align " << sec(t2 - t1) << ")");
            REQUIRE(std::isfinite(ali[0]));
        }
    }
}

namespace {

/** One tensor of the synthetic tower: GGUF dims (ne0 first) and data. */
struct SynthTensor {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;  // 0 F32, 30 BF16
    std::vector<std::byte> data;
};

/**
 * A seeded Janus vision GGUF image: a 64 x 64 image in 8 x 8 patches
 * (64 tokens), width 64 in 4 heads, FFN 128, 2 blocks, an aligner to
 * n_embd 48. Matrices are BF16, everything else F32.
 */
std::vector<std::byte> synth_tower_gguf(std::uint32_t n_embd) {
    constexpr std::uint32_t kImage = 64, kPatch = 8, kW = 64, kHeads = 4,
                            kFfn = 128, kBlocks = 2;
    constexpr std::uint32_t kTokens = (kImage / kPatch) * (kImage / kPatch);
    std::mt19937 rng(386);
    std::vector<SynthTensor> ts;
    const auto f32 = [&](std::string name, std::vector<std::uint64_t> dims,
                         float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        std::size_t n = 1;
        for (auto x : dims) n *= x;
        std::vector<std::byte> b(n * 4);
        for (std::size_t i = 0; i < n; ++i) {
            const float v = d(rng);
            std::memcpy(&b[i * 4], &v, 4);
        }
        ts.push_back({std::move(name), std::move(dims), 0, std::move(b)});
    };
    const auto bf16 = [&](std::string name, std::uint64_t rows,
                          std::uint64_t cols) {
        // About unit-variance outputs for unit-variance inputs.
        const float a = 1.0f / std::sqrt(static_cast<float>(cols));
        std::uniform_real_distribution<float> d(-a, a);
        std::vector<std::byte> b(rows * cols * 2);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            std::uint32_t u;
            const float v = d(rng);
            std::memcpy(&u, &v, 4);
            const auto h = static_cast<std::uint16_t>(u >> 16);
            std::memcpy(&b[i * 2], &h, 2);
        }
        ts.push_back({std::move(name), {cols, rows}, 30, std::move(b)});
    };
    f32("vit.patch_embd.weight", {kPatch, kPatch, 3, kW}, -0.05f, 0.05f);
    f32("vit.patch_embd.bias", {kW}, -0.1f, 0.1f);
    f32("vit.pos_embd", {kW, kTokens}, -0.5f, 0.5f);
    for (std::uint32_t i = 0; i < kBlocks; ++i) {
        const std::string b = "vit.blk." + std::to_string(i) + ".";
        f32(b + "ln1.weight", {kW}, 0.8f, 1.2f);
        f32(b + "ln1.bias", {kW}, -0.1f, 0.1f);
        bf16(b + "attn_qkv.weight", 3 * kW, kW);
        f32(b + "attn_qkv.bias", {3 * kW}, -0.1f, 0.1f);
        bf16(b + "attn_out.weight", kW, kW);
        f32(b + "attn_out.bias", {kW}, -0.1f, 0.1f);
        f32(b + "ln2.weight", {kW}, 0.8f, 1.2f);
        f32(b + "ln2.bias", {kW}, -0.1f, 0.1f);
        bf16(b + "ffn_up.weight", kFfn, kW);
        f32(b + "ffn_up.bias", {kFfn}, -0.1f, 0.1f);
        bf16(b + "ffn_down.weight", kW, kFfn);
        f32(b + "ffn_down.bias", {kW}, -0.1f, 0.1f);
    }
    f32("vit.post_ln.weight", {kW}, 0.8f, 1.2f);
    f32("vit.post_ln.bias", {kW}, -0.1f, 0.1f);
    bf16("vit.aligner.0.weight", n_embd, kW);
    f32("vit.aligner.0.bias", {n_embd}, -0.1f, 0.1f);
    bf16("vit.aligner.2.weight", n_embd, n_embd);
    f32("vit.aligner.2.bias", {n_embd}, -0.1f, 0.1f);

    GgufBuilder g;
    g.header(ts.size(), 12)
        .kv_string("general.architecture", "llama")
        .kv_u32("janus.vision.image_size", kImage)
        .kv_u32("janus.vision.patch_size", kPatch)
        .kv_u32("janus.vision.embedding_length", kW)
        .kv_u32("janus.vision.head_count", kHeads)
        .kv_u32("janus.vision.feed_forward_length", kFfn)
        .kv_u32("janus.vision.block_count", kBlocks)
        .kv_u32("janus.vision.image_placeholder", 100)
        .kv_u32("janus.vision.end_of_image", 101)
        .kv_u32("janus.begin_of_image", 102)
        .kv_f32("janus.vision.layer_norm_eps", 1e-6f)
        .kv_u32("general.alignment", 32);
    std::uint64_t off = 0;
    for (const auto& t : ts) {
        g.tensor(t.name, t.dims, t.type, off);
        off += (t.data.size() + 31) / 32 * 32;
    }
    g.pad();
    for (const auto& t : ts) {
        g.raw_bytes(t.data);
        g.pad();
    }
    const auto b = g.bytes();
    return {b.begin(), b.end()};
}

}  // namespace

// The tower on the GPU (one resident VulkanSeq) against the CPU on a
// seeded 2-block model, so its ops (the patch matmul, affine
// LayerNorm, attention, erf GELU, residual adds) run in CI without the
// 4 GB Janus GGUF.
TEST_CASE("janus vision tower on vulkan matches the cpu on a tiny model",
          "[janus][image][vision][vulkan]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device");
    }
    constexpr std::uint32_t kEmbd = 48;
    const auto buf = synth_tower_gguf(kEmbd);
    // The weights are uploaded keyed by their host addresses; drop them
    // before buf goes, so a later test at the same address cannot hit
    // them.
    struct Release {
        const std::vector<std::byte>& b;
        ~Release() {
            locus::backend::vulkan_release_weights(b.data(), b.size());
        }
    } release{buf};
    const auto g = locus::gguf::GgufFile::parse(buf);
    const auto v = locus::image::JanusVision::load(g, kEmbd);
    REQUIRE(v.tokens() == 64);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> px(0, 255);
    std::vector<std::uint8_t> rgb(std::size_t{64} * 64 * 3);
    for (auto& c : rgb) c = static_cast<std::uint8_t>(px(rng));
    const auto chw = v.to_pixels(rgb);
    const auto& cpu = locus::backend::best_backend().ops;
    const auto& gpu = locus::backend::vulkan_image_ops();
    const auto f_cpu = v.features(chw, cpu);
    const auto f_gpu = v.features(chw, gpu);
    const auto a_cpu = v.align(f_cpu, cpu);
    const auto a_gpu = v.align(f_gpu, gpu);
    const auto rel = [](const std::vector<float>& got,
                        const std::vector<float>& ref) {
        double dd = 0.0, ss = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            dd += (double{got[i]} - ref[i]) * (double{got[i]} - ref[i]);
            ss += double{ref[i]} * ref[i];
        }
        return std::sqrt(dd / ss);
    };
    REQUIRE(f_gpu.size() == f_cpu.size());
    REQUIRE(a_gpu.size() == a_cpu.size());
    WARN("features " << rel(f_gpu, f_cpu) << ", aligned "
                     << rel(a_gpu, a_cpu));
    REQUIRE(rel(f_gpu, f_cpu) < 1e-5);
    REQUIRE(rel(a_gpu, a_cpu) < 1e-5);
}
