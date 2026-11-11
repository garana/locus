#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "catch_amalgamated.hpp"
#include "httplib.h"
#include "json.hpp"
#include "locus/engine/disk_prefix_key.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/server/server.hpp"
#include "locus/tok/tokenizer.hpp"

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

/** Server bound to an ephemeral port, serving on a thread. */
struct TestServer {
    TestServer(const locus::model::TransformerModel& m,
               const locus::tok::SpmTokenizer& tok,
               locus::server::OpenAiServer::Options opt)
        : server(m, tok, std::move(opt)) {
        port = server.bind_any_port("127.0.0.1");
        REQUIRE(port > 0);
        thread = std::jthread([this] { server.listen_after_bind(); });
    }
    ~TestServer() {
        server.stop();
        thread.join();
    }

    locus::server::OpenAiServer server;
    int port = -1;
    std::jthread thread;
};

/** POSTs a greedy completion; @returns the parsed reply. */
json complete(int port, const std::string& prompt) {
    httplib::Client client("127.0.0.1", port);
    const json req{{"prompt", prompt}, {"max_tokens", 8},
                   {"temperature", 0}};
    auto res = client.Post("/v1/completions", req.dump(),
                           "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    return json::parse(res->body);
}

}  // namespace

TEST_CASE("server serves a prompt prefix from disk after a restart",
          "[server][disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    auto tok = locus::tok::SpmTokenizer::from_gguf(g);
    const fs::path dir = fs::temp_directory_path() /
                         ("locus-sdp-" + std::to_string(::getpid()));
    fs::remove_all(dir);
    const std::string prompt =
        "Once upon a time, there was a little girl named Lily who loved "
        "to explore the forest near her home every day. One day she "
        "found a big red ball under a tree.";
    const auto options = [&] {
        locus::server::OpenAiServer::Options opt;
        opt.engine.disk_prefix = locus::engine::open_disk_prefix(
            dir.string(), 64 << 20, g, model_path(), model,
            locus::kv::KvType::kF32);
        return opt;
    };

    json first;
    {
        TestServer ts(model, tok, options());
        first = complete(ts.port, prompt);
        REQUIRE(first["usage"]["prompt_tokens_details"]["cached_tokens"] ==
                0);
    }  // shutdown: the engine writes its RAM prefix cache to disk
    REQUIRE(fs::exists(dir));
    REQUIRE_FALSE(fs::is_empty(dir));

    // A new server on the same directory: nothing in RAM, the prefix
    // comes from disk, and the output is the same.
    TestServer ts(model, tok, options());
    const json again = complete(ts.port, prompt);
    REQUIRE(again["usage"]["prompt_tokens_details"]["cached_tokens"] >= 16);
    REQUIRE(again["choices"][0]["text"] == first["choices"][0]["text"]);
    fs::remove_all(dir);
}

TEST_CASE("the disk prefix key follows the model file and its settings",
          "[disk_prefix]") {
    if (!fs::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    auto model = locus::model::TransformerModel::load(g);
    std::size_t bb = 0;
    const auto k = locus::engine::disk_key_for(g, model_path(), model,
                                               locus::kv::KvType::kF32, bb);
    REQUIRE(k.model_size == fs::file_size(model_path()));
    REQUIRE(k.backend == model.active_backend().name);
    REQUIRE(k.block_tokens > 0);
    REQUIRE(bb > 0);
    // The same file gives the same key; another KV type another one.
    std::size_t bb2 = 0;
    const auto same = locus::engine::disk_key_for(
        g, model_path(), model, locus::kv::KvType::kF32, bb2);
    REQUIRE(same.model_hash == k.model_hash);
    REQUIRE(same.overrides == k.overrides);
    REQUIRE(bb2 == bb);
    const auto q8 = locus::engine::disk_key_for(
        g, model_path(), model, locus::kv::KvType::kQ8, bb2);
    REQUIRE(q8.kv_type == locus::kv::KvType::kQ8);
    REQUIRE(bb2 != bb);

    // Q8_K activations change the K/V bits, so they are in the key.
    const bool q8k = model.q8k_activations();
    model.use_q8k_activations(!q8k);
    REQUIRE(model.q8k_activations() != q8k);
    const auto flipped = locus::engine::disk_key_for(
        g, model_path(), model, locus::kv::KvType::kF32, bb2);
    REQUIRE(flipped.overrides != k.overrides);
    model.use_q8k_activations(q8k);

    // So do the LOCUS_* switches that pick other kernels.
    REQUIRE(std::getenv("LOCUS_BATCH_DEQUANT") == nullptr);
    ::setenv("LOCUS_BATCH_DEQUANT", "1", 1);
    const auto dequant = locus::engine::disk_key_for(
        g, model_path(), model, locus::kv::KvType::kF32, bb2);
    ::unsetenv("LOCUS_BATCH_DEQUANT");
    REQUIRE(dequant.overrides != k.overrides);

    // A copy of the file with another mtime is another model.
    const fs::path copy = fs::temp_directory_path() /
                          ("locus-dpk-" + std::to_string(::getpid()) +
                           ".gguf");
    fs::copy_file(model_path(), copy, fs::copy_options::overwrite_existing);
    fs::last_write_time(copy, fs::last_write_time(model_path()) +
                                  std::chrono::seconds(10));
    const auto moved = locus::engine::disk_key_for(
        g, copy.string(), model, locus::kv::KvType::kF32, bb2);
    REQUIRE(moved.model_hash != k.model_hash);
    fs::remove(copy);
}
