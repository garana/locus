#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "locus/backend/registry.hpp"
#include "locus/backend/variants.hpp"
#include "locus/backend/vulkan_forward.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/kv/paged_cache.hpp"
#include "locus/model/transformer.hpp"
#include "locus/pipeline/message.hpp"
#include "locus/pipeline/stage.hpp"
#include "locus/pipeline/stage_executor.hpp"

using locus::pipeline::Message;
using locus::pipeline::MsgType;
using locus::pipeline::PipelineStage;
using locus::pipeline::StageExecutor;
using locus::model::TransformerModel;
using Seq = locus::kv::PagedKvCache::Seq;

namespace {

std::string model_path() {
    return std::string(LOCUS_SOURCE_DIR) +
           "/tests/models/stories260K.gguf";
}

std::size_t argmax(const std::vector<float>& v) {
    return static_cast<std::size_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

void require_bit_eq(const std::vector<float>& a,
                    const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(std::bit_cast<std::uint32_t>(a[i]) ==
                std::bit_cast<std::uint32_t>(b[i]));
    }
}

void collect(StageExecutor& ex, int wake_r,
             std::vector<StageExecutor::Completion>& out,
             std::size_t want) {
    while (out.size() < want) {
        char buf[64];
        if (::read(wake_r, buf, sizeof(buf)) <= 0) {
            break;
        }
        ex.drain(out);
    }
}

}  // namespace

// i#567: a two-stage pipeline on the Vulkan GPU, as two locus-stage
// processes run it (PipelineStage and its StageExecutor), gives the
// single-process Vulkan forward's logits.
TEST_CASE("a two-stage vulkan pipeline gives the vulkan forward's logits",
          "[executor][vulkan][pipeline]") {
    if (!locus::backend::vulkan_backend_usable()) {
        SKIP("no usable Vulkan device / kernels not built");
    }
    if (!std::filesystem::exists(model_path())) {
        SKIP("model not present; run scripts/fetch-test-model.sh");
    }
    auto g = locus::gguf::GgufFile::open(model_path());
    const auto& vk = *locus::backend::find_backend("vulkan");
    auto full = TransformerModel::load(g);
    full.use_backend(vk);
    const std::uint32_t L = full.hparams().n_layers;
    const std::uint32_t V = full.hparams().n_vocab;
    const std::uint32_t k = L / 2;
    auto a = TransformerModel::load(g, 0, k);
    auto b = TransformerModel::load(g, k, L);
    a.use_backend(vk);
    b.use_backend(vk);
    PipelineStage sa(a, 0, k);
    PipelineStage sb(b, k, L);
    REQUIRE(sa.supports_batch());  // the executors batch on Vulkan

    constexpr int N = 4;
    constexpr std::uint32_t kPos = 20;  // more than one 16-token block
    auto token = [V](int i, std::uint32_t p) {
        return static_cast<locus::tok::TokenId>(1 + (p * 11 + i * 17) %
                                                    (V - 1));
    };

    SECTION("stepping one session at a time: bit for bit") {
        // The reference: vulkan_forward per session (single kernels, as
        // a stage's lone step runs).
        auto fc = full.make_cache();
        std::vector<Seq> fs(N), qa(N), qb(N);
        std::vector<float> want(V);
        for (std::uint32_t p = 0; p < kPos; ++p) {
            for (int i = 0; i < N; ++i) {
                CAPTURE(p, i);
                REQUIRE(fc.ensure_capacity(fs[i], 1));
                REQUIRE(locus::backend::vulkan_forward(full, token(i, p), fc,
                                                       fs[i], want));
                const Message h = sa.step(
                    qa[i], locus::pipeline::make_token(i, p, token(i, p)));
                REQUIRE(h.type == MsgType::kActivation);
                const Message out = sb.step(qb[i], h);
                REQUIRE(out.type == MsgType::kLogits);
                require_bit_eq(out.data, want);
            }
        }
    }

    SECTION("through both StageExecutors: the same tokens") {
        // Coalescing is timing-dependent, so a batch may take the multi
        // kernels: compare the top token, not the bits.
        int wa[2], wb[2];
        REQUIRE(::pipe(wa) == 0);
        REQUIRE(::pipe(wb) == 0);
        for (const int fd : {wa[1], wb[1]}) {
            ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        }
        auto fc = full.make_cache();
        std::vector<Seq> fs(N), qa(N), qb(N);
        std::vector<float> want(V);
        {
            StageExecutor ea(sa, wa[1]);
            StageExecutor eb(sb, wb[1]);
            for (std::uint32_t p = 0; p < 3; ++p) {
                std::vector<StageExecutor::Completion> da, db;
                for (int i = 0; i < N; ++i) {
                    ea.submit_step(i, &qa[i],
                                   locus::pipeline::make_token(i, p,
                                                               token(i, p)));
                }
                collect(ea, wa[0], da, N);
                REQUIRE(da.size() == static_cast<std::size_t>(N));
                for (const auto& d : da) {
                    REQUIRE(d.ok);
                    eb.submit_step(d.session, &qb[d.session], d.out);
                }
                collect(eb, wb[0], db, N);
                REQUIRE(db.size() == static_cast<std::size_t>(N));
                for (const auto& d : db) {
                    REQUIRE(d.ok);
                    REQUIRE(d.out.type == MsgType::kLogits);
                    const int i = d.session;
                    REQUIRE(fc.ensure_capacity(fs[i], 1));
                    REQUIRE(locus::backend::vulkan_forward(
                        full, token(i, p), fc, fs[i], want));
                    CAPTURE(p, i);
                    REQUIRE(argmax(d.out.data) == argmax(want));
                }
            }
            std::vector<StageExecutor::Completion> rel;
            for (int i = 0; i < N; ++i) {
                ea.submit_release(i, &qa[i]);
                eb.submit_release(i, &qb[i]);
            }
            collect(ea, wa[0], rel, N);
            rel.clear();
            collect(eb, wb[0], rel, N);
        }
        for (const int fd : {wa[0], wa[1], wb[0], wb[1]}) {
            ::close(fd);
        }
    }
}
