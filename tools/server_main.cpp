#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "backend_cli.hpp"
#include "cli_spec.hpp"
#include "locus/engine/disk_prefix_key.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/model/transformer.hpp"
#include "locus/server/server.hpp"
#include "locus/sys/features.hpp"
#include "locus/tok/tokenizer.hpp"
#include "router_cli.hpp"

/**
 * OpenAI-compatible inference server (M4):
 *   locus-server [--backend NAME] <model.gguf> [port]
 *   locus-server --backends
 *   locus-server --router [--listen HOST:PORT] INSTANCE...
 */
namespace {

/**
 * Turns SIGINT / SIGTERM into OpenAiServer::stop(), so main returns and
 * the destructors run: the engine then writes its RAM prefix cache to
 * --prefix-cache-dir (i#416). The signals are blocked in the
 * constructor, which must run before any other thread starts (threads
 * inherit the mask), and only this object's thread takes them, through
 * sigwait. Before a server is attached (the model load) a signal exits
 * at once, as the default action would.
 */
class SignalStopper {
  public:
    SignalStopper() {
        sigemptyset(&set_);
        sigaddset(&set_, SIGINT);
        sigaddset(&set_, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &set_, nullptr);
        thread_ = std::thread([this] { run(); });
    }
    ~SignalStopper() {
        done_.store(true);
        pthread_kill(thread_.native_handle(), SIGTERM);  // wake sigwait
        thread_.join();
    }
    SignalStopper(const SignalStopper&) = delete;
    SignalStopper& operator=(const SignalStopper&) = delete;

    /** Sets the server a signal stops (nullptr: none any more). */
    void attach(locus::server::OpenAiServer* s) {
        std::lock_guard lk(mu_);
        server_ = s;
    }

  private:
    /** The first signal stops the server; a second one (or one before
     *  a server is attached) exits at once. */
    void run() {
        bool stopping = false;
        for (;;) {
            int sig = 0;
            sigwait(&set_, &sig);
            if (done_.load()) {
                return;
            }
            std::lock_guard lk(mu_);
            if (server_ == nullptr || stopping) {
                std::_Exit(128 + sig);
            }
            std::fprintf(stderr,
                         "locus-server: signal %d, stopping (again to "
                         "exit at once)\n",
                         sig);
            server_->stop();
            stopping = true;
        }
    }

    sigset_t set_;
    std::mutex mu_;
    locus::server::OpenAiServer* server_ = nullptr;
    std::atomic<bool> done_{false};
    std::thread thread_;
};

}  // namespace

int main(int argc, char** argv) {
    // locus-server --router ...: run locus-router instead (no model).
    if (argc > 1 && std::string(argv[1]) == "--router") {
        return locus_tools::run_router(argc, argv, 2);
    }
    auto args = locus_tools::parse_backend_args(argc, argv);
    if (args.list) {
        locus_tools::print_backends();
        return 0;
    }
    if (args.list_archs) {
        locus_tools::print_archs();
        return 0;
    }
    if (args.list_quants) {
        locus_tools::print_quants();
        return 0;
    }
    if (args.list_tokenizers) {
        locus_tools::print_tokenizers();
        return 0;
    }
    if (args.list_capabilities) {
        locus_tools::print_capabilities();
        return 0;
    }
    if (args.help) {
        std::printf(R"(usage: %s [options] <model.gguf> [port]

OpenAI + Anthropic compatible inference server (default port 8080):
  POST /v1/completions
  POST /v1/chat/completions
  POST /v1/messages            (Anthropic)
  POST /v1/embeddings          (CPU and CUDA backends, or
                               --embedding-model)
  POST /v1/rerank              (needs --rerank-model)
  POST /v1/classify            (needs --image-classifier)
  GET  /v1/models
  GET  /v1/models/{id}
  POST /v1/audio/speech        (needs --audio-decoder)
  POST /v1/images/generations  (needs --image-decoder or --image-model)
  POST /v1/images/edits        (needs --image-model with FLUX.1 Kontext)
  POST /v1/videos              (needs --video-model and ffmpeg)
  GET  /v1/videos/{id}[/content]
  GET  /health
  GET  /metrics                (path set by --metrics-path)

%s)",
                    argv[0], locus_tools::kCommonHelp);
        return 0;
    }
    if (!args.kv_type_ok) {
        std::fprintf(stderr,
                     "%s: unknown --kv-type '%s' (f32, f16, q8, q4)\n",
                     argv[0], args.kv_type_name.c_str());
        return 2;
    }
    if (!args.bad_number_flag.empty()) {
        std::fprintf(stderr,
                     "%s: %s needs a non-negative integer, got '%s'\n",
                     argv[0], args.bad_number_flag.c_str(),
                     args.bad_number_value.c_str());
        return 2;
    }
    if (!args.prefix_cache_dir.empty() && args.prefix_cache_disk_mib == 0) {
        std::fprintf(stderr, "%s: --prefix-cache-disk-mib must be > 0\n",
                     argv[0]);
        return 2;
    }
    if (args.positional.empty()) {
        std::fprintf(stderr,
                     "usage: %s [--backend NAME] <model.gguf> "
                     "[port]\nsee %s --help\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string model_path = args.positional[0];
    // Guarded parse: atoi read "808O" or "abc" as 0, which binds an
    // OS-assigned port, and let values above 65535 through (i#198).
    int port = 8080;
    if (args.positional.size() > 1 &&
        (!locus_tools::parse_nonneg_int(args.positional[1], port) ||
         port < 1 || port > 65535)) {
        std::fprintf(stderr, "%s: port must be 1..65535, got '%s'\n",
                     argv[0], args.positional[1].c_str());
        return 2;
    }
    // --register (i#388): checked before the model loads, so a typo
    // does not cost the load.
    locus::router::Registrar::Config reg;
    if (!args.register_url.empty() || !args.register_as.empty() ||
        !args.register_key_file.empty()) {
        std::string host;
        int rport = 0;
        if (!locus::router::parse_router_url(args.register_url, reg.router_host,
                                             reg.router_port) ||
            !locus::pipeline::parse_hostport(args.register_as, host, rport) ||
            host.empty() || rport <= 0 || args.register_key_file.empty()) {
            std::fprintf(stderr,
                         "%s: --register http://HOST:PORT needs --register-as "
                         "HOST:PORT and --register-key-file F\n",
                         argv[0]);
            return 2;
        }
        std::ifstream f(args.register_key_file);
        std::stringstream ss;
        ss << f.rdbuf();
        const auto keys = locus::router::parse_register_keys(ss.str());
        if (!f || keys.empty()) {
            std::fprintf(stderr, "%s: no key in --register-key-file %s\n",
                         argv[0], args.register_key_file.c_str());
            return 2;
        }
        reg.address = args.register_as;
        reg.key = keys.front();
    }

    SignalStopper stopper;  // before any other thread starts
    try {
        if (const std::string err = locus_tools::apply_device(args);
            !err.empty()) {
            std::fprintf(stderr, "%s: %s\n", argv[0], err.c_str());
            return 2;
        }
        auto g = locus::gguf::GgufFile::open(model_path);
        auto model = locus::model::TransformerModel::load(g);
        const auto& backend =
            locus::backend::resolve_backend(args.choice);
        model.use_backend(backend);
        const std::string gpu_warn = locus::backend::gpu_idle_warning(
            locus::backend::backend_defaulted(args.choice), backend);
        locus::backend::vulkan_set_prefill_f16(args.prefill_f16);
        auto tok_ptr = locus::tok::tokenizer_from_gguf(g);
        auto& tok = *tok_ptr;

        locus::server::OpenAiServer::Options opt;
        // Advertise the basename, not the absolute path: /v1/models
        // and /metrics echo model_name to unauthenticated clients.
        opt.model_name = locus_tools::public_model_id(model_path);
        if (const auto a = g.get_string("general.architecture")) {
            opt.model_arch = std::string(*a);
        }
        if (args.ctx > 0) {
            opt.engine.n_blocks = (args.ctx + 15) / 16;
        }
        opt.engine.speculative = args.speculative;
        if (args.spec_draft > 0) {
            opt.engine.spec_draft = args.spec_draft;
        }
        opt.engine.spec_adaptive = !args.spec_fixed;
        opt.engine.kv_type = args.kv_type;
        if (!args.prefix_cache_dir.empty()) {
            opt.engine.disk_prefix = locus::engine::open_disk_prefix(
                args.prefix_cache_dir,
                std::uint64_t{args.prefix_cache_disk_mib} << 20, g,
                model_path, model, args.kv_type);
        }
        opt.chat_template =
            locus::chat::ChatTemplate::from_gguf(g);
        if (!args.metrics_path.empty()) {
            opt.metrics_path = args.metrics_path;
        }
        // Capped at a day, so the milliseconds fit in 32 bits.
        opt.sse_keepalive_ms =
            std::min<std::uint32_t>(args.sse_keepalive, 86400) * 1000;
        opt.audio_decoder_path = args.audio_decoder;
        opt.embedding_model_path = args.embedding_model;
        opt.rerank_model_path = args.rerank_model;
        opt.image_classifier_path = args.image_classifier;
        if (args.max_tokens > 0) {
            opt.max_tokens_cap = args.max_tokens;
        }
        if (args.audio_max_tokens > 0) {
            opt.audio_max_tokens = args.audio_max_tokens;
        }
        opt.image_decoder_path = args.image_decoder;
        opt.image_model_dir = args.image_model;
        opt.lora_dir = args.lora_dir;
        opt.video_model_dir = args.video_model;
        opt.ffmpeg_path = args.ffmpeg;
        opt.model_path = model_path;
        locus::server::OpenAiServer server(model, tok, opt);
        std::fprintf(
            stderr,
            "%s\nbackend: %s | chat template: %s\n",
            locus::sys::to_string(locus::sys::detect()).c_str(),
            std::string(model.active_backend().name).c_str(),
            std::string(opt.chat_template.name()).c_str());
        if (!gpu_warn.empty()) {
            std::fprintf(stderr, "%s\n", gpu_warn.c_str());
        }
        std::fprintf(stderr, "locus-server listening on :%d\n", port);
        // Registers a moment after this, once listen has bound, and
        // deregisters when the server stops (a kill leaves it to run
        // out at the router).
        std::unique_ptr<locus::router::Registrar> registrar;
        if (!reg.router_host.empty()) {
            registrar = std::make_unique<locus::router::Registrar>(reg);
            registrar->start();
            std::fprintf(stderr, "registering as %s with %s:%d\n",
                         reg.address.c_str(), reg.router_host.c_str(),
                         reg.router_port);
        }
        stopper.attach(&server);
        const bool bound =
            server.listen(locus::server::OpenAiServer::kAnyHost, port);
        stopper.attach(nullptr);
        if (!bound) {
            std::fprintf(stderr, "error: cannot bind port %d\n",
                         port);
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
