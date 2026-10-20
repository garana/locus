# locus

A pure C/C++ LLM inference server: a full serving engine whose
distinctive lever is running models that do not fit in RAM/VRAM. It
began by filling the gap between llama.cpp and the Python serving
stacks (vLLM, LightLLM), and now spans:

- vendor-neutral compute: CPU, Vulkan (incl. Apple GPUs via MoltenVK),
  and CUDA,
- token-block paged KV cache with copy-on-write prefix sharing, and
  continuous (iteration-level) batching,
- OpenAI- and Anthropic-compatible HTTP API, with sampling,
  constrained decoding, prompt caching, and speculative decoding,
- weight streaming for bigger-than-memory models, now extending across
  hosts (pipeline + expert parallelism, in progress) to pool memory,
- a minimal, auditable, vendored dependency surface.

See [docs/DESIGN.md](docs/DESIGN.md) for the full MVP design and
milestone plan, [docs/CLIENTS.md](docs/CLIENTS.md) for supported
clients/endpoints (OpenAI + Anthropic, incl. Claude Code and GitHub
Copilot CLI offline), and [docs/COMPARISON.md](docs/COMPARISON.md)
for how locus compares to other engines.

## Status

| Milestone | Description                             | State       |
|-----------|-----------------------------------------|-------------|
| M0        | Scaffolding, CMake, Catch2, allocator   | done        |
| M1        | GGUF loader + tokenizer                 | done        |
| M2        | CPU forward pass, single sequence       | done        |
| M3        | Paged KV + continuous batching          | done        |
| M4        | HTTP server (OpenAI API, SSE)           | done        |
| M5        | Vulkan backend                          | done        |

All MVP milestones are complete. The post-MVP roadmap (BPE
tokenizer, chat templates, K-quants, MoE, MLA, Kimi Delta
Attention -- the path to DeepSeek V3/R1, Kimi K2, and Kimi K3) is
in docs/DESIGN.md section 7. Greedy output is validated
token-exact against llama.cpp on stories260K, Llama-3.2-1B
(Q8_0 and Q4_K_M), DeepSeek-V2-Lite, Moonlight-16B-A3B, and
GLM-5.2 744B (UD-IQ1_S, streamed from a 203GB split GGUF on a
32GB machine -- see the R8 notes in DESIGN.md). Streaming
models decode with madvise readahead (default-on), optional
threaded matvecs (LOCUS_THREADS) and static-weight pinning
(LOCUS_PIN_STATIC) per DESIGN.md R9; GLM-5.2's DSA lightning
indexer is implemented, so its context is no longer capped at
2048 tokens.

The server exposes the table-stakes serving features (DESIGN.md
R12), each default-off so greedy stays token-exact: sampling
(temperature, top-k/p, min-p, penalties, seed), JSON-constrained
decoding, prefix/prompt caching, and prompt-lookup speculative
decoding. The API surface (DESIGN.md R14/R15) adds POST
/v1/embeddings, GET /v1/models, GET /metrics (Prometheus), and
OpenAI-shaped logprobs plus usage accounting, alongside the
OpenAI and Anthropic (/v1/messages) chat/completion endpoints. A
request whose client disconnects is cancelled (streaming and
non-streaming), so the engine stops decoding it instead of running
the abandoned request to max_tokens.

## Building

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build

## Testing

    cmake --build build --target locus_tests
    ./build/tests/locus_tests

## Math backends

Every SIMD/GPU variant is compiled into the one binary; at startup
the best one supported by the running machine is picked (pbw-style
runtime dispatch). Inspect and override:

    locus-run --backends
    locus-run --backend scalar model.gguf "prompt" 32
    LOCUS_BACKEND=neon locus-server model.gguf 8080

Current entries: neon (arm64), sse4 (x86-64), scalar (reference),
vulkan, and cuda. neon/sse4 are hybrid backends -- F32/Q8_0 matvec
vectorized, the rest scalar. cuda runs F32/Q8_0/Q4_K/IQ1_S matvec
on an NVIDIA GPU through a paged weight pool (LOCUS_GPU_POOL_MB
budget, LRU eviction, one-step-ahead prefetch on a second
stream; other types scalar), validated on sm_50 (GTX 750 Ti,
CUDA 12.x). vulkan is the full GPU forward pass instead:
matmul shaders (F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K, weights
resident, uploaded once), rmsnorm, RoPE, SwiGLU, and paged
attention reading K/V from a GPU-mapped cache pool, all recorded
as one command batch per token. MoE layers route on the GPU too
(moe_route + batched expert matvecs) when the pool is unbudgeted
and all experts fit in 80% of available device memory, else on
the host; LOCUS_VK_PROFILE=1|2 prints per-token host, wait
and GPU time (2: per kernel). On an M4 Max (MoltenVK) decode runs
~95 tok/s on llama-3.2-1b Q8_0 and ~49 tok/s on
deepseek-coder-v2-lite Q8_0 (#94). Prompts ingest in chunks of up
to 512 tokens as tiled GPU matmuls (vulkan_forward_batch: each
weight read once per chunk, routed experts grouped per expert,
token-exact to per-token prefill); on the same M4 Max prefill
runs ~855 tok/s on a 2.5k-token prompt (llama-3.2-1b) and ~146
tok/s on a 3.1k-token prompt (deepseek-coder-v2-lite), up from 66
and 12 tok/s token by token. Every selectable backend must
reproduce the llama.cpp golden output token-exact (tested, single
and concurrent streams). The Vulkan full-GPU path is validated
end to end on a real MoE + MLA model: deepseek-v2-lite runs
token-exact on a Raspberry Pi 5 (VideoCore VII) and fits in
~2GB RSS for a 10GB model via the paged weight pool (upload only
routed experts, LOCUS_GPU_POOL_MB budget with LRU eviction).
Whether the GPU beats the CPU is device- and workload-specific:
a desktop/MoltenVK GPU wins at real-model matvec sizes (2048^2
f32: ~750us vs ~3550us scalar), but a tile GPU streaming a
bigger-than-pool model (Pi's V3D on deepseek) is compute-bound
and slower than NEON -- there Vulkan is a capability win, not a
speed win. Remaining GPU work (IQ Vulkan shaders) and x86
avx2/avx512 (avx2 kernel exists, pending an AVX2 host) slot into
src/locus/backend/ as they land.

## Dependencies

No external packages at runtime. Vendored (pinned, checksummed,
with license files; see docs/DESIGN.md section 6): Catch2 (tests
only), cpp-httplib, nlohmann/json. Optional, auto-detected at
configure time:

- Vulkan loader (GPU backend, milestone M5)
- SIMD compiler flags per arch (neon/sse4/avx2/avx512); kernel
  variants are selected at runtime via `locus::sys::detect()`
