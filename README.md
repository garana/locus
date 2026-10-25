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
(Q8_0 and Q4_K_M), DeepSeek-V2-Lite, Moonlight-16B-A3B,
Qwen1.5-MoE-A2.7B (qwen2moe, Q4_K_M; CPU and Vulkan),
Qwen3-Coder-30B-A3B (qwen3moe, Q4_K_M; CPU and Vulkan), GLM-5.2
744B (UD-IQ1_S, streamed from a 203GB split GGUF on a 32GB machine --
see the R8 notes in DESIGN.md), Muse Spark Open 816B-A42B
(muse-spark, UD-IQ1_S, 176GB streamed on the CPU; one prompt
token-exact, one checked teacher-forced past a near-tie), and
DeepSeek-V4.1-Flash-1017 (deepseek41, UD-IQ1_S, 233GB streamed on the
CPU; token-exact against the draft llama.cpp#30817, including a
529-token prompt through the compressed history and indexer).
Streaming
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

Text-to-speech: with `--audio-decoder <snac.gguf>` and an
Orpheus-TTS model, POST /v1/audio/speech {input, voice} returns a
24 kHz WAV (or raw PCM16 with response_format "pcm"). Orpheus emits
audio as codec tokens; locus decodes them with the SNAC neural codec,
built from its conv1d kernels and matching the torch reference to
under 1e-6 on sampled speech (fixtures from
scripts/make-snac-goldens.py). Tokens are sampled (defaults
temperature 0.6, top_p 0.8, repetition_penalty 1.3, each overridable,
plus seed and max_tokens; the seed used is returned in X-Audio-Seed so
any utterance can be replayed on the same commit, backend and machine
type, since float rounding differs across them); greedy decoding never
ends the utterance on this model and decodes to near-silence (true
greedy is temperature 0 plus repetition_penalty 1).
`--audio-max-tokens N` caps a request (default 1024; 7 tokens per 85
ms frame). `"stream": true` returns the same audio as a chunked body,
each piece sent once the 5 frames (425 ms) after it exist, so playback
can start early; a streamed WAV header has 0xFFFFFFFF sizes, and the
token count, stop token and timing headers are not sent.
`scripts/convert-snac.py --out
tests/models/snac_24khz_decoder.gguf` makes the SNAC GGUF from the
torch checkpoint (dev-time: needs torch and network; it checks its own
weight fold and pins its tested versions and output sha256).
Speech is still slower than real time on an M2 Pro (i#173): a 3.8 s
clip takes 67 s on the CPU and 13 to 15 s with `--backend vulkan`,
where a stream's first audio arrives after 2.8 s. The SNAC decode takes
0.5 s on threaded CPU kernels, so token generation is most of it (24
to 27 tokens/s on Vulkan, run to run; real time needs about 82). The
response's Server-Timing header gives the prefill, generate and decode
times.

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
~204 tok/s on llama-3.2-1b Q8_0 and ~88 tok/s on
deepseek-coder-v2-lite Q8_0, with same-input projections and a MoE
layer's shared experts folded into one dispatch. Prompts ingest in
chunks of up to 512 tokens (1024 for MoE models) as tiled GPU matmuls
(vulkan_forward_batch: each weight read once per chunk, routed experts
grouped per expert, token-exact to per-token prefill), and attention
runs as matmuls over the paged pool (S = K Q^T, causal softmax,
O = V^T P; a GQA group's query heads, or all MLA heads, share each K/V
tile; a decode token splits O over positions, reading each K/V row
once per head group). On the same M4 Max prefill
runs ~1940 tok/s on a 2.5k-token prompt (llama-3.2-1b) and ~520
tok/s on a 3.1k-token prompt (deepseek-coder-v2-lite), up from 66
and 12 tok/s token by token, and decode after those prompts runs
~159 and ~66 tok/s (was 37 and 5). Every selectable backend must
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
