# locus

A pure C/C++ LLM inference server aiming at the unfilled niche
between llama.cpp and the Python serving stacks (LightLLM, vLLM):

- vendor-neutral (CPU first, Vulkan for GPU later),
- token-block paged KV cache with copy-on-write prefix sharing,
- continuous (iteration-level) batching,
- OpenAI-compatible HTTP API,
- minimal, auditable, vendored dependency surface.

See [docs/DESIGN.md](docs/DESIGN.md) for the full MVP design and
milestone plan.

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
32GB machine -- see the R8 notes in DESIGN.md).

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
vectorized, the rest scalar. cuda is likewise hybrid: F32/Q8_0
matvec on an NVIDIA GPU (streamed per call for now; other types
scalar), validated on sm_50 (GTX 750 Ti, CUDA 12.x). vulkan is the
full GPU forward pass instead: F32/Q8_0 matmul shaders (weights
resident, uploaded once), rmsnorm, RoPE, SwiGLU, and paged
attention reading K/V from a GPU-mapped cache pool, all recorded
as one command batch per token. Every selectable backend must
reproduce the llama.cpp golden output token-exact (tested, single
and concurrent streams). At real-model sizes the GPU wins (2048^2
f32 matvec: ~750us vs ~3550us scalar CPU, Apple M-series via
MoltenVK); on the tiny 260K test model dispatch overhead keeps
CPU ahead. F16/Q4_0 GPU shaders, x86 avx2/avx512 (avx2 kernel
exists, pending validation on an AVX2 host), and CUDA residency/
streaming policies (R8) slot into src/locus/backend/ as they land.

## Dependencies

No external packages at runtime. Vendored (pinned, checksummed,
with license files; see docs/DESIGN.md section 6): Catch2 (tests
only), cpp-httplib, nlohmann/json. Optional, auto-detected at
configure time:

- Vulkan loader (GPU backend, milestone M5)
- SIMD compiler flags per arch (neon/sse4/avx2/avx512); kernel
  variants are selected at runtime via `locus::sys::detect()`
