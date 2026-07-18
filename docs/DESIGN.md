# cpp-llm MVP Design

Status: draft v1 (2026-07-17)

## 1. The gap

No pure C/C++ project today combines:

- vendor-neutral GPU support (not NVIDIA-locked),
- token/block-level paged KV cache,
- continuous (iteration-level) batching,
- a production serving front end,
- a minimal, auditable dependency surface.

llama.cpp has the portability and dependency discipline but not the
high-concurrency scheduler; TensorRT-LLM has the scheduler but is
NVIDIA-only with a heavy Python toolchain; LightLLM/vLLM/SGLang have
both but sit on a large Python package ecosystem (supply-chain risk).

cpp-llm targets that intersection: llama.cpp's footprint with a
LightLLM-class serving core.

## 2. MVP thesis

The differentiating components are the KV-cache manager, the
scheduler, and the batch executor -- not the kernels. Kernels are
commodity and swappable. Therefore the MVP:

1. Builds the serving core (paged KV, continuous batching) first,
   against a simple CPU backend where paged attention is trivial to
   implement correctly (a gather loop).
2. Treats the compute backend as a narrow interface so a Vulkan
   backend (vendor-neutral GPU) can be added without touching the
   scheduler.
3. Reuses the GGUF model format for ecosystem compatibility: every
   quantized model published for llama.cpp becomes loadable here.

Correctness first, single-node CPU throughput second, GPU third.

## 3. Non-goals (MVP)

- Training or fine-tuning.
- Multi-node serving / tensor parallelism across hosts.
- Non-Llama-family architectures (MoE, multimodal) before M5.
- Python bindings.
- Windows support (macOS + Linux only initially).

## 4. Architecture

    +------------------------------------------------------------+
    | HTTP server (OpenAI-compatible, SSE streaming)         M4  |
    +------------------------------------------------------------+
    | Scheduler: admission, continuous batching, preemption  M3  |
    +---------------------+--------------------------------------+
    | PagedKvCache        | BatchExecutor                     M3 |
    | block alloc,        | ragged batch assembly,               |
    | ref-count/COW       | one forward pass per iteration       |
    +---------------------+--------------------------------------+
    | Model runtime: Llama-family forward pass               M2  |
    +------------------------------------------------------------+
    | Backend interface: matmul, rmsnorm, rope, attention        |
    |   CpuBackend (M2)  ...  VulkanBackend (M5)                 |
    +------------------------------------------------------------+
    | GGUF loader (hardened) + tokenizer                     M1  |
    +------------------------------------------------------------+

### 4.1 PagedKvCache

- KV memory is a pool of fixed-size blocks (default 16 tokens per
  block, configurable). A sequence's cache is a list of block ids;
  blocks need not be contiguous.
- Blocks are ref-counted so sequences forked from a shared prefix
  (system prompts, beam candidates) share blocks copy-on-write.
- `BlockAllocator` (implemented, `src/cppllm/kv/`) owns the free
  list, ref counts, and utilization stats. It is pure bookkeeping:
  no tensor memory, so it is unit-testable in isolation and reused
  unchanged by every backend.

### 4.2 Scheduler (continuous batching)

- Iteration-level: after every forward pass, finished sequences
  leave the batch and queued sequences join it. No static batches.
- Admission control: a request is admitted when the allocator can
  cover its prompt blocks plus a configurable decode headroom.
- Preemption (when the pool runs dry): evict the newest sequence
  and recompute it later (MVP policy: recompute, not swap-to-host).
- Fairness: FCFS with a cap on per-iteration prefill tokens so long
  prompts cannot starve decoding sequences (chunked prefill).

### 4.3 Backend interface

Narrow, tensor-op level API (about 8 entry points): embedding
lookup, matmul (weights possibly quantized), rmsnorm, rope, paged
attention (takes a block table, not a contiguous cache), silu/mul,
softmax, sampling helpers. CPU implementation uses plain C++ with
optional Accelerate/BLAS; correctness reference for all later
backends. Vulkan is the chosen GPU path because it is the only
vendor-neutral option that ships on macOS (MoltenVK), Linux/AMD,
and Linux/NVIDIA without per-vendor toolchains.

Backends are runtime-selectable: a registry (backend/registry.cpp)
lists every compiled variant with availability on the running
machine; the model routes heavy ops (matvec, dequant) through the
chosen entry. Selection order: --backend flag > CPPLLM_BACKEND env
var > best available. `--backends` prints the registry. Every
variant must match the scalar reference in tests, and the golden
e2e runs through whatever backend is the machine's default.

Vectorized CPU kernels follow the scheme proven in ../pbw: each
SIMD variant (neon, sse4, avx2, avx512) lives in its own source
file compiled with only that variant's flags, gated at configure
time by check_cxx_compiler_flag; `cppllm::sys::detect()` probes
the running machine and the backend picks the best variant at
runtime. This keeps one binary portable across CPU generations.
Build-time detection for both SIMD and Vulkan (find_package +
loader link, later glslc-to-embedded-SPIR-V as in pbw) is wired
up since M0; the kernels themselves land in M2 (CPU) and M5
(Vulkan).

### 4.4 GGUF loader

Parsing untrusted model files is an attack surface (llama.cpp has
had parser CVEs). Rules: bounds-check every offset against file
size, no allocation sized by unvalidated fields, fuzz corpus in
tests from day one, mmap read-only.

### 4.5 HTTP server

OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
with SSE streaming, plus `/health`. Decision (M4): vendored
cpp-httplib (MIT) for HTTP and nlohmann/json (MIT) for parsing
untrusted request bodies -- hand-rolled JSON parsing is exactly
where API vulnerabilities come from. A worker thread drives
Engine::step(); handler threads submit and wait on a condition
variable (EngineLoop). Chat messages are flattened to plain text
until chat templates land (post-MVP).

## 5. Milestones

| Milestone | Deliverable                                  | Exit test                                   |
|-----------|----------------------------------------------|---------------------------------------------|
| M0        | Repo, CMake, Catch2, BlockAllocator          | unit tests green                            |
| M1        | GGUF loader + tokenizer                      | loads TinyLlama GGUF; fuzz corpus passes    |
| M2        | CPU forward pass, greedy decode, 1 sequence  | token-exact vs llama.cpp golden output      |
| M3        | PagedKvCache + scheduler + batch executor    | N concurrent streams, correct + no leaks    |
| M4        | HTTP server, SSE streaming                   | OpenAI client works against it              |
| M5        | Vulkan backend with paged attention          | M2/M3 tests pass on GPU, faster than CPU    |

## 6. Dependency policy

- Runtime deps: libc, libc++, OS APIs. Nothing else by default.
- Vendored (in-tree, pinned, sha256 recorded in the commit that
  imports them, license files alongside): Catch2 v3.7.1 (BSL-1.0,
  tests only), cpp-httplib v0.18.3 (MIT), nlohmann/json v3.11.3
  (MIT).
- No FetchContent/network at configure or build time.
- Optional system deps behind CMake flags: BLAS/Accelerate, Vulkan.

## 7. Testing strategy

- Catch2 unit tests per component (allocator, scheduler invariants,
  GGUF parser, tokenizer round-trips).
- Golden-token integration tests: fixed prompt + greedy sampling
  must reproduce llama.cpp output token-for-token per model.
- Malformed-GGUF corpus for the loader (truncations, overflowing
  offsets, hostile metadata).
- Scheduler property tests: no block leaked, no double free, ref
  counts return to zero after every simulated workload.
