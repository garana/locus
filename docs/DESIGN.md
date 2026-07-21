# locus MVP Design

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

locus targets that intersection: llama.cpp's footprint with a
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
- `BlockAllocator` (implemented, `src/locus/kv/`) owns the free
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
chosen entry. Selection order: --backend flag > LOCUS_BACKEND env
var > best available. `--backends` prints the registry. Every
variant must match the scalar reference in tests, and the golden
e2e runs through whatever backend is the machine's default.

Vectorized CPU kernels follow the scheme proven in ../pbw: each
SIMD variant (neon, sse4, avx2, avx512) lives in its own source
file compiled with only that variant's flags, gated at configure
time by check_cxx_compiler_flag; `locus::sys::detect()` probes
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
variable (EngineLoop). Chat messages are formatted by
chat::ChatTemplate (R2): the GGUF Jinja source is fingerprinted
to a known family (llama3, chatml, llama2, zephyr, deepseek) and
rendered by a hardcoded formatter; unknown templates fall back to
plain concatenation.

## 5. Milestones

| Milestone | Deliverable                                  | Exit test                                   |
|-----------|----------------------------------------------|---------------------------------------------|
| M0        | Repo, CMake, Catch2, BlockAllocator          | unit tests green                            |
| M1        | GGUF loader + tokenizer                      | loads TinyLlama GGUF; fuzz corpus passes    |
| M2        | CPU forward pass, greedy decode, 1 sequence  | token-exact vs llama.cpp golden output      |
| M3        | PagedKvCache + scheduler + batch executor    | N concurrent streams, correct + no leaks    |
| M4        | HTTP server, SSE streaming                   | OpenAI client works against it              |
| M5        | Vulkan backend with paged attention          | M2/M3 tests pass on GPU, faster than CPU    |

M5 exit status (2026-07-18): golden and engine tests pass on the
vulkan backend (full GPU forward incl. paged attention over the
GPU-mapped pool); "faster than CPU" holds at real-model matrix
sizes (2048^2 matvec ~4.7x vs scalar), while the 260K test model
stays CPU-bound by per-token dispatch overhead.

## 6. Dependency policy

- Runtime deps: libc, libc++, OS APIs. Nothing else by default.
- Vendored (in-tree, pinned, sha256 recorded in the commit that
  imports them, license files alongside): Catch2 v3.7.1 (BSL-1.0,
  tests only), cpp-httplib v0.18.3 (MIT), nlohmann/json v3.11.3
  (MIT).
- No FetchContent/network at configure or build time.
- Optional system deps behind CMake flags: BLAS/Accelerate, Vulkan.

## 7. Post-MVP roadmap: DeepSeek-family and Kimi K3

Goal: run DeepSeek V3/R1-class models (which also covers Kimi K2,
same "deepseek2" GGUF architecture) and eventually Kimi K3. Phases
are ordered so each lands independently testable on hardware we
have; the big-model phases are correctness-first via small
same-architecture models (e.g. DeepSeek-V2-Lite, 16B total / 2.4B
active) before any large-scale run.

| Phase | Deliverable                       | Unblocks                    |
|-------|-----------------------------------|-----------------------------|
| R1    | Byte-level BPE tokenizer (done    | all non-SPM models          |
|       | 2026-07-18, incl. llama3 rope     |                             |
|       | scaling via rope_freqs.weight)    |                             |
| R2    | Chat templates (done 2026-07-18:  | instruct/chat checkpoints   |
|       | fingerprinted families, no Jinja) |                             |
| R3    | K-quants (Q4_K/Q5_K/Q6_K) CPU +   | most published GGUFs of     |
|       | Vulkan kernels (done 2026-07-19)  | large models                |
| R4    | MoE FFN: top-k router, 3-D expert | Mixtral-style llama-arch    |
|       | tensors, per-expert matvec (done  | MoE; DeepSeek gating        |
|       | 2026-07-19: softmax gating,       | (sigmoid+bias, shared       |
|       | Mixtral-style; GPU MoE dispatch   | experts) moves to R5 with   |
|       | added 2026-07-20)                 | the deepseek2 arch          |
| R5    | MLA + deepseek2 arch (done        | DeepSeek-V2 verified        |
|       | 2026-07-19: absorbed latent-KV    | token-exact vs llama.cpp;   |
|       | cache, interleaved YARN rope,     | V3/R1/K2 still need         |
|       | DeepSeek gating, shared experts,  | q_lora_rank + sigmoid       |
|       | Q5_0; full GPU MLA+MoE dispatch   | gating at scale             |
|       | added 2026-07-20, 3x NEON on      |                             |
|       | V2-Lite)                          |                             |
| R6    | Architecture registry (done       | new arch = one ArchSpec     |
|       | 2026-07-20: ArchSpec hooks for    | entry; --archs lists them   |
|       | hparams/tensors/attention/kv_dim) |                             |
| R7    | Kimi K3: Kimi Delta Attention     | K3 once weights + GGUF      |
|       | (per-layer recurrent state        | support exist (due          |
|       | alongside paged KV), MXFP4        | 2026-07-27); arch id not    |
|       | blocks, AttnRes                   | yet upstreamed              |
| R8    | Scale-out ergonomics: expert      | practical big-MoE hosting   |
|       | weights served straight from the  | on big-RAM hosts            |
|       | read-only mmap (OS page cache as  |                             |
|       | the working set), multi-token     |                             |
|       | GPU batching for prefill          |                             |

Notes:
- MoE fits the existing design well: the scheduler, paged KV, and
  server layers are unchanged; the work concentrates in the
  loader (3-D tensors, new quant types), the FFN block, and the
  backend op tables (per-expert matvec).
- MLA *shrinks* the KV cache (latents instead of full K/V), which
  compounds with paged allocation; KDA layers instead carry fixed
  -size recurrent state per sequence -- a new cache kind next to
  the paged pool, owned by the same allocator story.
- Hardware reality: K3-class (2.8T, ~1.5 TB MXFP4) stays
  cluster/big-RAM territory regardless of engine quality. The
  roadmap therefore optimizes for correctness on small
  same-architecture models locally, with mmap + page-cache
  residency as the path to "runs on a 2 TB RAM host".
- R8 starts with measurement, not code: on the first real
  bigger-than-RAM model we can load (GLM-5.2 target), record the
  passive-streaming baseline -- peak RSS vs model size, page
  faults, cold/warm tokens/sec, and unique experts touched per
  token window (the true working set). Deliberate residency and
  prefetch policies are only justified by beating that baseline;
  the same numbers gate the Vulkan/CUDA paging designs.
- Baseline, first cut (2026-07-21, GLM-5.2 UD-IQ1_S 216 GB,
  neon, cold cache, 8 forwards): 509 s wall (~60 s/token), peak
  RSS 15.4 GB (~7% of the model; only ~1 GB dirty -- the rest is
  evictable page cache), 8.7 M page faults. Expert telemetry via
  LOCUS_MOE_STATS (locus-run prints it at exit); Moonlight
  reference point: 11 forwards touch 43.8% of layer-expert
  slots, so routing spreads fast -- prefetch wins, if any, come
  from *within-token* expert readahead (overlap SSD reads across
  the 8 routed experts per layer), not from cross-token reuse.
- First policy result (2026-07-21): LOCUS_EXPERT_READAHEAD
  (madvise WILLNEED on all routed experts at selection time)
  measures 12% faster wall (524.4s -> 461.7s over 8 cold
  forwards) at equal fault counts (8.74M both legs -- proof both
  ran equally cold) and -23% sys time. Output identical. Next
  levers: readahead one layer AHEAD (route layer l while l-1
  computes is not possible -- routing depends on l's input -- but
  shared-expert and dense tensors of l+1 are known statically);
  and larger overlap via aggregating madvise calls per layer.
- Multimodal (K3 vision) is explicitly out of scope until the
  text path is proven.

## 8. Testing strategy

- Catch2 unit tests per component (allocator, scheduler invariants,
  GGUF parser, tokenizer round-trips).
- Golden-token integration tests: fixed prompt + greedy sampling
  must reproduce llama.cpp output token-for-token per model.
- Malformed-GGUF corpus for the loader (truncations, overflowing
  offsets, hostile metadata).
- Scheduler property tests: no block leaked, no double free, ref
  counts return to zero after every simulated workload.
