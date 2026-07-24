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
- Second policy result (2026-07-21): LOCUS_LAYER_READAHEAD
  (hint layer l+1's static weights at the start of layer l)
  stacked on expert readahead measures 351.4s over the same 8
  cold forwards -- 33% under the 524.4s passive baseline and
  24% under expert-readahead alone (461.7s), at equal fault
  counts (8.74M; equally cold) with sys time down 60.9s ->
  22.9s. RSS rises 9.2GB -> 13.2GB (~6% of the model) as
  readahead keeps more of the window resident. ~44s/token.
  Both policies are DEFAULT-ON since then (the flags
  LOCUS_EXPERT_READAHEAD / LOCUS_LAYER_READAHEAD are gone);
  LOCUS_NO_READAHEAD=1 restores passive streaming for A/Bs.
- Future work (2026-07-21): overlap I/O with compute one step
  ahead. Mechanism notes: each shard is mmap'd once, whole-file
  (PROT_READ MAP_PRIVATE, never written) -- weight pages are
  clean file-backed, so eviction drops them and re-reads from
  the GGUF; they never hit swap. WILLNEED is already async
  kernel readahead, so a thread touching the SAME pages the
  hints cover would only duplicate it (Gonzalo's observation).
  Plan, in order: (1) hints-first -- at the start of layer l,
  WILLNEED layer l+1's statically-known tensors (attn, dense,
  shared expert), which today get no hint at all; (2) A/B it;
  (3) PARKED (Gonzalo, 2026-07-21): a forcing prefetch thread
  that touches pages. Revisit only if, after wider hints, large
  sys time remains as evidence the kernel drops or caps
  advisory readahead.
- GLM-5.2 correctness + telemetry (2026-07-21): bounded
  CPU-only llama-completion (-ngl 0 --no-warmup -c 512; Metal
  full-offload wired 216GB and crashed the host -- never again)
  produces "Once upon a time, there was a" -- token-exact with
  locus at 4 greedy tokens. LOCUS_MOE_STATS over the same 8
  forwards: 4256 activations, 29-52 unique experts per MoE
  layer, 14.7% of the 76x256 layer-expert slots touched.
  Moonlight hit 43.8% in 11 forwards; at 256 experts the
  working set disperses slower in relative terms but the
  absolute bytes stay huge -- confirms within-token prefetch
  over cross-token caching at GLM scale too.
- GLM-5.2 re-anchored after the nextn/blk.78 exclusion
  (2026-07-23, on vx): locus (sse4) and a CPU-only llama.cpp
  (llama-simple, GGML_CUDA=OFF, -ngl 0, build b1-e8e6c7a)
  agree BYTE-EXACT on 16 greedy tokens for "Once upon a time":
  "Once upon a time, there was a little girl named Lily who
  loved to explore the world around her". Confirms the MTP
  block exclusion is correct past the old 4-token anchor.
  256-expert telemetry over 16 forwards: 11400 activations,
  52-108 unique experts per MoE layer, 27.4% of the 75x256
  slots touched. Cold llama.cpp on the vx RAID: ~124 s/tok
  (67min load + 33min decode), 171k major faults, 28.5GB RSS;
  locus's 40min run was warm-cache (ran second) so not a fair
  cold perf comparison -- the ~124 s/tok cold reference is the
  real "GLM on spinning disk" datapoint.
- Cold(-ish) locus (2026-07-23, GLM shards fadvise-evicted on
  vx): 50.7 min total for 16 tok (~190 s/tok all-in; lazy mmap,
  no separate load step), 82.5 GB read, 28.3 GB reclaimable RSS.
  So locus is faster END-TO-END than the cold llama.cpp
  reference (50.7 vs 92 min) because it streams lazily instead
  of front-loading the whole model. Methodology caveat to
  remember: locus major-fault counts are a FALSE coldness signal
  -- R8 readahead faults pages in ahead of access, so they never
  count as major faults (55.8k cold vs 54.6k warm, ~equal); use
  bytes-read (82.5 GB cold vs 70 GB warm) to judge coldness. The
  16-tok working set (27.4% of slots) exceeds the 28 GB page
  cache, so GLM is disk-bound regardless of warmth on that RAID.
- Memory bounding (2026-07-22, after llama.cpp's Metal
  full-offload crashed the 32GB host): locus's dirty memory is
  the KV pool + workspaces; weights are clean file-backed pages
  the kernel can always evict. Two knobs tighten it further:
  --ctx N caps the KV pool in tokens from the CLI, and
  LOCUS_WEIGHT_WINDOW=1 madvises DONTNEED (inward-aligned) on
  each layer's routed experts right after use so streamed
  models never build up page-cache pressure -- gated on the
  weights being file-backed, because DONTNEED on an anonymous
  in-memory image would discard the contents. Statics, shared
  experts and pinned weights are never dropped.
- Multimodal (K3 vision) is explicitly out of scope until the
  text path is proven.

### R8-GPU: weight paging (Vulkan; mirrored on CUDA)

Problem. The GPU backends have no middle ground today: Vulkan
uploads every weight once into an unbounded resident map
(matvec_vulkan State::resident, keyed by host pointer, never
evicted), so a model must fit in device memory; CUDA re-uploads
per matvec call, so nothing is ever reused. Neither serves
bigger-than-VRAM models.

Design: a weight pager per GPU backend.

- Page unit: one whole tensor for static weights; one expert
  slice (ExpertMat::expert_bytes) for routed-expert tensors, so
  paging granularity matches routing granularity.
- Budget: fixed device-byte budget. Knob LOCUS_GPU_POOL_MB;
  default queries the device-local heap and takes ~80% after
  the KV pool and activation buffers. The budget counts bytes
  as uploaded, which differs per backend (agreed with
  claude-vx-locus 2026-07-21): both CUDA and Vulkan keep
  weights QUANTIZED on-device for types their kernels dequant
  in-shader (Q4_K is ~0.5 B/weight, so ~8x more model fits in
  2GB VRAM); paths that dequantize at upload count F32 size
  (fused wkv_b on both backends; on Vulkan also any type
  without a matvec shader).
- Page table: host pointer -> {buffer, last_use, pin count}.
  Pages referenced by the command batch being recorded are
  pinned; unpinned pages evict LRU when an upload would exceed
  the budget. Eviction frees the device buffer only -- the
  source of truth stays the host mmap, exactly like the kernel
  page cache story on the CPU side.
- Prefetch: same schedule that won 33% on the CPU. While layer
  l's compute is recorded/submitted, upload layer l+1's static
  tensors; at MoE selection time (routing is already CPU-side
  per layer) upload the routed experts before their matmuls.
  Vulkan uses a dedicated transfer queue when the device has
  one (semaphore into the compute submit), else copies batch
  ahead of the compute dispatches on the same queue. CUDA
  mirrors with cudaMemcpyAsync on a second stream + events.
  LOCUS_NO_READAHEAD=1 disables prefetch (demand-only paging),
  keeping one opt-out for the whole R8 family.
- Pipeline composition: the CPU-side WILLNEED hints stay on --
  SSD -> page cache (kernel readahead) -> VRAM (pager) overlap
  the same one-step-ahead schedule end to end.
- Telemetry: pool hit rate, bytes uploaded/token, evictions,
  printed with the LOCUS_MOE_STATS report.

Exit test: deepseek-v2-lite Q4_K on vulkan with
LOCUS_GPU_POOL_MB set below the model's weight bytes must
reproduce the llama.cpp golden token-exact with evictions > 0
in telemetry -- paging proven, not mere residency. The CUDA
mirror validates the same way on vx (2GB VRAM makes the
sub-model budget the natural case). Full-GPU GLM additionally
needs IQ matvec shaders; that stays parked behind this work.

Prefetch hook contract (agreed with claude-vx-locus
2026-07-21): backend::Ops grows an optional
`void (*prefetch)(const Mat& w)` -- the backend MAY begin an
async host-to-device upload of w so a later matvec(w) finds it
resident; fire-and-forget; nullptr on backends without a pager
(CPU always; Vulkan/CUDA until theirs lands). The model calls
it, gated by the same readahead-enabled condition as the
madvise hints and mirroring those sites 1:1: forward() fires
it for layer l+1's static Mats next to advise_layer_statics,
moe_ffn() for the three expert(e) slices next to
advise_willneed. Invariant: the Mat passed to prefetch is
IDENTICAL (same host .data) to the one later passed to matvec
-- that pointer is the pager's page-table key.

Rollout order: (1) pager with demand paging only, default
budget = unbounded to preserve today's behavior; (2) LRU +
pinning under an explicit budget; (3) the prefetch schedule;
(4) IQ shaders. Phases 1-2 for CUDA are backend-internal and
started on vx; the Ops::prefetch field + llama.cpp call sites
land together once the registry side exists.

Vulkan pager status (2026-07-24): phases 1-2 landed
(matvec_vulkan WeightPool, header src/locus/backend/vulkan/
weight_pool.hpp). Host-pointer-keyed device buffers, byte
budget via LOCUS_GPU_POOL_MB (default unbounded = identical to
the old resident map), LRU eviction, buffers pinned for the
open command batch and unpinned at end_batch, transient
(uncached) fallback for a weight too big or when all resident
buffers are pinned, [vulkan-pool] telemetry. The eviction core
is unit-tested GPU-free via an injected buffer factory
(test_vulkan_pool.cpp). One structural limit: Vulkan records a
whole command batch per (sub-)token and end_batch submits+waits,
so eviction only frees buffers between batches -- a single
batch's working set must still fit device memory. A model whose
per-batch working set exceeds VRAM needs finer batch splitting
(future work); this is why deepseek-v2-lite on MoltenVK/M2
(~10GB uploaded in one batch) fails with "end cmd" both with
and without the pager -- a device limit, not a pager bug.
Validation host: the Raspberry Pi 5 (VideoCore VII Vulkan,
small VRAM) forces real cross-batch eviction on modest models
and is where the real-model exit test runs; heavy runs stay off
the 32GB MacBook.

### R9: threaded execution and static pinning (CPU)

Motivation (measured, GLM-5.2 on the 32GB m2). With readahead
default-on the profile is 78% single-core compute, ~15% I/O
stall: we are compute-bound on ONE core. llama-completion on
the same machine/model does 14.3 s/token CPU-only by using all
cores. Faults say ~18GB streams per token: ~8.4GB routed
experts (irreducible at temp-0 decode) + ~13GB statics that
are identical every token. Ladder: threading -> ~15 s/token
(then I/O-bound); + pinned statics and deeper overlap ->
~4-6 s/token; MTP on top approaches the ~2-3 s/token disk
floor.

- R9.1 threaded matvec. Model-level row split: a heavy matvec
  becomes T slices via mat_rows(w, r0, n) dispatched to a
  persistent locus::sys::ThreadPool, each slice through the
  ACTIVE backend's op.matvec into a disjoint out span. Each
  row's dot is computed by the same kernel code as today, so
  logits are bit-identical for every thread count. Knob
  LOCUS_THREADS (default: hardware cores, 1 disables). Applies
  to the CPU forward's large matvecs (attention projections,
  dense FFN, expert FFN, output head); matvec_t (small latent
  ops) and the vulkan full forward are untouched. Parallel
  slices also mean parallel page faults -- deeper NVMe queue,
  which is the second half of the win on streamed models.
- R9.2 static pinning. LOCUS_PIN_STATIC=1 mlocks every
  non-expert weight (attention, norms via their mats, dense
  FFN, shared experts, router, embeddings, output head) at
  load. Best-effort like advise_willneed: on failure (rlimit)
  it stays a plain fault-in. Opt-in because wiring ~13GB is a
  policy decision on a 32GB machine, not a default.
- R9.3 (parked): MTP decode via the nextn tensors GLM/DeepSeek
  ship -- each accepted speculative token amortizes the per-
  token weight stream. Revisit after R9.1/R9.2 measurements.

Exit test: logits bit-identical across LOCUS_THREADS=1/2/8 on
the synthetic models and the real-model goldens (validated on
vx); GLM-5.2 wall time at or below llama.cpp's measured 14.3
s/token on the same machine.

Status (2026-07-22): R9.1/R9.2 landed with the DSA indexer
(commit 0ffd1df); vx validated the full suite green plus cuda
under LOCUS_THREADS=4 token-exact -- concurrent matvec slices
compose with the weight pager (95% hit rate with prefetch, up
from 92.3% demand-only; 22% vs 0% under a thrashing 512MB
budget). R8-GPU phase 3 (op.prefetch model wiring + cuda
second-stream pipeline) is complete end-to-end. The GLM-5.2
wall-time measurement on the m2 is still pending (heavy run).

### DSA indexer: GLM-5.2 beyond top_k tokens

Today glm-dsa runs dense-equivalent attention with n_ctx
capped at attention.indexer.top_k (2048); the model's real
context_length is 1M. The lightning indexer (DeepSeek-V3.2
reference implementation, inference/model.py -- GLM-5.2 ships
the same DSA) selects which cached tokens the MLA attention
sees:

- Per layer, per token: k_idx(s) = LayerNorm_{w,b}(
  W_k x_s) in R^128, cached; q_idx(t) = W_qb q_a(t) viewed as
  32 heads x 128; head weights w(t) = W_proj x_t in R^32.
- Rope: applied to the FIRST rope.dimension_count (64) dims of
  BOTH q_idx and k_idx, HALF-SPLIT (non-interleaved) -- the
  reference is explicit that the indexer differs from the main
  attention's interleaved rope here.
- Score(t,s) = sum_h w_h(t) * ReLU(q_h(t) . k_idx(s)). Global
  scale factors (n_heads^-0.5, softmax scale) do not change
  the top-k ORDER, so selection ignores them.
- Selection: top min(top_k, positions) scores; attention and
  its softmax then run over the selected positions only. For
  sequences <= top_k every position is selected, so the
  existing dense-equivalent golden anchors correctness of the
  whole path below 2048 tokens bit-exactly.
- Storage: glm-dsa's cache row grows kv_dim 576 -> 704 (576
  MLA latent+rope || 128 post-norm post-rope k_idx).
- The n_ctx cap lifts for glm-dsa once the indexer is active
  (KV pool still defaults to min(n_ctx, 4096) tokens).

Validation limits, recorded honestly: llama.cpp IGNORES the
indexer tensors (dense full-context fallback), so there is no
external golden beyond top_k. Long-context real-model runs are
impractical on current hardware (>2048 sequential forwards at
tens of s/token). So: synthetic-model unit tests assert (a)
short sequences reproduce the dense path bit-exactly, (b)
beyond top_k exactly k positions are attended and outputs stay
finite, (c) selection picks the positions a scalar reference
scores highest. Real-model long-context quality is future work
on faster hardware.

## 8. Testing strategy

- Catch2 unit tests per component (allocator, scheduler invariants,
  GGUF parser, tokenizer round-trips).
- Golden-token integration tests: fixed prompt + greedy sampling
  must reproduce llama.cpp output token-for-token per model.
- Malformed-GGUF corpus for the loader (truncations, overflowing
  offsets, hostile metadata).
- Scheduler property tests: no block leaked, no double free, ref
  counts return to zero after every simulated workload.
