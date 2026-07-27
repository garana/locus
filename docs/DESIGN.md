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
sub-model budget the natural case).

Full-GPU GLM Vulkan quant coverage (done): GLM-5.2 uses four quant
types Vulkan previously fell back to host scalar on -- Q2_K (5c1a6e2)
and IQ2_XXS/IQ3_XXS/IQ4_XS (abe0b1a, 830966c). Each ports its
cpu_ops dequant_block_* into a GLSL matvec (dequant fused into the
dot), validated on the M2 (MoltenVK) vs the scalar reference in
[vulkan]. The IQ2/IQ3 grid tables reach the GPU the way CUDA does
(device_iq1s_grid): a persistent SSBO of grid + ksigns_iq2xs bound
as a 4th input for those kernels (the pipe table already carries a
per-kernel binding count); IQ4_XS is grid-free (kvalues_iq4nl
embedded as a GLSL const). So every GLM quant type now has a Vulkan
matvec -- no host-scalar fallback. The CUDA twin (Q2_K + IQ2/IQ3/IQ4
device kernels) is the remaining gap for full-GPU GLM on CUDA.

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

Exit test PASSED (2026-07-24, Raspberry Pi 5): llama-3.2-1b
Q4_K_M on --backend vulkan produces byte-identical output to the
scalar and unbounded-vulkan runs ("Once upon a time, in a small
village nestled in the", 8 greedy tokens) under
LOCUS_GPU_POOL_MB=64. Telemetry: budget 64MB, resident 63MB,
hits 378, misses 1566, evictions 341, uploaded 8483MB (the
0.8GB model re-streamed ~10x under the sub-working-set budget).
So eviction fires and stays token-exact -- paging proven, not
mere residency. (The Pi's tile GPU is slower than its CPU here,
so the "GPU faster than CPU" perf assertion is a reported WARN,
not a gate.)

Prefetch (phase 3) DROPPED for Vulkan (2026-07-24, measured on
V3D by profiling matvec_vulkan). On llama-3.2-1b, compute
(end_batch submit+wait over the matvec shaders) is a fixed
~66.8s for 16 tokens whether the pool is unbounded or paging
hard; upload (a plain memcpy into HOST_COHERENT unified memory,
no DMA engine) is 0.28% of compute unbounded and only 4.5% even
under pathological paging (589 evictions, 14GB re-uploaded). So
prefetch's absolute ceiling is <5%, and the overlap-with-compute
fraction on a synchronous end_batch is realistically <2% -- not
worth a second queue or a warm() pool variant. The measure-first
plan paid off: the numbers killed it before any code landed.

The real Vulkan MoE inefficiency (found in the same pass, now
the next item): the MoE path uploads ALL hp.n_expert experts per
layer per token via whole(), though only expert_used_count are
routed (64 vs 6 on deepseek-v2-lite) -- ~10x wasted memcpy and
residency. Fix: pool.acquire per PICKED expert (each
expert(e).data is a distinct key, matching the pager's
per-expert granularity), remapping the shader w_off from
e*expert_bytes to the compact per-expert buffer. Likely the
cause of, and fix for, the Pi's 10GB-MoE Vulkan crash: on the
4GB Pi, deepseek-v2-lite on --backend vulkan SIGSEGVs
(pool=2048MB) / OOM-kills (pool=256MB) because the whole-expert
upload plus F32 dequant temporaries exceed RAM on top of the
mmap -- it crashes rather than degrading. A graceful
device-alloc guard (clean throw instead of SIGSEGV) is a
secondary follow-up.

deepseek-on-Vulkan root cause (2026-07-24, gdb on the Pi by
claude-pi-locus -- NOT memory, as first assumed): the model has
mixed expert quant (ffn_down_exps is Q5_0 in 14 of 26 MoE
layers), and Vulkan has no Q5_0 matvec shader, so
vulkan_forward() returns false at the first Q5_0 tensor and the
model falls back to the generic per-op forward. That path drives
ops.matvec through R9's matvec_mt (the thread pool), but the
Vulkan matvec drives a single VulkanContext singleton -> N
threads doing begin/dispatch/end_batch concurrently SIGSEGV in
V3DV. Two fixes: (a) matvec_mt now runs a backend inline when
Ops::mt_safe is false (vulkan) -- so ANY unshadered-type model
degrades instead of crashing (commit cc5dfc1); (b) add a Q5_0
Vulkan matvec shader so deepseek stays on the full-GPU path
(claude-pi-locus). With routed-expert upload for residency,
deepseek should then run on the 4GB Pi. (Q5_0 is the sole
unshadered type deepseek needs; Q5_1 is a companion but unused
here.) Operational note: do not run deepseek-on-Vulkan on the
4GB Pi until Q5_0 lands -- it thrashes hard enough to trip the
OOM killer on unrelated processes.

MILESTONE -- first full-GPU MoE+MLA on real hardware (2026-07-24,
Raspberry Pi 5 V3D, after Q5_0 f099051 + routed-expert 0daa990):
deepseek-v2-lite-q4_k_m on --backend vulkan is token-exact with
--backend scalar ("Once upon a time, there was a", 4 tok) and
FITS -- vulkan peak RSS 2011MB / min MemAvailable 1850MB on the
4GB box (a 10GB model on the GPU path; scalar peaked higher,
3114MB, since it dequants into host memory while the vulkan pool
stays leaner). Confirms the full-GPU MLA + DeepSeek-MoE path end
to end on a real model -- unvalidatable before (M2 MoltenVK
can't run it, vx has no Vulkan device). Routed-expert upload
verified staging ~6 experts/token not 64 ([vulkan-pool] @768MB:
uploaded 13.6GB, hits 0, evictions 5627 -- pure streaming,
per-token working set >> budget; that residency cut is what
moved peak RSS from OOM-kill to 2.0GB). PERFORMANCE reality on
V3D: vulkan 85s vs scalar/NEON 21s for prompt+4tok -- the GPU is
~4x SLOWER here, consistent with the prefetch profiling (V3D is
compute-bound and this run re-streams 13.6GB through the pager).
Takeaway: on the Pi, Vulkan is a CAPABILITY win (correct full-GPU
path, leaner RSS), not a speed win for bigger-than-pool models --
NEON CPU wins those; Vulkan pays off only for models that FIT the
pool (no re-streaming).

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
second-stream pipeline) is complete end-to-end.

GLM-5.2 wall-time exit test (2026-07-27, PASS, on vx -- 750 Ti /
31GB RAM / 7200rpm SATA HDD, GLM-5.2 UD-IQ1_S 216GB, 16 tokens,
cold, same box/model/prompt): llama.cpp (CPU) 5523 s vs locus
(sse4, streaming) 3042 s -- locus 1.82x faster and token-exact.
locus <= llama.cpp: PASS. This is the bigger-than-RAM streaming
thesis realized: a 216GB model on a 31GB box, resident set ~15GB,
served faster than llama.cpp. The full-GPU CUDA path also runs the
real model coherently (validates the Q2_K/IQ2/IQ3/IQ4 device
kernels beyond synthetic units), but on the 750 Ti's ~1.2GB usable
VRAM the pager thrashes (24.4% hit, 131k evictions, 240GB
re-uploaded over PCIe for 6 tokens), so CUDA is PCIe-bound there and
sse4 is the faster locus backend on that box. CUDA's streaming win
needs a larger-VRAM GPU -- the one place a cloud box helps here is
VRAM (bigger pager working set), not compute.

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

### R10: batched forward (prefill + continuous-batch decode)

The executor -- Engine::advance() -> model_.forward() -- runs ONE
token per call. So it processes N tokens as N full passes over
the weights, whether those N are prompt positions or N
concurrent sequences' decode tokens. locus already does
continuous batching at the SCHEDULER level (admits/tracks N
running sequences, shared paged-KV pool, no head-of-line
blocking) but NOT at the executor: it gets continuous batching's
memory/scheduling benefits without its compute/bandwidth
benefit. R10 closes that gap with a batched forward.

Where the batch of N comes from -- one mechanism, three
consumers:
1. PREFILL: N prompt positions of one sequence. Cuts prompt-
   ingestion I/O ~N x (largest for long prompts / RAG).
2. CONTINUOUS-BATCH DECODE (the throughput core, the LightLLM/
   vLLM niche this project targets): N concurrent requests each
   emit one decode token per step -> batch across sequences.
   Serving N users drops from N weight-passes/step to one;
   aggregate throughput scales ~N x, per-served-token weight I/O
   drops ~N x. This is what makes serving a bigger-than-RAM
   model to multiple users viable -- they share one weight
   stream per step.
3. SPECULATIVE / MTP (future, R9.3): draft K tokens via the
   nextn head, verify all K in one batched forward -- the batch
   win inside a single stream. Same machinery.
Single-stream, non-speculative decode cannot batch (auto-
regressive); that is the only case R10 does not help.

The win, three ways, all in the weight-bearing ops (projections,
dense/expert FFN):
- disk I/O: each weight read/faulted once, not N times
  (streaming models).
- DRAM bandwidth: matvec is memory-bound; a weight ROW is a few
  KB (fits L1), so reading it once and reusing it across the N
  tokens cuts weight DRAM traffic ~N x (in-RAM large models --
  prefill is bandwidth-bound today).
- dequant compute: OPTIONAL follow-on kernel, NOT part of the
  byte-identical core. A dedicated batched kernel can dequantize
  each weight block ONCE and reuse the F32 across the N tokens
  (the ~78%-single-core cost on GLM's IQ1_S is dequant). It
  matches the SCALAR path exactly but reorders sums vs the fused
  neon/sse4 kernels, so it rides the token-exact bar (like the
  SIMD kernels), not the byte-identical check. The disk-I/O and
  DRAM-bandwidth wins above do NOT need it.

Safety property of the byte-identical core: forward_batch does
EXACTLY the sequential per-token matvecs, only reordered so each
weight is used for all N tokens before moving to the next
weight. Each output y[row][token] = dot(W[row], x[token]) (via
the active backend's matvec) is unchanged regardless of loop
order -- weight-stationary, NOT a GEMM that reorders summation.
So forward_batch(t_1..t_N) must produce a KV cache and logits
BYTE-IDENTICAL to N sequential forward() calls -- the exit test.
(The dequant-once kernel trades this byte-identity for the
compute win and is validated token-exact instead.)

Shape:
- Primitive matvec_batch(W, x[0..N], out[0..N]): a weight-
  stationary loop calling the backend's matvec per token, so the
  weight stays hot in cache / page cache across the batch.
  Byte-identical to N matvec() calls. Carries the disk-I/O and
  cache-residency wins; dequant-amortization is the separate
  kernel noted above.
- Position-wise ops batch through it: wq/wk/wv/wo, dense/expert
  FFN, embedding; the output head runs only for tokens that need
  logits (the last prompt token in prefill; every token in
  decode).
- Attention uses no large weights, so it stays PER TOKEN: after
  the batched Q/K/V projection, each token does scores/softmax/
  weighted-sum against ITS OWN sequence's KV (prefill tokens
  share one causal KV; decode tokens each hit their sequence's
  cache). No amortization lost there.
- MoE routing is per token; the union of experts a layer's N
  tokens pick is faulted once and each applied to the tokens
  that chose it.
- Memory: neutral. Weights (mmap) and the KV pool are unchanged;
  only activations go N-wide (~7-14MB at N=64 across our models);
  logits stay 1-wide where only the last token is sampled. The
  attention scores buffer stays 1-wide (reused per token) -- a
  naive N-wide att would be N x n_ctx and is the one trap to
  avoid.

Rollout (status 2026-07-25): (1) matvec_batch, (2)
forward_batch, (3) all-arch coverage, and (4a) engine prefill
batching are DONE.
- (1)+(2) commit f954a65; (3) commit 42846d4 refactored
  forward_batch to reuse the shared per-token attention
  (spec_->attention) so llama / deepseek2 (MLA) / glm-dsa are
  all covered, dense FFN batched via matvec_batch, MoE per token.
  [batch] tests assert byte-identity to N forward() calls
  (logits + KV) on synthetic dense-llama, deepseek2-MLA and
  llama-MoE. supports_batch() is true for every CPU/CUDA backend
  (Vulkan opts out -- it has its own full forward).
- (4a) commit 20d560c: Engine::advance() ingests the remaining
  prompt as one forward_batch (Config::batched_prefill, default
  on); [engine] A/B test confirms identical output to per-token.
- (3b) commit 77cfc0e: moe_ffn_batch applies routed experts
  WEIGHT-STATIONARY (route all n, group (token,slot) by expert,
  read each expert once across the tokens that picked it; union
  read-ahead/prefetch/weight-window once). Byte-identical:
  each token's mixture is summed in moe_select order (routed
  then shared) before adding to the residual, matching moe_ffn.
  forward_batch's MoE layers now use it. [batch] llama-MoE test
  covers it.
- (4b, model) commit d76fa7f: forward_batch_decode decodes one
  token from each of N different sequences in a pass -- per-token
  attention against each sequence's own KV, weight-bearing ops
  batched across all N. Byte-identical to N forward() calls
  ([batch] test: 4 sequences, distinct contexts). This is the
  executor-level continuous-batching primitive.
- (4b, engine) commit 724ff13: step_batched() (Config::
  batched_decode, default OFF) runs prefill -> sample -> ONE
  forward_batch_decode for every running sequence with a pending
  token, with per-request logits and preemption confined to the
  not-yet-gathered tail. Byte-identical to the per-sequence
  scheduler: [engine] A/B test matches generated tokens across 4
  streams on a comfortable pool AND a tight pool (forces
  preemption + recompute in both paths).
So the whole batched-forward stack -- prefill + continuous-batch
decode, all archs, batched MoE experts -- is byte-identical and
in place. Optional step (5) is now landed too: the
dequant-amortization batched kernel (matvec_batch_deq)
dequantizes each weight row to f32 once, then dots it against all
n token columns -- cutting dequant work from once-per-(row,token)
to once-per-row. Because it dequants the whole row before the
dot rather than interleaving (as the fused SIMD matvec does), it
is token-exact and deterministic but NOT byte-identical, so it is
opt-in behind LOCUS_BATCH_DEQUANT and matvec_batch routes to it
only when that is set. Its own [batch] test proves f32 stays
byte-identical (dequant_row is a copy there) and q8_0 matches the
fused matvec within a 1e-5 margin.

(5b, threading) The first Pi measurement caught a real gap:
matvec_batch was single-threaded (a serial `for t in n:
op.matvec`), while the sequential forward's FFN uses the R9
threaded matvec_mt. So on a large model over multiple cores,
batched decode forfeited multicore and lost to the per-token path
(Pi llama-1b in-RAM: 3.2 -> 1.8 tok/s), even though it won on tiny
models where matvec_mt inlines (stories260K: ~2x on both Pi and
M2). Fix: matvec_batch (and matvec_batch_deq) now row-slice across
the ThreadPool via the shared mt_slices()/for_row_slices() helper,
exactly like matvec_mt (honors mt_safe and LOCUS_THREADS). Each
slice carries the whole n-token batch, so it keeps the weight-
stationary reuse AND regains multicore -- batched decode should be
>= per-token in every regime. Byte-identity holds (row-split is
bitwise-equal per row); the [batch] test now checks the threaded
fused path at THREADS=1/2/4 on a 200-row matrix.

(6, done) Measured on Pi (NEON) and vx (x86/sse4): with the
threading fix, batched decode is >= per-token in every regime --
Pi 1B in-RAM ties (4.5 vs 4.4 tok/s), tiny model wins (1.4-2x),
deepseek streaming cuts disk bytes; vx 1B ties (4.4 vs 4.3). So
batched_decode is now default-ON (Engine::Config), with step()
falling back to the per-token scheduler when the model does not
support batching (Vulkan drives its own forward) so the default is
safe on every backend. Prefill batching stays a clean I/O win on
the Pi (deepseek: 2.44x fewer bytes read, 1.74x faster).

Measurement also pinned the ceiling: decode is memory-bandwidth-
bound (vx callgrind: matvec 95% of instructions; 66% parallel
efficiency at 4 cores on DDR3), and the current fused matvec_batch
re-streams each weight row N times (per-token op.matvec loop), so
it saves no DRAM traffic on a large in-RAM model -- hence the tie,
not a win. That motivates R11.

## R11: cache-blocked batched matvec

The batched decode of N concurrent sequences is a skinny GEMM
Y = W . X^T (X is N x cols). Today's kernel loops op.matvec per
token, streaming |W| from DRAM N times. Reorder to weight-
stationary at the register level: outer loop over weight rows/col-
blocks, inner loop over the N tokens, so each weight block is
loaded+dequantized ONCE and reused across all N token dots while
hot in vector registers (N running SIMD accumulators). Weight DRAM
traffic drops ~N-fold; the kernel moves from memory-bound toward
compute-bound. Needs N >= 2 (single-stream decode has no reuse and
stays bandwidth-bound); the win scales with N until the N-vector X
tile spills L1, past which cols is cache-tiled. Byte-identity is
preservable: if the blocked SIMD kernel keeps the same per-col-
block accumulation order as op.matvec, each token's dot is
bitwise-unchanged (unlike matvec_batch_deq, which reordered sums).

Plan: add an optional matvec_batch hook to backend::Ops (nullptr
-> today's per-token fallback); wire model matvec_batch() to
prefer it. Scalar reference + byte-exact test here; per-arch SIMD
kernels for the hot quants (Q4_K/Q6_K): NEON owner claude-pi-locus,
sse4/avx2 owner claude-vx-locus. Cheap pre-validation before any
SIMD: re-measure the now-threaded matvec_batch_deq (row read once,
scalar dots) on the 1B -- if it ties/beats the fused per-token path
despite ~8x more compute, that confirms traffic is the wall.

Status (done for CPU Q4_K/Q6_K). Hook landed a04f269
(Ops::matvec_batch; output row-major out[r*n+t] so the model row-
slices it across threads then transposes back; scalar reference +
[ops][batch] byte-test). Pre-validation on vx confirmed the wall:
threaded matvec_batch_deq did ~8x more compute for only 16% slow-
down -- the fingerprint of a memory-bound loop. SIMD kernels then
landed: sse4 6c3ecdb, neon 2e6e15e, both Q4_K/Q6_K register-blocked
(read+dequant each block once, FMA into n per-token accumulators).
Byte-identity held by keeping matvec's exact per-block reduction
order (a single end-of-row reduction diverges -- float add is not
associative); [ops][batch][sse4]/[neon] byte-tests are bit-exact,
[moe][batch] e2e token-exact at THREADS=1/2/4.

Measured ratio (fused-fallback vs kernel, 1B Q4_K, --concurrent 8,
in-RAM warm):
  cuda / GTX 750 Ti   : 2.2 -> 3.4 tok/s  = 1.55x
  neon / LPDDR4X (Pi) : 4.4 -> 6.2 tok/s  = ~1.35x
  sse4 / DDR3   (vx)  : 4.5 -> 4.9 tok/s  = 1.09x
The more bandwidth-starved box gets the bigger traffic-amortization
win, as predicted; after the n-fold weight-traffic cut the loop
goes SIMD-compute-bound, so 128-bit sse4 caps vx's ratio (avx2 +
DDR5 would show more). The CUDA kernel (07736a4) wins most: one
launch replaces ~N*slices tiny per-token launches AND each weight
block is read from VRAM once for all n tokens ([cuda-pool] identical
both sides, so the gain is pure kernel). It uses batch_self_parallel
(the model hands it the whole matrix, one grid over all rows) plus
the weight-pool device pointer, so it stacks on the pager's shared-
upload win; a modern GPU raises the ratio further. A backend kernel
that special-cases only some types must fall back to n calls of the
SAME backend's matvec() (scattered row-major), never the scalar
reference -- else F32/Q8_0 diverge from that backend's per-token
path (contract note in registry.hpp). Target set is Q4_K/Q6_K (the
Q4_K_M tensors we run); Q5_K/Q5_0 register-blocking is parked until a
served model needs it. Remaining R11: avx2 only (parked until a
Haswell+ cloud box; sse4 is the template).

Risk note: forward_batch duplicates the forward + per-arch
attention shape, so it lands as NEW code validated against the
sequential path before the engine ever calls it -- the
6-model token-exact goldens keep guarding the default path
throughout.

## 8. Testing strategy

- Catch2 unit tests per component (allocator, scheduler invariants,
  GGUF parser, tokenizer round-trips).
- Golden-token integration tests: fixed prompt + greedy sampling
  must reproduce llama.cpp output token-for-token per model.
- Malformed-GGUF corpus for the loader (truncations, overflowing
  offsets, hostile metadata).
- Scheduler property tests: no block leaked, no double free, ref
  counts return to zero after every simulated workload.
