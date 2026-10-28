# locus design

Why locus is built the way it is, and the measurements behind the
decisions. README.md has the current state and the pending work. The
`R<n>` labels name design areas; code comments and `--help` refer to
them.

## 1. The gap

No pure C/C++ project combined:

- vendor-neutral GPU support (not NVIDIA-locked);
- a token-block paged KV cache;
- continuous (iteration-level) batching;
- a production serving front end;
- a minimal, auditable dependency surface.

llama.cpp has the portability and the dependency discipline, but not
a high-concurrency scheduler. TensorRT-LLM has the scheduler, but is
NVIDIA-only with a heavy Python toolchain. vLLM, SGLang and LightLLM
have both, but sit on a large Python package ecosystem (a supply-chain
risk).

locus targets that intersection: llama.cpp's footprint with a
LightLLM-class serving core. Its distinctive lever is running models
that do not fit in memory, by streaming weights and amortizing the
stream across a batch. AirLLM also streams layers, but one sequence at
a time with no batching, so it is I/O-bound (docs/COMPARISON.md).

## 2. Thesis

The differentiating parts are the KV-cache manager, the scheduler and
the batch executor, not the kernels. Kernels are commodity and
swappable. So:

1. The serving core (paged KV, continuous batching) came first, on a
   simple CPU backend where paged attention is a plain gather loop.
2. The compute backend is a narrow interface, so a GPU backend plugs
   in without touching the scheduler.
3. The model format is GGUF, so every quantized model published for
   llama.cpp loads here.

Correctness first, then CPU throughput, then GPU.

## 3. Non-goals

- Training or fine-tuning.
- Tensor parallelism across hosts (section 14 explains why pipeline
  and expert parallelism are the multi-host axes instead).
- Python bindings.
- Windows (macOS and Linux only).

## 4. Architecture

    +------------------------------------------------------------+
    | HTTP server: OpenAI + Anthropic APIs, SSE streaming        |
    +------------------------------------------------------------+
    | Scheduler: admission, continuous batching, preemption      |
    +---------------------+--------------------------------------+
    | PagedKvCache        | Batch executor                       |
    | block alloc,        | ragged batch assembly,               |
    | ref-count / COW     | one forward pass per iteration       |
    +---------------------+--------------------------------------+
    | Model runtime: per-architecture forward pass (ArchSpec)    |
    +------------------------------------------------------------+
    | Backend interface: matvec, dequant, batched matvec, ...    |
    |   neon / avx2 / sse4 / scalar (CPU)   vulkan   cuda        |
    +------------------------------------------------------------+
    | GGUF loader (hardened) + tokenizers + chat templates       |
    +------------------------------------------------------------+

### 4.1 PagedKvCache

- KV memory is a pool of fixed-size blocks (16 tokens by default). A
  sequence's cache is a list of block ids; blocks need not be
  contiguous.
- Blocks are ref-counted, so sequences forked from a shared prefix
  (system prompts) share blocks copy-on-write.
- `BlockAllocator` (`src/locus/kv/`) owns the free list, the ref
  counts and the usage stats. It holds no tensor memory, so it is
  unit-tested alone and shared by every backend.
- The pool can hold f32, f16, q8 or q4 rows (R14). On Vulkan the pool
  is GPU-mapped, so attention reads K and V where they are written.

### 4.2 Scheduler (continuous batching)

- Iteration-level: after every forward pass, finished sequences leave
  the batch and queued ones join. No static batches.
- Admission: a request is admitted when the allocator can cover its
  prompt blocks plus a decode headroom.
- Preemption, when the pool runs dry: evict the newest sequence and
  recompute it later (recompute, not swap to host).
- Fairness: first come, first served, with a cap on prefill tokens per
  iteration (chunked prefill), so long prompts do not starve decoding
  sequences.

### 4.3 Backend interface

A narrow, op-level table (`backend::Ops`): matvec, dequant, batched
matvec, the 8-bit activation matvec (R16), an optional prefetch hook
and a KV allocator. The scalar CPU backend is the correctness
reference for all others. Vulkan is the GPU path because it is the
only vendor-neutral option that ships on macOS (MoltenVK), on Linux
with AMD, and on Linux with NVIDIA, without a per-vendor toolchain.
CUDA is a second GPU path for NVIDIA hosts.

Backends are selected at runtime. A registry lists every compiled
variant and whether the machine supports it; `--backend` beats
`LOCUS_BACKEND`, which beats the best available (the first usable
entry, a CPU one). When that default leaves a usable GPU idle,
locus-run and locus-server print a warning naming the `--backend` to
pass. Every variant must
match the scalar reference in tests, and the golden end-to-end tests
run on every selectable backend.

SIMD kernels follow the scheme of the sibling `pbw` project: each
variant (neon, sse4, avx2) is its own source file, compiled with only
that variant's flags and gated at configure time;
`locus::sys::detect()` probes the running machine. One binary stays
portable across CPU generations. Vulkan shaders are compiled to
SPIR-V and embedded at build time.

### 4.4 GGUF loader

Parsing untrusted model files is an attack surface (llama.cpp has had
parser CVEs). Rules: bounds-check every offset against the file size;
no allocation sized by an unvalidated field; a fuzz corpus in the
tests; mmap read-only.

### 4.5 HTTP server

Vendored cpp-httplib for HTTP and nlohmann/json for untrusted request
bodies: hand-rolled JSON parsing is where API vulnerabilities come
from. A worker thread drives `Engine::step()`; handler threads submit
and wait (`EngineLoop`). Chat messages are formatted by
`chat::ChatTemplate`: the GGUF's Jinja source is fingerprinted to a
known family (llama3, chatml, llama2, zephyr, deepseek, deepseek2,
deepseek-coder) and
rendered by a hardcoded formatter; an unknown template falls back
to plain concatenation. Endpoints that run the model on their own
thread (images, vision chat, audio) share the GPU backend through one
process-wide lock (#268).

## 5. Milestones

The MVP milestones, each with its exit test:

| Milestone | Deliverable                                  | Exit test                                   |
|-----------|----------------------------------------------|---------------------------------------------|
| M0        | Repo, CMake, Catch2, BlockAllocator          | unit tests green                            |
| M1        | GGUF loader + tokenizer                      | loads a TinyLlama GGUF; fuzz corpus passes  |
| M2        | CPU forward pass, greedy decode, 1 sequence  | token-exact vs llama.cpp golden output      |
| M3        | PagedKvCache + scheduler + batch executor    | N concurrent streams, correct, no leaks     |
| M4        | HTTP server, SSE streaming                   | an OpenAI client works against it           |
| M5        | Vulkan backend with paged attention          | M2/M3 tests pass on the GPU                 |

The areas built after the MVP:

| Area | What                                                        | Section |
|------|-------------------------------------------------------------|---------|
| R1   | byte-level BPE tokenizer, llama3 rope scaling               |         |
| R2   | chat templates (fingerprinted families, no Jinja engine)    | 4.5     |
| R3   | K-quants on CPU and Vulkan                                  |         |
| R4   | MoE FFN: router, 3-D expert tensors, per-expert matvec      |         |
| R5   | MLA and the deepseek2 architecture                          |         |
| R6   | architecture registry (`ArchSpec`; `--archs`)               |         |
| R8   | weight streaming and GPU weight paging                      | 7, 8    |
| R9   | threaded matvec and static pinning                          | 9       |
| R10  | batched forward: prefill and continuous-batch decode        | 11      |
| R11  | cache-blocked batched matvec                                | 12      |
| R12  | sampling, JSON mode, prompt caching, speculative decoding   | 13      |
| R13  | more architectures and quant types                          | 13      |
| R14  | embeddings, KV-cache quantization                           | 13      |
| R15  | API polish; multi-server inference                          | 13, 14  |
| R16  | 8-bit activations (llama.cpp's Q8_K / Q8_0 dots)            | 15      |

## 6. Dependency policy

- Runtime dependencies: libc, libc++, OS APIs. Nothing else by
  default.
- Vendored in-tree, pinned, sha256 recorded in the importing commit,
  license files alongside: Catch2 v3.7.1 (BSL-1.0, tests only),
  cpp-httplib v0.18.3 (MIT), nlohmann/json v3.11.3 (MIT), and ggml's
  IQ lookup tables (`third_party/ggml/iq_grids.h`, MIT).
- No FetchContent and no network at configure or build time.
- Optional system dependencies, found at configure time: the Vulkan
  loader and a shader compiler, the CUDA toolkit.

## 7. Weight streaming (R8)

A model larger than RAM runs by streaming its weights. Each GGUF
shard is mmap'd once, read-only and private, so weight pages are clean
and file-backed: the kernel can always evict them, and they never go
to swap. The dirty memory of a run is only the KV pool and the
workspaces.

### Readahead

Streaming is CPU-bound only if reads overlap compute. Two hints,
issued with `madvise(WILLNEED)`:

- at MoE routing time, on the routed experts of the current layer;
- at the start of layer l, on layer l+1's static tensors (attention,
  dense FFN, shared experts), which are known ahead.

Measured on GLM-5.2 (UD-IQ1_S, 216 GB, neon, 8 cold forwards, equal
fault counts):

| Policy                        | Wall time | Sys time |
|-------------------------------|-----------|----------|
| passive streaming             | 524.4 s   | 60.9 s   |
| + expert readahead            | 461.7 s   |          |
| + next-layer readahead        | 351.4 s   | 22.9 s   |

Both hints are on by default when the weights may stream (the model is
over half of RAM, or RAM is unknown). A model that fits stays resident
after one pass, and hinting resident pages every token is not free
(about 4.5 ms per token on Orpheus 3B). `LOCUS_NO_READAHEAD=1`
restores passive streaming for comparisons.

A thread that touches the same pages would only duplicate the
kernel's asynchronous readahead, so there is none. It is worth
revisiting only if large sys time remains with the hints on.

### Working set

Routing spreads fast. On Moonlight, 11 forwards touch 43.8% of the
layer-expert slots; on GLM-5.2 (256 experts), 16 forwards touch 27.4%.
So prefetch pays within a token (overlap the reads of the routed
experts), not across tokens (an expert cache). `LOCUS_MOE_STATS=1`
prints this telemetry.

On a cold spinning disk, GLM-5.2 runs faster end to end than a
CPU-only llama.cpp (50.7 versus 92 minutes for 16 tokens), because
locus streams lazily instead of loading the whole model first. Major
fault counts are a false coldness signal under readahead (pages fault
in ahead of access); bytes read are the reliable one.

### Memory bounding

- `--ctx N` caps the KV pool in tokens.
- `LOCUS_WEIGHT_WINDOW=1` drops each layer's routed-expert pages right
  after use (`madvise(DONTNEED)`, inward-aligned), so a streamed model
  builds no page-cache pressure. It applies only to file-backed
  weights: on an anonymous in-memory image, DONTNEED would discard the
  contents. Static, shared and pinned weights are never dropped.

## 8. GPU weight paging (R8-GPU)

Each GPU backend keeps weights in a pager, so a model larger than
device memory still runs.

- Page unit: a whole tensor for static weights; one expert slice for
  routed experts, so paging granularity matches routing granularity.
- Budget: `LOCUS_GPU_POOL_MB`; unbounded by default, so a model that
  fits stays resident. Weights stay quantized on the device for every
  type whose kernel dequantizes in-shader (Q4_K is about 0.5 bytes per
  weight).
- Page table: host pointer to {buffer, last use, pinned}. Buffers used
  by the command batch being recorded are pinned; unpinned ones evict
  least-recently-used. The host mmap stays the source of truth.
- Prefetch (CUDA): `Ops::prefetch` starts an async upload of the next
  layer's static tensors and of each routed expert at selection time,
  on a second stream. The model calls it at the same sites as the
  `madvise` hints, with the same `Mat` (same host pointer, the page
  table key) that the later matvec uses.
- Vulkan has no prefetch. On the Pi's V3D, upload (a memcpy into
  host-coherent unified memory) is 0.28% of compute with an unbounded
  pool and 4.5% even under heavy paging, so prefetch could gain at
  most a few percent.
- Vulkan uploads only the routed experts of a token, not every expert
  of the layer (6 of 64 on DeepSeek-V2-Lite).

Vulkan records one command batch per token and frees buffers only
between batches, so one batch's working set must fit device memory.

Validated: Llama-3.2-1B Q4_K_M under `LOCUS_GPU_POOL_MB=64` on the Pi
5 is byte-identical to the unbounded and scalar runs, with 341
evictions. DeepSeek-V2-Lite (10 GB, MoE + MLA) runs token-exact on the
Pi 5's GPU within about 2 GB of RSS.

On a tile GPU streaming a model larger than its pool, Vulkan runs but
slower than the CPU (DeepSeek-V2-Lite on the Pi: 85 s against 21 s for
NEON). There it is a capability win, not a speed win.

## 9. Threaded execution and static pinning (R9)

- R9.1, threaded matvec: a large matvec splits into row slices
  (`mat_rows`) on a persistent thread pool, each slice through the
  active backend's matvec into a disjoint output span. Each row's dot
  is the same code, so logits are bit-identical for every thread
  count. `LOCUS_THREADS` caps the fan-out (default: the performance
  cores; 1 runs single-threaded). Parallel slices also issue parallel
  page faults, a deeper NVMe queue on streamed models. Independent
  matvecs reading the same input (q/k/v, gate/up) share one pool
  dispatch, and decode attention spreads its heads across the pool.
- R9.2, static pinning: `LOCUS_PIN_STATIC=1` mlocks every non-expert
  weight at load. Opt-in, because wiring about 13 GB on a 32 GB
  machine is a policy decision.
- R9.3, MTP decode through the nextn tensors that GLM and DeepSeek
  ship: not built.

GLM-5.2 (UD-IQ1_S, 216 GB) on a 31 GB host with a GTX 750 Ti and a
7200 rpm disk, 16 tokens cold, all token-exact:

| Engine             | Time    | Per token | Speed-up |
|--------------------|---------|-----------|----------|
| llama.cpp (CPU)    | 5523 s  | 345 s     | 1.00x    |
| locus, sse4        | 3042 s  | 190 s     | 1.82x    |
| locus, CUDA        | 1793 s  | 112 s     | 3.08x    |

CUDA wins despite about 1.2 GB of usable VRAM (12.4% pool hits, 601 GB
re-uploaded over PCIe): the experts stay in the host page cache, so
re-uploads are cheap next to the GPU matvec.

## 10. DSA indexer (GLM-5.2 beyond top_k)

GLM-5.2 ships DeepSeek-V3.2's sparse attention: a lightning indexer
selects which cached tokens the MLA attention sees, so the context is
not capped at `attention.indexer.top_k` (2048).

- Per layer and token, with $x$ the layer input:
  - the cached key: $k_{idx}(s) = \mathrm{LayerNorm}(W_k x_s)$, 128
    values;
  - the query: $q_{idx}(t) = W_{qb}\  q_a(t)$, viewed as 32 heads of
    128;
  - the head weights: $w(t) = W_{proj}\  x_t$, 32 values.
- Rope on the first 64 dimensions of both $q_{idx}$ and $k_{idx}$,
  half-split (not interleaved, unlike the main attention).
- Score:
  $\mathrm{score}(t, s) = \sum_h w_h(t) \mathrm{ReLU}(q_h(t) \cdot k_{idx}(s))$.
  Global scale factors do not change the top-k order, so selection
  ignores them.
- Attention and its softmax run over the top-scoring
  $\min(\mathrm{topk}, n)$ positions only, of the $n$ cached. Up to
  top_k tokens every position is selected, so the dense golden anchors
  the whole path below 2048 tokens.
- The cache row grows from 576 to 704 floats (the MLA row plus the 128
  of $k_{idx}$).

llama.cpp ignores the indexer tensors (dense fallback), so there is no
external golden beyond top_k, and long real-model runs are impractical
at tens of seconds per token. Unit tests on synthetic models check
that short sequences match the dense path bit-exactly, that exactly k
positions are attended beyond top_k, and that selection picks the
positions a scalar reference scores highest.

## 11. Batched forward (R10)

A one-token forward reads every weight once per token. A batch of N
tokens can share that read, from three sources:

1. Prefill: N prompt positions of one sequence.
2. Continuous-batch decode: one token from each of N running
   sequences. N users then share one weight stream per step, which is
   what makes serving a bigger-than-RAM model to several users viable.
3. Speculative decoding: verify K drafted tokens in one forward.

Single-stream decode without speculation cannot batch.

The weight-bearing ops (projections, dense and expert FFN, output
head) run weight-stationary: each weight is used for all N tokens
before the next weight. Each output $y_{r,t} = W_r \cdot x_t$ comes
from the same backend matvec as in the per-token path, so
`forward_batch` gives a KV cache and logits byte-identical to N
`forward()` calls; the tests assert it. Attention reads no large
weights, so it stays per token, each against its own sequence's KV.
MoE layers route per token, then apply each picked expert once to all
the tokens that chose it. Activations go N wide; the attention scores
buffer stays one wide (an N-wide one would be N x n_ctx).

- `forward_batch_decode` decodes one token from each of N sequences in
  one pass. The engine uses it by default (`Engine::Config::
  batched_decode`); `locus-run` enables it with `--batch-decode`.
- Prompt ingestion is batched by default (`--no-batch-prefill` turns
  it off). Vulkan has its own chunked prefill, `vulkan_forward_batch`
  (chunks of up to 512 tokens, 1024 for MoE; token-exact, not
  byte-identical, to per-token prefill).
- `LOCUS_BATCH_DEQUANT=1` selects a kernel that dequantizes each
  weight row once for all N tokens. It sums in another order, so it is
  token-exact, not byte-identical, and opt-in.
- The batched matvec row-slices across the thread pool like the
  per-token one, so batching never loses the cores.

## 12. Cache-blocked batched matvec (R11)

Batched decode is a skinny GEMM, $Y = W X^\top$ with $X$ of N rows.
Looping the per-token matvec streams $W$ from DRAM N times. The
register-blocked kernels load and dequantize each weight block once
and keep N accumulators, so weight traffic drops about N-fold. They
keep matvec's per-block reduction order, so each token's dot is
bitwise unchanged (a single end-of-row reduction would differ: float
addition is not associative).

`Ops::matvec_batch` is the hook. Kernels exist for Q4_K and Q6_K on
sse4 and neon, and on CUDA (one launch for all N tokens). A backend
kernel that covers only some types falls back to N calls of the same
backend's matvec, never the scalar reference, so its batched and
per-token paths agree.

Llama-3.2-1B Q4_K, 8 concurrent streams, in RAM:

| Backend and memory        | Fallback  | Kernel    | Ratio  |
|---------------------------|-----------|-----------|--------|
| cuda, GTX 750 Ti          | 2.2 tok/s | 3.4 tok/s | 1.55x  |
| neon, LPDDR4X (Pi 5)      | 4.4 tok/s | 6.2 tok/s | 1.35x  |
| sse4, DDR3                | 4.5 tok/s | 4.9 tok/s | 1.09x  |

The box with less bandwidth gains more. After the weight-traffic cut,
the loop is compute-bound, so 128-bit sse4 caps the x86 ratio. An
avx2 kernel waits on an AVX2 host (#61).

## 13. Serving features (R12 to R15)

Each is off by default or greedy-preserving, so the goldens keep
guarding the default path.

- Sampling: temperature, top-k, top-p, min-p, repeat / frequency /
  presence penalties, per-request seed. Temperature 0 is argmax. The
  filter order matches llama.cpp (penalties, constraint, top-k,
  temperature and softmax, top-p, min-p, draw).
- JSON mode: an incremental JSON-grammar validator masks tokens whose
  bytes would break the grammar (`response_format: json_object`).
- Prompt caching: exact token-prefix keys, ref-counted block pins, LRU
  eviction; the longest shared prefix is adopted, leaving at least one
  token to recompute the last logits. Lookup is indexed by a 64-bit
  digest of each block boundary, so it costs O(prompt); the token
  sequence stays the verified key, so a collision never returns a
  wrong prefix.
- Speculative decoding: prompt lookup (no draft model). An n-gram
  match against the context proposes drafts, verified in one batched
  forward; the longest correct prefix is accepted. Greedy only, so
  byte-exact to plain greedy.
- Embeddings: `/v1/embeddings` returns the last token's final-normed
  hidden state, L2-normalized, on a dedicated cache. CPU and CUDA
  backends only (`TransformerModel::embed` refuses vulkan).
- API: `/v1/models`, `/metrics` (Prometheus), OpenAI-shaped logprobs
  (a log-softmax over the raw logits) and usage accounting.
- Metrics: every `/metrics` series carries `model` and `backend`
  labels, and `locus_info` adds `arch`, `chat_template` and `kv_type`.
  Request counters and KV gauges sit beside per-request histograms
  (`locus_queue_seconds`, `locus_prefill_seconds`,
  `locus_time_to_first_token_seconds`, `locus_decode_token_seconds`,
  `locus_request_seconds`), token counters for prefill and decode, the
  decode wall time, the engine's step time
  (`locus_engine_busy_seconds_total`,
  `locus_engine_steps_total`) and the running and waiting counts. The
  engine writes them as relaxed atomics, a few clock reads per request
  and two per step, and `/metrics` reads them without the engine lock;
  decode speed is unchanged within noise.
- KV-cache quantization: Q8 or Q4 blocks of 32 (f16 scale) in a
  separate byte pool, for the llama-family attention. Each layer's
  past rows are dequantized into scratch, so the attention math is the
  f32 path's. Checked by cosine similarity (Q8 above 0.999). F16 KV
  also runs on the GPU for GQA models. MLA already shrinks the KV
  cache, so its quantized path is low priority (#63).
- More architectures and quant types: qwen2moe, qwen3moe, dbrx, glm-
  dsa, muse-spark, deepseek41; ternary TQ1_0 / TQ2_0 and the IQ
  family.
- Multi-GPU weight sharding (#62): shard the streamed weight window
  across the GPUs of one host. Needs multi-GPU hardware.

## 14. Multi-server inference (R15+)

Pooling the memory of several hosts runs a model that fits none of
them. It is a capacity lever, not a throughput lever: throughput
scales by replication (full copies behind a load balancer), capacity
by splitting the model.

### Pipeline parallelism

Each host owns a contiguous block of layers. A token's hidden state
flows host to host; the last host produces the logits. Weights load
once and stay resident on each host. Only activations cross the
network: one hidden vector per token per boundary, about $2d$ bytes
at 16 bits for a model of width $d$. That small payload is why this
works over a LAN. Tensor parallelism instead exchanges data inside
every layer and needs an NVLink-class link, so it is not the target.

Latency rises by about (hops x one-way delay). Throughput need not
fall, if sends are asynchronous and enough requests are in flight to
keep every stage busy:

```math
\mathrm{concurrency} \approx \mathrm{stages} \times \left(1 + \frac{\mathrm{delay}}{\mathrm{compute\ per\ stage}}\right)
```

Network delay does not lower the throughput ceiling; it raises the
concurrency needed to reach it. Prefill can be chunked and streamed
stage to stage, so it fills the pipe even for a single request.

The KV cache is per layer, so each stage owns exactly its layers'
cache. Prompt caching works per stage, and the cluster's larger
memory keeps more prefixes.

### Expert parallelism

For MoE models, experts can live on different hosts, each processing
the tokens routed to it. Routing adds traffic, but experts are
independent, so this is a second axis that also suits a LAN. The
multi-GPU pager (#62) shards weights within one host; a pipeline stage
can use it underneath.

### Stage serving

`locus-stage` runs a layer range and `locus-driver` drives generation
across a stage chain. Activations travel as length-prefixed frames
over TCP, tagged by request and position. A connect proves only that
the downstream listener is up; liveness needs an app-level probe.

A node separates I/O from compute (#24):

- One I/O thread owns every socket (`sys::Poller`), decodes frames and
  marks sessions ready. It never blocks on compute.
- An executor is one compute device running a continuous-batching
  loop: each iteration it gathers the ready sessions, runs one batched
  forward (`forward_batch_layers`), and returns the outputs. It owns
  that device's paged KV pool.
- One executor per device; a session is pinned to one executor, so its
  KV stays on that device.
- The two sides talk through lock-protected queues and a self-pipe
  wake in each direction; all socket I/O stays on the I/O thread.

Three handoff rules keep the split safe:

- A session in flight is only marked cancelled when its client drops;
  its KV is released when the batch completes. Freeing it mid-batch is
  a use-after-free.
- At most one frame per session is in flight, which makes ordering
  free.
- One wake can cover many completions, so the I/O thread drains the
  whole completion queue per wake.

The executor copies the engine's scheduler structure (admission,
running set, preemption, budget) but is not `engine::Engine`: a stage
has no prompt, no stop rule and no sampling, and a middle stage emits
no token.

CPU executors share one thread pool, so the useful middle setting (a
few executors, each multi-threaded) needs per-executor pools (#45). A
CUDA executor binds its device per thread and fails loudly on a bad
device; several GPUs in one process need a multi-GPU host to validate.
A Vulkan executor needs a batched, layer-range Vulkan forward (#48)
and a device-select seam (#47).

Exit test: a two-host split gives byte-identical output to the
single-host run; at concurrency at or above the pipeline depth,
throughput approaches the single-box ceiling, with only the hop
latency added.

## 15. 8-bit activations (R16)

llama.cpp does not dot quantized weights against f32 activations. For
QK_K weights it quantizes the activation row to Q8_K (an f32 scale,
256 int8 values, 16 int16 block sums) and integer-dots the two; for
block-32 weights (Q8_0, Q4_0, IQ4_NL) it uses a Q8_0 activation.
locus ports those dots (`backend::matvec_q8k`):

| Backend | 8-bit activation dots                                         |
|---------|---------------------------------------------------------------|
| scalar  | all 15 QK_K types and the block-32 types                      |
| neon    | Q4_K, Q6_K and the block-32 types; the rest through scalar    |
| sse4    | the five K-quants; the rest through scalar                    |
| cuda    | all 15 QK_K types                                             |

The vectorized and CUDA dots replay the scalar accumulation order, so
they are bit-identical to it.

"Bit-exact with llama.cpp" is unreachable in principle: ggml is not
bit-exact across its own backends, because the final f32 sum runs in
an implementation-specific order. With 8-bit activations, argmaxes sit
close enough that this order noise tips tokens on some goldens.

The mode is on when the architecture requires it
(`Hparams::q8k_default`: muse-spark, deepseek41) or when the backend's
8-bit dot is its fast path (`Ops::prefers_q8k`: neon, sse4, scalar).
avx2, cuda and the Vulkan per-op fallback keep the f32 dot.
`LOCUS_F32_ACTIVATIONS` (any value but "0" or empty) or
`use_q8k_activations(false)` selects the f32 path, the reference for
exactness checks. On the M4 Max CPU, Orpheus 3B Q4_K_M decodes at 63
tok/s with 8-bit activations against 29.4 with f32.

Prefill and decode take the same gate (`uses_quantized_activation`),
so they agree. F32, F16 and BF16 weights always take the f32 dot.

Goldens: Qwen1.5-MoE and DeepSeek-V2-Lite are pinned to f32, because
8-bit activations tip their greedy text at a near-tie. Muse Spark's
second prompt, DeepSeek-V4.1 and the Orpheus codec frame are checked
teacher-forced: each reference token must be within 0.25 of the top
logit.

## 16. Testing strategy

- Catch2 unit tests per component (allocator, scheduler invariants,
  GGUF parser, tokenizers, kernels against the scalar reference).
- Golden tests: a fixed prompt and greedy decoding reproduce the
  reference (llama.cpp, or the torch reference for multimedia models)
  token for token, or teacher-forced within a margin where a near-tie
  tips greedy decoding.
- stories260K runs on every selectable backend; the larger goldens
  run on the CPU and on Vulkan (qwen2moe also on CUDA).
- A malformed-GGUF corpus for the loader (truncations, overflowing
  offsets, hostile metadata).
- Scheduler property tests: no block leaked, no double free, ref
  counts back to zero after every workload.
- Tests that need a model file skip without it; heavy ones also need
  `LOCUS_HEAVY_TESTS=1`.
