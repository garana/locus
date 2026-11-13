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
  recompute it later (recompute, not swap to host); with a disk tier,
  its full blocks are written there and read back on readmission.
- Fairness: admission is first come, first served, with a cap on
  prefill tokens per iteration (chunked prefill). Within it the prompt
  with the shortest remainder is fed first, so a short prompt behind a
  long one does not wait for the whole long prefill (a prompt unfed for
  two prefill iterations goes first, so short ones cannot starve a long
  one), and each prompt
  is fed in the chunks it gets alone (the whole cap, then its
  remainder), which keeps its output independent of the batch. A
  prompt resumed from a cached prefix (the RAM cache, or the disk tier,
  also after a preemption) is fed from the prefix's end, so its chunks
  match an uncached run's only when the block size divides the cap and
  the prefix ends on a chunk boundary. While
  a sequence decodes, every other iteration feeds no prompt, so long
  prompts do not starve decoding sequences.

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

Each SIMD kernel variant (neon, sse4, avx2) is its own source file,
compiled with only that variant's flags and gated at configure time;
`locus::sys::detect()` probes the running machine. One binary stays
portable across CPU generations. When the compiler's default arm64
target lacks the 8-bit dot product (GCC on the Pi 5), the neon kernels
are also built with it (`matvec_neon_dotprod.cpp`), and the neon
backend uses that build on a CPU that reports dotprod; the integer
sums are exact either way, so the results are the same. The sse4
kernels are likewise built with AVX (`matvec_sse4_avx.cpp`) for CPUs
with AVX but not AVX2 (Sandy and Ivy Bridge): the sse4 backend uses
that build when the CPU reports AVX. It keeps the sse4 float order (the
batched Q8_0 kernel's two 4-lane sums share one 256-bit register), so
the results are the same, 7 to 19% faster per kernel on an i5-2400.
Vulkan
shaders are compiled to
SPIR-V and embedded at build time.

### 4.4 GGUF loader

Parsing untrusted model files is an attack surface (llama.cpp has had
parser CVEs). Rules: bounds-check every offset against the file size;
no allocation sized by an unvalidated field; a fuzz corpus in the
tests; mmap read-only.

### 4.5 HTTP server

A first-party HTTP/1.1 server (`src/locus/http`) and vendored
nlohmann/json for untrusted request bodies: hand-rolled JSON parsing
is where API vulnerabilities come from. A worker thread drives
`Engine::step()`; handler threads submit and wait (`EngineLoop`).
Chat messages are formatted by `chat::ChatTemplate`: the GGUF's
Jinja source is fingerprinted to a known family (llama3, chatml,
llama2, zephyr, deepseek, deepseek2, deepseek-coder) and rendered by
a hardcoded formatter; an unknown template falls back
to plain concatenation. Endpoints that run the model on their own
thread (images, vision chat, audio) share the GPU backend through one
process-wide lock (#268). `http::Client` is the first-party client
for locus's own upstream calls (the router of #388): a pure response
parser with the request parser's hardening, a streaming chunk
decoder, and connect and read deadlines with one kept-alive
connection, resent once when the server closed it while idle.
`http::LoopConnection` serves HTTP on a caller's own event loop
(`sys::Poller`) instead of a thread per connection: the loop accepts
and owns the fd, the connection reads, parses, calls a non-blocking
handler and writes, and says what to watch next; for small endpoints
on a thread that already runs a loop, such as a stage's `/metrics`
(#22).

`locus-router` (`router::Router`) proxies each request to one of
several instances by kind: text, image-to-text (a chat, messages or
responses body with an image part), image, video, speech, rerank or
classify, from the path and, for chat, the body; `/v1/embeddings` under
a classify instance's model goes to that instance. The instances are
given as host:port or as `srv:NAME`, whose DNS SRV targets (the
lowest priority's) are looked up again on every refresh, so instances
join and leave with the record; one that leaves is never picked
again, and an empty answer keeps the last list.
`scan:A.B.C.D/N:PORT` (IPv4, at most 1024 addresses) probes every
address of the range for a `/v1/models` at start and every sixth
refresh, 64 at a time with a 300 ms connect
timeout; the ones that answer are its instances, and a scan that no
longer finds one drops it (an empty scan is an answer, unlike an empty
SRV lookup). A scan admits any server in the range that answers with
a `data` list, not only locus, and with `--upstream-auth client` it
gets the clients' prompts and keys, so scan only a range where
everyone who can listen is trusted. A host that accepts the connect
and then sends nothing holds a probe for the 2 s read timeout, so a
range full of them makes a scan last about 1024 / 64 x 2.3 s = 37 s.
`mdns:SERVICE` (for example `mdns:_locus._tcp`) sends one DNS-SD
browse on every refresh: a PTR question for `SERVICE.local` to
224.0.0.251:5353 from an ephemeral port, so responders answer by
unicast (RFC 6762 6.7) and the router joins no group; for 500 ms it
reads the PTR, SRV and A (or AAAA) records into host:ports
(`pipeline::mdns_browse`, its own bounded parser). An empty browse
keeps the last list, since mDNS packets get lost. Instances advertise
with the OS (`dns-sd -R`, `avahi-publish-service`). Only replies from
port 5353 and from loopback or an interface's own network are read
(a container bridge such as docker0 is one, so a container on the
host can answer too), but anyone on the LAN can still answer, so the
scan's trust rule holds for it too; with `--upstream-auth client`, an
`mdns:` or `scan:` source makes the router warn at startup that client
keys go to whatever answers.
With `--register-keys`, an instance can also add
itself (`locus-server --register`): it POSTs its host:port to
`/locus/register` with one of the keys, again every third of its TTL
(30 s), and the router drops it when that runs out or on
`/locus/deregister`; without keys the route is a 404, since anyone who
could register could take the router's traffic. A registration key is
worth as much as a client key: the router sends the requests it routes
to a registered address, with the client's own key in the default
`--upstream-auth client`. An instance found by SRV or registration
gets an `--instance-keys` key only from a line that names its
host:port, never the `*` one, so the router sends its own keys only to
addresses its operator named. Every proxied request carries
`X-Locus-Router: 1`, and a router refuses (508) a request with it, so
a registration at a router's own address cannot loop. An instance
lists its kinds in
`/v1/models` (`locus_kinds`), which the router reads at start and
every `--refresh-ms`; a request that names a listed model goes to that
instance. Image edits (`image-edit`: FLUX.1 Kontext, Qwen-Image-Edit)
are a kind apart from generations (`image`), so an edit-only instance
gets no generation; while no instance lists `image-edit` (older
servers), edits go to `image` instances. Buffered replies are
forwarded whole; a chunked one (SSE) streams through, and a client
that leaves closes the upstream connection, so the instance stops the
work. An instance that stays
silent longer than the read timeout fails the request:
`--media-timeout` for image, speech and video and `--text-timeout` for
the rest, 30 minutes each by default, since a buffered image or chat
sends nothing until it is done and a stream nothing during its
prefill.
With several instances of a kind, a session key (the client's own, or
a hash of the system prompt and first user turn) orders them by
rendezvous hashing, so a conversation keeps its instance's prefix
cache, every router agrees with no shared state, and a leaving
instance moves only its own keys.
An instance over 1.25 times the candidates' average in-flight count
goes last, and a connect failure or a busy 503 moves on to the next.
A video job id from the router names the instance that made it
(`video_`, its host:port in hex, `_`, the instance's own id without
its `video_` prefix, or after a second `_` whole when it has none), so a
`GET /v1/videos/{id}` reaches that instance with no table, also after
a router restart. The router's own `/metrics` counts, per instance,
requests, upstream errors, streams a client left, wall seconds and
the requests in flight, and whether the last poll answered. With
`--auth-helper` the router checks client keys on the request head
through the same helper protocol as the server (`AuthClient`);
`/health` stays open. Toward the instances it sends the client's own
credential, none, or a key per instance from an `--instance-keys`
file, which also authenticates its `/v1/models` polls.

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
  cpp-httplib v0.18.3 (MIT, tests only: their HTTP client),
  nlohmann/json v3.11.3 (MIT), ggml's
  IQ lookup tables (`third_party/ggml/iq_grids.h`, MIT) and llama.cpp's
  Unicode tables (`third_party/llama/unicode-data.{h,cpp}`, MIT, commit
  cd959bae9).
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
prints this telemetry. `LOCUS_MOE_TRACE=<file>` writes every pick as
`layer expert`; `scripts/moe-trace-sim.py` replays a trace against
cache policies (LRU, LFU, Belady, step WLRU, static top) for a given
GPU budget. On Vulkan both variables route MoE layers on the host, so
the picks reach the CPU.

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
- MoE hybrid (CUDA; on by default when the model's weights exceed
  90% of the pool, `LOCUS_MOE_HYBRID=0` off, `=1` on): only the routed
  experts in a hot zone run on the GPU; the other picks run on the
  best CPU backend at the same time, on a second thread. The hot
  zone is a step WLRU over every (layer, expert) of the model: one
  ranked list, each pick moves its expert `LOCUS_MOE_STEP` ranks up
  (default 16), and the first N ranks are resident. An expert that
  enters the zone starts an async upload (it runs on the CPU until
  then); the one it pushes out is freed at once (`Ops::evict`). N is
  `LOCUS_MOE_HYBRID_SLOTS`, else what fits in 90% of the pool after
  the non-expert weights. Batched prefill moves the zone the same way,
  pick by pick in token order. On a GTX 750 Ti (2 GB) with an i5-2400
  (sse4), a 12 to 13 token prompt and 64 decode tokens, Q4_K_M, model
  file in the page cache:

  | model            | sse4 | cuda | hybrid s=1 | hybrid s=16 | hot zone |
  |------------------|------|------|------------|-------------|----------|
  | DeepSeek-V2-Lite | 1.2  | 1.1  | 2.6        | 2.5         | 5.9%     |
  | Moonlight-16B    | 0.8  | 1.0  | 2.0        | 2.2         | 4.3%     |
  | Qwen1.5-MoE-A2.7 | 0.7  | 1.7  | 4.1        | 4.2         | 3.0%     |

  (decode tok/s). Prefill with s=16 against cuda: 5.0 s against 11.0
  s, 5.2 against 11.0, 3.0 against 6.5. The gain comes from not
  uploading a missed expert: GPU hits are 4 to 13% of picks with
  zones that small. The CPU reads the cold experts from the model
  file, so a model that does not stay in the page cache streams them
  from disk. Each layer asks for its cold experts with
  `advise_willneed` before the GPU work starts, so the reads overlap
  it: with a cold cache on spinning disks, DeepSeek-V2-Lite decodes
  15 tokens in 21.8 s against 28.3 s without; its 17-token prefill
  (78 s) stays disk bound, since it touches nearly all 10 GB.
- Vulkan has no prefetch. On the Pi's V3D, upload (a memcpy into
  host-coherent unified memory) is 0.28% of compute with an unbounded
  pool and 4.5% even under heavy paging, so prefetch could gain at
  most a few percent.
- Vulkan uploads only the routed experts of a token, not every expert
  of the layer (6 of 64 on DeepSeek-V2-Lite).
- Where the device has `VK_EXT_external_memory_host` (MoltenVK),
  Vulkan imports a tensor's mmap'd pages instead of copying them, for
  every type stored as the shaders read it (all but Q8_0, which is
  repacked), 64 KB or larger, a multiple of 16 bytes (so the buffer
  needs no zeroed padding), and inside a model file's mapping. The buffer binds at the memory
  alignment below the tensor and its descriptors start at the tensor.
  Imported weights cost no device memory of their own and don't count
  toward the pool budget or the MoE expert fit. Qwen3-Coder-30B Q4_K_M
  on an M4 Max: peak footprint 21.2 GB to 2.8 GB, and a cold first
  prefill 18.6 s to 7.4 s, at the same speed once warm.
  `LOCUS_VK_NO_IMPORT=1` keeps the copies. Unmapping a model file
  calls a hook (`sys::set_unmap_hook`) that drops every pool buffer
  over its pages first, imports and copies alike, so no buffer
  outlives the memory it reads and a later mapping at the same
  address gets fresh buffers.

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
  batched_decode`); `locus-run` enables it with `--batch-decode`. On
  Vulkan it runs `vulkan_forward_multi` on groups of up to 8 sequences
  (dense models with Q8_0, Q4_K, Q6_K, F32, F16 or BF16 weights; others
  run one forward per sequence; Q8_0 and Q4_K run gate, up and silu_mul
  in one dispatch, and Q8_0 q, k and v in another). In batch-invariant
  mode a lone sequence takes the multi-sequence kernels too, so a
  request's logits are bit-identical alone and batched. It is the
  default on Apple GPUs, where a lone sequence costs little; elsewhere
  a lone sequence keeps the single-sequence forward (on V3D the mode
  costs a lone Q8_0 sequence 3% with the NT 1 build below, but stays
  off by default until i#682 is understood). Under the default a lone
  sequence on a Q4_K or Q6_K model keeps it too (the K-quant
  short-batch kernels cost it 9 to 11% on an M2 Pro), so such a model
  is batch-invariant only with the mode set.
  `LOCUS_VK_BATCH_INVARIANT=1` / `0` (or `vulkan_set_batch_invariant`)
  overrides. Q8_0 batches of 1 or 2 take the NT 2 build of the
  short-batch kernel (on V3D a lone sequence the NT 1 build, with
  matvec_q8_0's inner loop: V3D is register-bound and the NT 2 loop
  cost it 39 to 50% per byte; V3D only, as its loop shape gives the
  other builds' bits there but not under MoltenVK's fast math;
  `LOCUS_VK_Q8N1=0` turns it off, i#511), 3 or 4 the NT 4 build (#516)
  and larger ones the NT 8 build (Q4_K and Q6_K too,
  `matvec_k_n.comp`, #436, with its NT 4 builds from #541); the builds
  run the same per-token arithmetic, so a
  sequence's bits do not depend on the batch (#432; a test checks the
  builds agree bit for bit on the device). Llama 3.2 1B Q8_0 on an M4
  Max: 2, 3, 4 and 8 sequences make 314, 361, 405 and 404 tok/s in all
  (312 and 347 at 3 and 4 with the NT 8 build), against 198 alone (210
  with the single-sequence forward); on an M2 Pro the two are level
  (93.9 and 93.0). Orpheus 3B Q4_K_M on the M4 Max: 2, 4 and 8 sequences
  make 140, 155 and 166 tok/s in all, against 102 one forward per
  sequence (M2 Pro: 46.5, 43.2 and 45.8 against 41.5), and 3 and 4
  sequences 147 -> 154 and 156 -> 162 tok/s with the NT 4 builds.
- Prompt ingestion is batched by default (`--no-batch-prefill` turns
  it off). Vulkan has its own chunked prefill, `vulkan_forward_batch`
  (chunks of up to 512 tokens, 2048 for MoE; token-exact, not
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
sse4 and neon, and on CUDA (one launch for up to 32 tokens) for Q4_K,
Q5_K, Q6_K, Q8_0, IQ4_NL and IQ4_XS. A backend
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
- Structured output: every format (JSON mode, a JSON Schema, a regex)
  becomes a GBNF grammar. A matcher keeps the set of pushdown stacks
  the output so far allows, over UTF-8 code points; a rule reference
  at the end of a sequence drops the caller's frame, so repetition
  does not grow the stacks, and left recursion is rejected at parse
  time. A state keeps at most 4096 stacks; one that has more drops
  the rest, logs it once and counts the reply once in
  `locus_grammar_stack_overflows_total`, since a dropped stack can
  reject output the grammar allows. Only the JSON formats stop a
  complete reply at its last brace when nothing but whitespace could
  follow; a grammar or a regex may go on after whitespace. Each step's allowed tokens come
  from one walk of a trie of the vocabulary's bytes, cached by matcher
  state (inside a string the state repeats). EOS is allowed only on a
  complete match and is the only choice once nothing but whitespace
  can follow, so a reply ends at its closing brace. With a 128k
  vocabulary on
  an M2 Pro (Llama 3.2 1B, CPU) a JSON Schema costs about 3.5 ms per
  token over free decoding.
- Prompt caching: exact token-prefix keys, ref-counted block pins, LRU
  eviction; the longest shared prefix is adopted, leaving at least one
  token to recompute the last logits. Lookup is indexed by a 64-bit
  digest of each block boundary, so it costs O(prompt); the token
  sequence stays the verified key, so a collision never returns a
  wrong prefix. A disk tier (`--prefix-cache-dir`, i#416) keeps one
  file per prefix with the full key (the model's GGUF header, size and
  mtime, the backend and device, the KV geometry, RoPE settings and
  the numeric modes that change KV bits, and on Vulkan whether the MoE
  experts run on the GPU, settled once the KV pool exists; a model with
  the MoE hybrid on is refused, since its hot set follows the requests
  served before), the tokens and a checksum per block, so a mismatch
  or a torn file reads as a miss; a byte cap evicts the least recently
  used file. Entries
  evicted from RAM are written back by a worker thread, and the rest
  at shutdown; a prompt whose prefix is at least one block longer on
  disk waits in a LOADING state while the worker reads it into its
  blocks, and the engine keeps stepping the other requests. A
  preempted sequence writes its full blocks (prompt and output so far,
  keyed by those tokens) to the disk tier too; on readmission it loads
  them the same way and recomputes only the tail (i#621); blocks still
  in the worker's write queue are copied from it (i#675). Blocks past
  the write backlog are written at preemption from the pool, so the
  step waits for the disk instead of a long recompute later (i#642).
- Speculative decoding: prompt lookup (no draft model). An n-gram
  match against the context proposes drafts, verified in one batched
  forward; the longest correct prefix is accepted. Greedy only, so
  byte-exact to plain greedy. Each draft's length (0 to
  `--spec-draft`, default 8) is the one with the most expected tokens
  per second: the request's decayed accept rate a gives
  $1 + a + \dots + a^d$ tokens for d drafts, priced by the engine's
  measured time for a verify of d drafts (the forward and the logit
  scans); a length not measured yet costs the measured cost of a row
  per extra row. On the M2 Pro CPU a verify row
  costs 0.55 to 0.75 of a decode step; on MoltenVK a verify of 9 rows
  costs about 10 decode steps against 4 for 5 rows, so fixed 8 drafts
  are slower than none. A plain decode step also checks a one-token
  draft against the decoded token, so the accept rate follows the
  text while no draft is verified. Llama 3.2 1B Q8_0 on an M2 Pro,
  160 tokens, tok/s over 3 runs (CPU runs vary by about 10%):

  | prompt      | backend | off      | fixed 4    | fixed 8  | adaptive   |
  |-------------|---------|----------|------------|----------|------------|
  | code rename | CPU     | 52 to 60 | 71 to 83   | 88 to 98 | 82 to 90   |
  | code rename | Vulkan  | 90       | 116 to 118 | 76       | 115 to 120 |
  | summary     | CPU     | 52 to 68 | 61 to 63   | 48 to 55 | 63 to 66   |
  | summary     | Vulkan  | 92       | 87         | 59       | 87 to 90   |
  | free text   | CPU     | 72 to 78 | 70 to 73   | 70 to 74 | 66 to 70   |
  | free text   | Vulkan  | 91 to 92 | 88 to 89   | 81       | 92         |
- Embeddings: with `--embedding-model`, a GGUF of arch `bert` (BERT
  with WordPiece, or XLM-R with SentencePiece) or `nomic-bert` (NEOX
  rope instead of a position table, fused qkv, SwiGLU, no biases;
  `embed::BertEncoder`) serves `/v1/embeddings` for every request
  whose model is not the chat model: the inputs run as one batch
  (passes of up to 4096 tokens, attention within each input), pooled
  as the GGUF's `pooling_type` says (mean, CLS or last),
  L2-normalized; an input past the model's context is a 400. Its
  activations stay f32, so it matches float64 arithmetic on the same
  Q8_0 weights to a cosine of 0.99999, and on bge-small it is nearer
  the f32 original (0.99989 to 0.99995) than llama.cpp, which rounds
  them to 8 bits (0.99971 to 0.99989; 0.99965 from exact on bge-m3's
  24 layers). nomic-embed v1.5 loses more to Q8_0 itself: its Q8_0
  weights are 0.99887 to 0.99935 from its F16 ones (llama.cpp 0.99789
  to 0.99869). The encoder runs on the best CPU backend; where the
  GPU imports host memory (Apple, MoltenVK), a pass of at least 100
  tokens and 1e10 multiply-adds runs on the Vulkan image ops instead
  (the dense ops PixArt uses: one batched matvec per matrix), since
  below that the GPU's fixed cost per call wins. For the same reason
  an input of at most 192 tokens has its attention on the CPU, a
  longer one on the GPU. On an M2 Pro that makes 32 inputs of 120
  tokens 3.9x faster with bge-m3 (18.2 s to 4.7 s), 3.7x with nomic
  v1.5 and 1.5x with bge-small, and a 50-document rerank 3.1x (12.8 s
  to 4.1 s). Elsewhere, with a usable CUDA device, those passes run
  on the CUDA ops: the matrices through the batched CUDA matvec, each
  weight read once per 32 tokens, and the attention on the CPU, since
  dense::attention keeps a GPU that caches weights by host pointer
  off its temporary buffers. The image classifier's convs stay on
  Vulkan only. Without CUDA, a discrete Vulkan GPU runs them on the
  Vulkan image ops, the weights copied into its pool (under 700 MB);
  V3D and llvmpipe keep the CPU, which is faster there (i#680). Otherwise
  `/v1/embeddings` returns the chat model's last token's final-normed
  hidden state, L2-normalized, on a dedicated cache; CPU and CUDA
  backends only (`TransformerModel::embed` refuses vulkan).
- Rerank: with `--rerank-model`, a GGUF of arch `bert` with
  pooling_type 4 (rank) serves `/v1/rerank` (also `/rerank`): each
  document is paired with the query as llama-server does (`<s> q
  </s></s> d </s>` for XLM-R, a document cut to fit the context), the
  pairs run as one batch, and the score is the raw logit of the
  classifier on the CLS state (`cls`, tanh, `cls.output`). Results come
  best first, `top_n` of them. On bge-reranker-v2-m3 Q8_0 the scores
  match float64 arithmetic on the same weights to 1e-5 and the f32
  original to 0.09, and rank the documents as the original does;
  llama.cpp drifts up to 0.3. The router gives `/v1/rerank` its own
  kind, `rerank`. ggml-org has no GGUF of the model, so
  `scripts/convert-xlmr-reranker.py` makes one from BAAI's safetensors,
  with the vocab of bge-m3's GGUF (the same SentencePiece model).
- API: `/v1/models`, `/metrics` (Prometheus), OpenAI-shaped logprobs
  (a log-softmax over the raw logits) and usage accounting.
- Metrics: every `/metrics` series carries `model` and `backend`
  labels, and `locus_info` adds `arch`, `chat_template`, `kv_type`
  and the hardware (`cpu`, `gpu`); `locus_device_info{kind,name}`
  lists the CPU and each GPU in use. Join on these with `group_left`
  instead of a hardware label on every series.
  Request counters and KV gauges sit beside per-request histograms
  (`locus_queue_seconds`, `locus_prefill_seconds`,
  `locus_time_to_first_token_seconds`, `locus_decode_token_seconds`,
  `locus_request_seconds`), token counters for prefill and decode, the
  decode wall time, the engine's step time
  (`locus_engine_busy_seconds_total`,
  `locus_engine_steps_total`) and the running and waiting counts.
  After each step the engine also sets `locus_batch_tokens` (the
  tokens the model computed in it: prefill, decode and speculative
  verify; 0 once the engine is idle) and `locus_kv_blocks_used{state}`: each block in use counted
  once, as held by a sequence still prefilling, else by one decoding,
  else only by the prefix cache, so the three and
  `locus_kv_blocks_free` sum to `locus_kv_blocks_total`. The engine
  writes them as relaxed atomics, a few clock reads per request and
  two per step, plus one pass over the pool's blocks per step, and
  `/metrics` reads them without the engine lock;
  decode speed is unchanged within noise. The HTTP layer adds, per
  route (the server's routes, anything else as `other`):
  `locus_http_requests_total{route,code}` for every answered request
  (head-stage rejects and parse or deadline errors included), the
  request and response body sizes (`locus_http_request_bytes`,
  `locus_http_response_bytes`, stream payload included), the time from
  head received to response written (`locus_http_request_seconds`)
  and `locus_http_requests_in_flight`. A request costs a route lookup
  and a few relaxed adds, about 27 ns on the M2 Pro. The forward pass
  charges its host time to `locus_stage_seconds_total{stage}` (embed,
  norm, attention, ffn, moe, layer, prefetch, head, sample) and, for
  the per-layer stages, `locus_layer_seconds_total{layer,stage}`;
  `layer` is an architecture's own whole-layer forward that is not
  split further. One clock read per stage boundary on every step, so
  it is always on; decode speed on llama-3.2-1b Q8_0 is unchanged
  within noise. These are host times. Vulkan runs the whole model as
  one submit, so its time goes to `stage="device"`; CUDA works per op,
  so a stage's time includes its wait for the device. Each GPU API
  the process used adds, with an `api` label: `locus_gpu_submits_total`
  (Vulkan batches, or CUDA synchronizing calls),
  `locus_gpu_dispatches_total` (Vulkan kernels),
  `locus_gpu_record_seconds_total` (Vulkan batch begin to submit),
  `locus_gpu_wait_seconds_total` (submit until the device is done),
  `locus_gpu_device_seconds_total` (GPU timestamps on a random 1 in 16
  Vulkan batches, scaled by 16; every batch under `LOCUS_VK_PROFILE`;
  an estimate that needs many batches, so a short request may miss its
  prefill), `locus_gpu_device_samples_total` (the batches timed),
  `locus_gpu_device_stage_seconds_total{stage}` (Vulkan device time
  per forward stage, the same stages as the host series, from a
  timestamp at each stage's end on 1 timed batch in 4, scaled by 64
  (a MoE layer routed on the host splits a forward into several
  batches, each sampled on its own);
  `other` is device work outside a stage, such as the image and video
  models; it tracks the device total over many batches),
  `locus_gpu_device_stage_samples_total` and
  `locus_gpu_memory_bytes{kind}` (weights, mapped, kv, scratch, table;
  `mapped` is weights the device reads in place from the file map).
  Sampling matters on MoltenVK, where a timestamp splits the
  compute encoder: on every batch it cost about 0.5% of decode speed,
  and stage marks on every timed batch about 1.3%. A stage timestamp
  there needs a barrier before it, or it does not wait for the
  kernels before it and the time lands in the wrong stage.
  `locus_fallbacks_total{from,to,reason}` counts work a GPU path
  handed to a slower one: `to="per_op"` (Vulkan's whole-model forward
  declined, so the host steps each layer), `per_token` (batched
  prefill declined), `host_route` (MoE router on the host, per MoE
  layer and token) and `cpu` (a kernel missing, a device error or an
  unsupported shape). Only fallbacks that happened have a series.
  The multi-sequence Vulkan decode declining is not counted: its
  sequences still run on the GPU, one forward each.
  `locus_moe_expert_calls_total{layer,expert,device,api}` counts each
  routed expert run: `device` is `gpu` or `cpu`, `api` is `cuda`,
  `vulkan`, `moltenvk` or the CPU kernel set (`neon`, `avx2`, `sse4`,
  `scalar`). Vulkan's GPU-routed MoE layers keep the picks on the
  device and are not counted.
- KV-cache quantization: Q8 or Q4 blocks of 32 (f16 scale) in a
  separate byte pool, for the llama-family attention. Each layer's
  past rows are dequantized into scratch, so the attention math is the
  f32 path's. Checked by cosine similarity (Q8 above 0.999). F16 KV
  also runs on the GPU for GQA models. MLA already shrinks the KV
  cache, so its quantized path is low priority (#63).
- More architectures and quant types: qwen2moe, qwen3moe, dbrx, glm-
  dsa, muse-spark, deepseek41, kolibri1; ternary TQ1_0 / TQ2_0 and
  the IQ family.
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
across a stage chain. A stage runs on the CPU, on CUDA or on Vulkan
(`--backend vulkan`, #48): on Vulkan its slice's KV cache holds only
its layers, a lone session runs the single-sequence kernels and a
batch of up to 8 the multi-sequence ones (dense llama-family models;
others run one GPU forward per session). With one `--device` per
executor, each Vulkan executor runs on its own device, with its own
backend state (#47). Activations travel as
length-prefixed frames over TCP, tagged by request and position. A
connect proves only that the downstream listener is up; liveness needs
an app-level probe.
The serve loop can report its live peers (each session's upstream
and downstream) with the kernel's smoothed TCP round-trip time, read
from `TCP_INFO` (Linux, FreeBSD) or `TCP_CONNECTION_INFO` (macOS)
with no probe of its own (#21); `locus-stage --peer-log N` prints
them every N seconds. `locus-stage --metrics-listen HOST:PORT` serves
them as Prometheus text on GET `/metrics`, from the same loop through
`http::LoopConnection`, with the session, token, byte and downstream
dial counters (#22). Each stage serves its own page, so a scraper
reads every stage directly; the driver does not gather them. The
`--allow` list does not cover that port.
A connection kept open across sequences (SESSION_END) dials its next
downstream when the next sequence sends its first frame, not at the
reset: a last stage's downstream is a driver, and a reset-time dial
reached the driver that had just exited, or the next one before any of
its sessions (#459). So a client sends its first token before it
accepts the logits connection.

A node separates I/O from compute (#24):

- One I/O thread owns every socket (`sys::Poller`), decodes frames and
  marks sessions ready. It never blocks on compute.
- An executor is one compute device running a continuous-batching
  loop: each iteration it gathers the ready sessions, runs one batched
  forward (`forward_batch_layers`), and returns the outputs. It owns
  that device's paged KV pool. With K > 1 CPU executors each has its
  own thread pool of cores / K threads (#45; `sys::ThreadPool::Scope`
  makes it current on the executor's thread and on its workers), so
  their loops do not serialize on one pool and a lone request still
  fans out.
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

The sse4 and CUDA dots, and the neon block-32 dots, replay the scalar
accumulation order, so they are bit-identical to it. The neon Q4_K
and Q6_K dots do not (below). The build turns off implicit fused
multiply-add (`-ffp-contract=off`), so the scalar path rounds the same
on x86-64 and arm64, and a stage chain that mixes the two is exact
(#458).

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

Prefill and decode take the same gate (`uses_quantized_activation`).
For prefill and batched decode, `Ops::matvec_q8k_batch` (neon)
quantizes each token once and dots each weight row against four
tokens at a time while it is in cache. The neon Q4_K and Q6_K kernels
unpack each block once and use sdot with one float step per block;
single-token decode runs the same kernel with one token, so a token's
value is the same alone or in a batch (batch-invariant), and within
f32 rounding of the scalar order per dot (about 1e-8 of the sum of
the terms' magnitudes, the same as the scalar order's own error).
Through a whole model those roundings can tip 8-bit activation
quants, so logits move by up to a few tenths against the scalar
path. The block-32 kernels replay the per-token order. On the M4 Max
this takes Orpheus 3B Q4_K_M prefill from 97 to 182 tok/s and decode
from 80 to 85 tok/s. F32, F16 and BF16 weights always take the f32
dot; on neon F16 and BF16 widen to f32 in registers, 8 weights a step
(the T5-XXL F16 prompt encode on an M2 Pro: 8.8 to 5.9 s).

Goldens: Qwen1.5-MoE and DeepSeek-V2-Lite are pinned to f32, because
8-bit activations tip their greedy text at a near-tie. Muse Spark's
second prompt, DeepSeek-V4.1 and the Orpheus codec frame are checked
teacher-forced: each reference token must be within 0.25 of the top
logit (0.75 at DeepSeek-V4.1's one known near-tie, #467).

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
