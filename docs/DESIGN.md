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
  is GPU-mapped, so attention reads K and V where they are written. On
  Metal it is a shared buffer, and the GPU forward takes f16 or f32;
  q8 and q4 run the per-op path there. f16 (K and V rounded to halves
  as written, read back as float; half the attention's reads) is the
  tools' default from Apple family 9 (i#1245: decode at 2k positions 5
  to 7% faster on the M4 Max); below it (the M2 Pro: -4% to +2%) and
  on the other backends f32 is.

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
`LOCUS_BACKEND`, which beats the default: Metal on a Mac and CUDA on a
host with a usable CUDA device when that GPU's whole-token forward
takes the model (#867, #1253), else the best available (the first
usable entry, a CPU one). When that default leaves a usable Vulkan
GPU that beats the CPU idle, locus-run and locus-server print a
warning naming the `--backend` to pass. Every variant must
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
no allocation sized by an unvalidated field; hand-written hostile
files in the tests (`[gguf]`: truncations, oversized arrays, hostile
tensor tables; a fuzz target is not in the tree yet, #625); mmap
read-only.

### 4.5 HTTP server

A first-party HTTP/1.1 server (`src/locus/http`) and vendored
nlohmann/json for untrusted request bodies: hand-rolled JSON parsing
is where API vulnerabilities come from. A worker thread drives
`Engine::step()`; handler threads submit and wait (`EngineLoop`),
asleep until the worker publishes their request's progress after a
step. A followed request (`follow`, #989) also gets its new tokens
queued after each step, for `take()` to collect without the engine
mutex, and a progress hook to wake whoever drives it.
Chat messages are formatted by `chat::ChatTemplate`: the GGUF's
Jinja source is fingerprinted to a known family (llama3, chatml,
llama2 and its -sys, -sys-bos and -sys-strip variants, Mistral's
mistral-v1, -v3, -v3-tekken, -v7 and -v7-tekken (#1204), zephyr,
deepseek, deepseek2, deepseek-coder; llama.cpp's names and rules) and
rendered by a hardcoded formatter, string for string as llama.cpp's
`llm_chat_apply_template`; an unknown template falls back
to plain concatenation. `locus-server --chat-template NAME` picks a
family for a GGUF that has no template (older Llama 2 conversions,
#1152) or whose template reads as another family (a Mistral GGUF with
the old `'[INST] '` template is llama2 by llama.cpp's rules too).
Endpoints that run the model on their own
thread (images, vision chat, audio) share the GPU backend through one
process-wide lock (#268). `http::Client` is the first-party client
for locus's own upstream calls (the router of #388): a pure response
parser with the request parser's hardening, a streaming chunk
decoder, and connect and read deadlines with one kept-alive
connection, resent once when the server closed it while idle. An
optional total deadline per request bounds a server that sends a byte
just under each read deadline.
`http::LoopConnection` serves HTTP on a caller's own event loop
(`sys::Poller`) instead of a thread per connection: the loop accepts
and owns the fd, the connection reads, parses, calls a non-blocking
handler and writes, and says what to watch next; for small endpoints
on a thread that already runs a loop, such as a stage's `/metrics`
(#22). An async handler answers later, from any thread, through an
`http::LoopReply` (#930): one buffered response, or a chunked stream
whose chunks queue up to a byte bound (the producer waits, for at most
the write deadline, while the socket lags); each call wakes the loop
through a pipe, and a client that leaves cancels the reply.
`http::LoopServer` (#931; `locus-server --event-loop`) serves the
server's routes that way: one I/O thread owns every connection (the
`allow` list and the per-address cap at accept, `/health` answered on
the loop; `/metrics` waits up to an engine step for its gauges, which
would stall every connection there), and a job pool runs the head
stage and the handler, a stream through a `Sink` that blocks at the
reply's queue bound. An idle or slow client then holds a buffer, not a
thread. A reply names its connection when it wakes the loop
(`http::WakeList`) and deadlines sit in a min-heap (#990), so the I/O
thread's work per chunk does not grow with the connections it holds.
A generation (completions, chat, messages, responses; #932)
gives its job thread back after submit (`RequestContext::defer`): one
`server::GenPump` thread, woken by the engines' progress hook, takes
each request's new tokens (`EngineLoop::take`) and queues its SSE
chunks or body on the reply, through the same `GenReply` steps the
thread path runs, so N streams take no thread each. A stream over its
queue bound is skipped until its reply calls back (queue taken,
cancelled or half-closed). The pump looks only at the requests the
progress hook names, the replies that call back and the due keepalives
(#1005). The known limit: that one thread decodes and builds the JSON
of every stream in turn.
`http::Server` (a thread a connection) stays the default for now.

Guards behind the router (i#1161): `--guard` names guard instances
(locus-servers with guards; host:port or `srv:NAME`, never discovered,
since a guard decides what passes). Each chat, completions, messages
or responses request (text or image-to-text) then goes first to one of
them, `POST /v1/guard/{api}` with the client's body, in rendezvous
order by the session key (a conversation keeps its guard's prefix
cache), the next after a connect failure, a busy 503, no reply within
`--guard-timeout` (300 s; a chat's 30 minutes would hold the request
behind a hung guard) or any reply that is not a verdict. Set it above
the guard's cold check time (121 s for a 2100-token chat with a 1B
llama-guard on the SSE4 CPU, 36 s on CUDA; Prompt Guard at 64 windows
about 15 s) and lower it for fast guards, to fail over sooner. The guard
endpoint marks each reply it writes with `X-Locus-Guard: verdict`; the
router allows only on a marked 200 `{"allow": true}` and relays only a
marked 400 or 500, so a proxy or another service at a guard's address
never decides what passes. The guard instance parses the body
as its own endpoint does and answers 200 `{"allow": true}`, or that
endpoint's
own 400 (flagged) or 500 (failed guard), which the router relays and
does not forward. None answering is a 503: the check fails closed. A
check runs on the guard instance's engine worker, not the chat one,
so a chat instance's streams do not wait for it.
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
and then sends nothing holds a probe for the 2 s read timeout (3 s in
all for one that trickles bytes), so a range full of them makes a scan
last about 1024 / 64 x 2.3 s = 37 s. Each refresh polls the instances'
`/v1/models` in parallel, 64 at a time, each poll capped at 8 s in
all, so a slow instance delays only its own poll.
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
itself (`locus-server --register`, with a `--host` the router can
reach, since it listens on 127.0.0.1 by default): it POSTs its host:port to
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
- Vendored in-tree, pinned, every file's sha256 in
  `third_party/SHA256SUMS` (the `[third_party]` test checks the files
  against it and that each vendored file is listed), license files
  alongside: Catch2 v3.7.1 (BSL-1.0, tests only),
  cpp-httplib v0.18.3 (MIT, tests only: their HTTP client),
  nlohmann/json v3.11.3 (MIT), ggml's
  IQ lookup tables (`third_party/ggml/iq_grids.h`, MIT) and llama.cpp's
  Unicode tables (`third_party/llama/unicode-data.{h,cpp}`, MIT, commit
  cd959bae9). Ported, not vendored: ggml's x86 AVX Q4_K / Q6_K dots
  (`backend/matvec_sse4.cpp`, the AVX one-token rows, from llama.cpp
  e8e6c7a), under the same MIT notice (`third_party/ggml/LICENSE`).
- No FetchContent and no network at configure or build time.
- Optional system dependencies, found at configure time: the Vulkan
  loader and a shader compiler, the CUDA toolkit. On CUDA builds, the
  toolkit's cuBLAS too (nothing vendored): it runs the prefill GEMM of
  chunks of 32 tokens or more (#1200); a toolkit without it builds the
  CUDA backend with the tile kernels only, and configure says so.

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
`layer expert`; `scripts/text/moe-trace-sim.py` replays a trace against
cache policies (LRU, LFU, Belady, step WLRU, static top) for a given
GPU budget. On Vulkan both variables route MoE layers on the host, so
the picks reach the CPU.

On Vulkan, when the experts do not fit the device (over 80% of its
available memory), each MoE layer routes on the host and the CPU runs
the picks from the file map, adding them into the residual stream it
read back; attention, the dense layers and the shared experts stay on
the GPU (i#1174). Uploading the picks each token instead
(`LOCUS_VK_HOST_EXPERTS=0`) costs PCIe bandwidth for nothing:
Qwen1.5-MoE on a 2 GB GTX 750 Ti decodes at 0.1 tok/s that way and
3.9 with the CPU picks. Routed on the host for another reason (a pool
budget, no batched kernel, a bias the GPU router lacks), the fit is
still checked, and misfitting picks run on the CPU too (i#1178);
experts that fit upload as before.

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
  fits stays resident. On a discrete GPU under Vulkan, a weight that
  misses the device heap (the driver falls back to pinned system
  memory) is a spill: the device budget drops to what the pool holds
  there, and spills are cached apart, under a quarter of the host's
  RAM (`LOCUS_GPU_SPILL_MB` overrides) with their own LRU; past that a
  spill serves one batch, so pinned memory stays under the cap plus
  one batch's overflow. A reset between models clears the spill state
  with the buffers. Once the device is full a miss goes to the
  spill cache first, since evicting device weights read every token
  would make each eviction the next miss. So a dense model a little
  larger than the device stays resident (Orpheus 3B Q4_K_M on a 2 GB
  GTX 750 Ti: 6.3 tok/s, as before), and an MoE's experts pin at most
  the cap (i#1164: they pinned 7.4 GB of a 16 GB host until a pin
  failed). Unified memory (V3D, MoltenVK) keeps the unbounded default.
  Weights stay quantized on the device for every type whose kernel
  dequantizes in-shader (Q4_K is about 0.5 bytes per weight).
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
- Where the device's mappable memory is not its own (a discrete GPU:
  the first HOST_VISIBLE, HOST_COHERENT type is not DEVICE_LOCAL),
  weights and tables live in DEVICE_LOCAL memory. The upload writes a
  host-visible staging buffer, the next dispatch records its copy, and
  the staging buffer is freed when that batch ends; a full device
  falls back to mappable memory. Activations, KV and scratch stay
  mappable. GTX 750 Ti, Llama 3.2 1B decode: Q8_0 4.8 to 25.2 tok/s,
  Q4_K_M 6.9 to 23.6 (#1121). `LOCUS_VK_HOST_WEIGHTS=1` keeps the
  weights in mappable memory.
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
  `LOCUS_VK_NO_IMPORT=1` keeps the copies; the NVIDIA proprietary
  driver always does, as it fails to import a file mapping and then
  every later allocation (#1122). Unmapping a model file
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
  is the same code, so today logits are bit-identical for every thread
  count (not required: section 16). `LOCUS_THREADS` caps the fan-out
  (default: the performance cores; 1 runs single-threaded). Parallel
  slices also issue parallel page faults, a deeper NVMe queue on
  streamed models. Independent
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
`forward_batch` gives a KV cache and logits equal to N `forward()`
calls within the matmul tolerance (byte-identical where the kernels
share an order, and always in the batch-invariant mode; section 16).
Attention reads no large weights, so it stays per token, each against
its own sequence's KV.
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
  default on Apple GPUs, on V3D and on NVIDIA GPUs, where a lone
  sequence costs little (on V3D 2.3% for a lone Q8_0 sequence with the
  NT 1 build below; on a GTX 750 Ti a lone Q8_0 decode is 6% faster,
  #977); elsewhere a lone sequence keeps the single-sequence forward.
  On V3D the kernel's v3d driver can reset the GPU for a "hang" in the
  middle of a long compute job (a 274 ms conv dispatch, about once in
  30), and the job's remaining workgroups never run, without an error
  (i#682). So there a grid of more than 32 workgroups is recorded as
  several `vkCmdDispatchBase` pieces, each its own short job, except
  for kernels that read `gl_NumWorkGroups` (found in their SPIR-V);
  the pieces see the same workgroup ids, so the results are the same.
  The one-token matvec kernels take a budget of 1024, as their
  workgroups are short (32 cost a 1B decode 10%; 1024 costs nothing
  measurable), and the multi-token builds less, by their work per
  workgroup (512 for NT 2, 256 for NT 4, 128 for NT 8 and the batched,
  pair and shared-expert ones).
  `LOCUS_VK_SPLIT=N` sets one budget for every kernel, 0 turns it off.
  Under the default a lone sequence on a Q4_K or Q6_K model keeps the
  single-sequence forward too (the K-quant short-batch kernels cost it
  9 to 11% on an M2 Pro), so such a model is batch-invariant only with
  the mode set.
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
  On Metal it runs `metal_forward_multi` on groups of up to 8
  sequences (#1184; dense models, as `metal_forward` takes them; a MoE
  model runs one forward per sequence), one command batch for the
  whole step. Each matrix takes the fastest kernel for its type and
  the sequence count:
  - Q8_0: `mv_q8_0_cols{2,4}` reads it once for all the rows (5 to 8
    columns as two dispatches), with the RMSNorm folded in as a
    prologue;
  - Q6_K, from 4 sequences: `mv_q6_k_cols{4,8}`, each block's x for
    all the columns staged in threadgroup memory (at 2 its kernel
    loses to 2 matvecs);
  - Q4_K, and the rest: one matvec a row. No Q4_K columns kernel tried
    beats N matvecs: on the M4 Max a K-quant decode matvec is bound by
    its x loads and decode, not its weight bytes (#1191).

  Each sequence takes its own RoPE position, K / V rows (written
  straight into its blocks when alone, else copied by `kv_scatter`)
  and attention. A column's sum runs in the same order whatever the
  column count, and the Q8_0 kernel gives a column `mv_q8_0`'s own
  arithmetic (the same lanes, block order and `half_simd_sum`, the
  folded norm's scale applied before the sum as `mv_q8_0_pn` does), so
  its bits are a matvec's and a lone row runs the matvec itself.
  Batch-invariant mode (`TransformerModel::metal_batch_invariant`)
  sends a lone sequence through `metal_forward_multi` too, as
  `forward()`, a whole-model `forward_layers` and `forward_batch`'s
  chunks too short for the prefill, and keeps every matrix on one
  kernel at any count (Q6_K on its matvec), so a request's logits are
  bit-identical alone and batched; the disk prefix key records it. It
  is the default for a model with no Q4_K or Q6_K weights, at no cost
  measurable to a lone sequence on the M4 Max (`metal_forward`'s RoPE
  epilogue and F32 fold against `metal_forward_multi`'s direct K / V
  write). For a K-quant model, as on Vulkan, the default takes the
  fastest kernels instead: `metal_forward` alone, Q6_K's columns
  kernel from 4. `LOCUS_METAL_BATCH_INVARIANT=1` / `0` (or
  `backend::metal::set_batch_invariant`) sets it for any model. Llama
  3.2 1B on an M4 Max, decode tok/s in all:

| model  | sequences | one forward a sequence | default        | invariant set |
|--------|-----------|------------------------|----------------|---------------|
| Q8_0   | 1         | 223.6                  | 224.8 (inv.)   | 224.8         |
| Q8_0   | 4         | 223.8                  | 689.2 (inv.)   | 689.2         |
| Q8_0   | 8         | 224.1                  | 780.5 (inv.)   | 780.5         |
| Q4_K_M | 1         | 300.0                  | 301.2          | 296.2         |
| Q4_K_M | 4         | 300.7                  | 472.3          | 449.4         |
| Q4_K_M | 8         | 300.4                  | 547.2          | 494.2         |

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
keep matvec's per-block reduction order, so today each token's dot is
bitwise unchanged (a single end-of-row reduction would differ: float
addition is not associative); that is a property of these kernels, not
a requirement (section 16).

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
  mtime, the backend and device, a digest of what computes KV
  (cmake/compute_digest.cmake over every source under src/locus,
  CMakeLists.txt and the build's compiler, version, build type, C++
  flags and shader compiler, made at build time: another build may sum
  in another order, so an upgrade or a rebuild with other flags
  misses; i#1180), the KV geometry, RoPE settings and
  the numeric modes that change KV bits (the LOCUS_* switches that
  pick other kernels among them, LOCUS_CPU_FAST too, and a prefill
  budget other than the model's own chunk, which sets a prompt's chunks
  and on the GPUs moves KV's last bits, #1189),
  and on Vulkan where the MoE
  experts run: routed on the GPU, or on the host with the picks on the
  GPU or on the CPU (i#1179), settled once the KV pool exists; a model with
  the MoE hybrid on is refused, since its hot set follows the requests
  served before, and so are remote experts, #1185),
  the tokens and a checksum per block, so a mismatch
  or a torn file reads as a miss; a byte cap evicts the least recently
  used file. The checksums catch damage, not tampering, so the
  directory must be the server user's own and not group- or
  world-writable (else startup fails), and symlinks in it are never
  followed. With `--prefix-cache-key-file`, each file (or helper blob)
  also ends with an HMAC-SHA-256 of its bytes, so one written without
  the key is a miss: for a store others can write (#830). Files of
  another key stay until the byte cap evicts them; the server prints
  its key's hash at boot (`prefix cache key <16 hex>`, the checksum of
  the key's header, which seeds every file name), and
  `locus-prefix-clean --keep KEY [--keep KEY ...] DIR` removes every
  file whose header hashes to none of them (one KEY for each server
  that shares DIR; `--dry-run` lists them). It never follows a
  symlink (DIR, an entry or a file), and leaves files whose header it
  cannot parse alone (i#1180). Entries
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
  Prefix hits, in RAM and on disk, are exact only up to the prefill's
  chunking: after a hit of k tokens the rest is chunked from k, and on
  Metal, Vulkan and CUDA (whose chunks of 32 tokens or more run a cuBLAS
  GEMM, #1200) other chunks give other last bits of KV than a fresh
  prefill of the same prompt (within batched prefill's tolerance of the
  per-token forward, i#730). Where a prefill resumes depends on what
  was cached before, so no key can hold it (#1189). The neon backend's
  prefill gave the same bits at every chunking measured.
- Prefix cache helper (`--prefix-cache-helper PATH`, i#655): the same
  tier kept by an external process, so the KV can live on remote or
  shared storage. Locus starts the helper and speaks the auth helper's
  text records (`req_id=`, `type=`, `key=value` lines, a blank line)
  on its stdin and stdout. Each prefix is one blob, byte for byte the
  disk tier's file, named by its 16 hex digit digest. The bytes go
  through a POSIX shm object (`/locus-pfx-<pid>-<seq>` in hex, 0600)
  that locus creates per call and unlinks after the reply, never
  through the pipe:

  | request                       | reply                   |
  |-------------------------------|-------------------------|
  | `list`                        | `names=<name> <name>..` |
  | `read name= shm= bytes=`      | `ok=1` (blob in shm)    |
  | `write name= shm= bytes=`     | `ok=1` (blob stored)    |
  | `evict name=` (`req_id=0`)    | none                    |

  Locus checks every blob it reads as it checks a file (key, tokens,
  checksums), so a wrong or stale blob is a miss and is evicted; the
  helper owns placement, eviction and the remote protocol. Lookups
  use a local set of names: the helper's list, read at start and
  again every 60 s (so another process's prefixes are found), plus
  this process's writes, minus reads that missed. A call over 120 s
  misses. A helper that dies is started again once; when that one
  dies too, the tier turns off with a warning. `locus-prefix-helper
  DIR [CAP_MIB]` is a reference helper that keeps the blobs as files
  in DIR, which the disk tier can also open.
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
  `scripts/text/convert-xlmr-reranker.py` makes one from BAAI's safetensors,
  with the vocab of bge-m3's GGUF (the same SentencePiece model).
- DeBERTa classifier: `embed::DebertaEncoder` runs a DeBERTa v3
  sequence classifier, GGUF arch `deberta-v2`, which
  `scripts/text/convert-deberta.py` makes from transformers'
  safetensors (llama.cpp has no such arch); the tokenizer is its
  SentencePiece unigram model, written as the `t5` (UGM) keys. It is
  BERT's post-norm encoder without absolute positions: each head adds
  to $q_i \cdot k_j$ the content-to-position and position-to-content
  terms $q_i \cdot pk_c$ and $k_j \cdot pq_c$, where $pk$ and $pq$ are
  the layer's own k and q projections of the normalized relative
  embeddings and $c$ the log bucket of $i - j$, all over
  $\sqrt{3 d}$. Only the relative rows the longest input reaches are
  projected, so a short input does not pay for all 512. Inputs run as
  one batch (attention within each), and longer than the trained 512
  tokens, as transformers allows; the head is the pooler (the CLS
  state through `cls` and GELU) then `cls.output`. On
  protectai/deberta-v3-base-prompt-injection-v2 the logits match
  float64 arithmetic on the same Q8_0 weights to 1.1e-5 and, from the
  F32 GGUF, the f32 original to 2.1e-5; Q8_0 itself moves them up to
  0.05. It serves as the `prompt-guard` guard (below).
- Guards: `server/guard.hpp`'s `Guard` judges a batch of requests,
  one verdict each (allow, reject with labels, or failed); each kind
  implements it. A server runs several (`--guard-model` first, then
  each `--guard` in order) and stops at the first that rejects; a
  failed guard rejects. Each guard's `GuardBatcher` joins concurrent
  checks: a lone check runs at once, and the checks that arrive while
  a batch runs form the next one. `/metrics` counts per guard
  (`locus_guard_checks_total`, `_failures_total`,
  `_check_seconds_total`, `_flagged_total{category}`, each with a
  `guard` label).
- Llama Guard 3 (`kind=llama-guard`): the GGUF (llama arch)
  loads on the chat model's backend with its own engine and KV pool
  (`--ctx`, else 4096 positions). Each chat request's turns become
  Llama Guard 3's prompt (`server/guard.cpp`), encoded with the
  messages as literal text, so a typed `<|eot_id|>` cannot close the
  guard's turn and write its verdict. A batch's prompts go to the
  engine together. The guard decodes greedily up to
  its EOS or `<|eot_id|>` (at most 64 tokens) on the chat engine's
  worker thread, between steps (`EngineLoop::run_on_worker`), so no
  backend runs on two threads; the chat engine waits while it runs.
  Its prefix cache keeps a chat's earlier turns: the conversation part
  of the prompt runs first as its own request, so a later turn
  prefills only the new turn and the closing instruction (on a
  2100-token chat, a Llama 3.2 1B stand-in: 121 s cold and 9.6 s for
  the next turn on an SSE4 CPU, 36 s and 3.4 s on CUDA). As with the
  chat engine's cache, every client shares it, so a fast check tells a
  client that someone sent the same conversation start. Only "safe", or
  "unsafe" and a line of codes, is a verdict; anything else, a reply
  cut at the cap, or a failed run rejects (fail closed).
- Qwen3Guard-Gen (`kind=qwen3-guard`, i#1106): the same generating
  guard with another `GuardFormat`. The GGUF (dense `qwen3` arch) gets
  its `tokenizer.chat_template`'s prompt for a last user turn, built in:
  the task, the safety policy, nine named categories and the
  conversation as `USER:` and `ASSISTANT:` lines (a system turn too, as
  the template has it), then an empty `<think>` block. It stops at
  `<|im_end|>`. Its verdict is `Safety: Safe|Unsafe|Controversial` and
  `Categories:` with names; Controversial rejects, whatever `labels=`
  names, unless `controversial=allow` (Qwen's strict and loose modes).
  On the x86
  host the 0.6B Q8_0 guard prefills its 300-token prompt at 49 tok/s on
  the CPU and 111 on CUDA (per op), and the prefix cache serves the
  fixed head after the first check.
- Prompt Guard (`kind=prompt-guard`, `server/classifier_guard.cpp`):
  a sequence classifier on the DeBERTa encoder, scoring the untrusted
  text of a request: every user turn (tool results arrive as user
  turns), not the system or assistant turns. Each turn is encoded
  without specials (typed `[CLS]` stays text) and cut into windows of
  the model's context, 510 tokens between `[CLS]` and `[SEP]`, the
  next starting 64 tokens before the last ends, so a phrase cut at one
  window's end lies whole in the next. Every window of the batch's
  requests is classified (16 per call), a label's score is its highest
  over the request's windows, through softmax or sigmoid as the GGUF's
  `classifier.problem_type` says (absent: as transformers' pipeline,
  softmax over several labels, a sigmoid on one; regression is
  refused), and a label rejects when its score reaches its threshold
  (`labels=INJECTION@0.9:JAILBREAK`, `threshold=` the default, 0.5).
  The labels come from the GGUF (Prompt Guard and Prompt Guard 2
  differ); `labels=` is required, a name the model lacks is refused
  at start, and a threshold must lie in (0, 1]. The encoder runs on
  the best CPU backend on the chat engine's worker, so the chat
  engine waits while it runs: 0.24 s per window for a DeBERTa v3 base
  on an M4 Max, so a request of more than 64 windows (about 29k
  tokens) fails closed rather than stall it longer. Prompt Guard
  itself is gated behind Meta's license; the tests use
  protectai/deberta-v3-base-prompt-injection-v2 (labels SAFE and
  INJECTION), the same layout. `max_windows=` sets the cap per guard.
- Encoder classifier (`kind=classifier`, `load_encoder_classifier`):
  the same guard over a BERT, XLM-R or nomic-bert sequence
  classifier, the reranker head with N outputs (the CLS state through
  the pooler `cls` and tanh, then `cls.output`). Labels come from
  `{arch}.classifier.output_labels` (llama.cpp's key; LABEL_0, ...
  when absent), the problem type from `{arch}.classifier.problem_type`;
  text is encoded literally (the plain encode parses typed special
  tokens since #1151), windows between the model's single-sequence
  first and last tokens. A head without `cls` is refused: llama.cpp's
  converter drops BertForSequenceClassification's pooler, so its
  scores are not the model's; `scripts/text/convert-bert-classifier.py`
  keeps it. On unitary/toxic-bert the F32 GGUF's logits match
  transformers' float32 within 1e-3 and the Q8_0's within 0.05
  (`tests/golden/toxic_bert_hf.json`); a llama.cpp reranker
  (bge-reranker-v2-m3) loads as one label.
- Linear head (`kind=linear-head`, `load_linear_head`): the same
  guard over an embedding model (`embedding=`, a bge, e5 or nomic
  GGUF) and a linear head on its vectors, for an operator's own
  labels. SetFit, the usual recipe, stores its head only as a pickle
  (`model_head.pkl`, joblib, for its scikit-learn and its torch head
  alike), which locus never loads; so the head is a GGUF of arch
  `linear-head`: `cls.output.weight` (labels x width) and `.bias`,
  `classifier.output_labels`, `classifier.problem_type`, a
  `temperature`, and the embedding model it was trained on. Each
  window is embedded as `/v1/embeddings` does (pooled,
  L2-normalized) and scored as (W v + b) / temperature. A head whose
  embedding model's `general.name`, width, pooling or file size
  differ from `embedding=`'s is refused (its sha256 is recorded, not
  checked at every start). `locus-embed` embeds a JSONL of labeled
  examples with the server's own code (the training sandbox has no
  network to call the server), and `scripts/text/train-linear-head.py`
  fits scikit-learn's logistic-regression objective (multinomial, or
  a sigmoid per label; L2 with C = 1) by L-BFGS in torch, with no
  scikit-learn. On 60 hand-written examples (prompt injections and
  benign requests) with bge-small, 11 of 12 held out are right, and
  the guard scores them as the script does to 1e-4.
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
  `locus_kv_blocks_free` sum to `locus_kv_blocks_total`. Each forward
  call adds one observation to `locus_step_tokens{phase}` and
  `locus_step_seconds{phase}`: a prompt chunk's tokens (`prefill`),
  the requests in a decode call (`decode`), or a verify's drafts plus
  one (`verify`), with its wall time; a phase not seen yet has no
  series. So a decode slowdown does not hide behind a prefill in the
  per-layer counters. The engine
  writes them as relaxed atomics, a few clock reads per request, two
  per step and two per forward call, plus one pass over the pool's
  blocks per step, and
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
  device and are not counted; picks its host-routed layers run on the
  CPU count as `cpu` with the CPU kernel set. Per expert with the same
  labels, `locus_moe_expert_runs_total` counts weight passes (a
  batched prefill run over several tokens once) and
  `locus_moe_expert_seconds_total` their host-clock time, so
  calls / runs is the tokens a run and seconds / runs its time; only
  CPU and CUDA runs (a CUDA matvec blocks on its copy back), since a
  Vulkan or Metal expert is recorded into a command buffer.
  `locus_moe_expert_run_tokens{device,api,phase}` is the spread of
  tokens a run, `phase` the forward call's (`prefill`, `decode`,
  `verify`, set by the engine around each call); the runs cost about
  1% of a resident MoE model's CPU decode. On a file-backed model
  with `LOCUS_MOE_DISK_BYTES=1`, `locus_moe_expert_disk_bytes_total
  {layer,expert}` adds the bytes of an expert's weights not in RAM
  when it is picked (mincore, before the readahead hint), to set
  against `locus_storage_read_bytes_total`, the process's storage
  reads. It is opt-in: a mincore over each pick's weights halved
  Qwen1.5-MoE's CPU decode on the M4 Max (92 to 46 tok/s).
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
can use it underneath. On the wire a token's owner sends `kExpertCall`
(the layer, the expert id and the hidden state) and the expert's host
replies with `kExpertOut` (that expert's output before the router
weight), so the owner can combine the experts in its own pick order
and keep one host's bits. `locus-experts` serves an expert range
through `TransformerModel::expert_out`, the step `moe_ffn` takes per
pick, one connection at a time, and accepts only loopback peers unless
given `--allow`; with `--key-file` (and locus-run's
`--remote-experts-key-file`) a peer first passes the stage's key
handshake, and each side refuses a frame longer than its largest
message from the length prefix (#850): n_embd floats for the owner,
`kMaxExpertBatch` x (n_embd + 1) values, about 34 MB at n_embd 2048,
for the expert host. On the token's
side, `TransformerModel::set_remote_experts` takes a
`model::RemoteExperts` (`pipeline::ExpertClient` over those
connections): `moe_ffn` starts the remote picks, computes the local
ones meanwhile, then adds every pick in pick order, and its readahead
skips the remote experts. The batched forwards (a prefill chunk, a
multi-sequence decode, a stage's layers) go through `moe_ffn_batch`,
which sends a layer's remote picks of all its tokens as one
`kExpertCallBatch` per host (the hidden states, then their expert ids
as u32; at most `kMaxExpertBatch` = 4096 pairs a call, a larger batch
in chunks) and computes the local experts meanwhile (#1186). The host
runs each expert once over its tokens (`expert_out_batch`: one read of
its weights instead of one per token) and replies with one
`kExpertOutBatch` in call order. The client keeps one batched call in
flight per host, since a reply can be tens of MB and the host writes
it before it reads again. On a CPU expert host the replies keep the
bits of single calls (matvec_batch matches matvec), so a batched
prefill gives the logits of a token-at-a-time one; a GPU expert host
may not. With the MoE hybrid on as well, the batched forwards run the
MoE a token at a time through `moe_ffn` (#1185); so does a Vulkan main
host, whose `forward_batch` goes per token when its GPU batch declines
the model. At connect,
`locus-run` sends an empty `kExpertCallBatch` (`probe_batch_calls`):
an older `locus-experts` closes the connection on it, and the client
reconnects and sends that host single calls. The GPU
whole-token forwards (Vulkan's, Metal's) never call `moe_ffn`, so they
decline such a model and it runs per op, its matvecs still on the GPU.
The remote picks carry the bits of whatever backend each expert host
runs, which the main host does not know, so the disk prefix tier is
refused with remote experts.
`locus-run
--remote-experts HOST:PORT=LO:HI` (repeatable) connects to each
`locus-experts` and installs the client. A host lost mid-run ends the
run: the client's error reaches the caller and nothing retries.

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
an app-level probe. A stage writes whatever its peer sends into its KV
cache, so a `locus-stage --listen` (or a driver's `--logits-listen`)
that is not loopback needs `--allow` (#847). With `--key-file` (the
same file on every stage and the driver), each accepted connection and
each new downstream dial first runs a shared-key HMAC handshake, and a
peer without the key is closed before any frame; a pooled downstream
passed it when it was dialed. In the serve loop the handshake is one
more per-session state with its own deadline, so a silent peer does
not stall other sessions (#849). Frames are not signed: a peer on the
network path is not stopped. A stage refuses an input frame longer than
one activation (n_embd floats) from its length prefix, and the serve
loop drops a session whose partial frame makes no progress for
`--frame-timeout` (30 s); a connection idle between frames stays, as a
pooled one must (#854).
On Linux, every `--allow` (locus-stage and its metrics port,
locus-experts, the driver's logits listen, and `http::Server`'s
`allow`) is also a classic BPF socket filter on the listening socket:
a SYN from outside the ranges is dropped in the kernel and never
becomes a connection; the check at accept stays, and a reload
replaces the filter (#851).
macOS has no per-socket filter; pf can do the same with root, for
example an anchor `locus` (listed in pf.conf) loaded with
`sudo pfctl -a locus -f FILE`, where FILE holds

    pass in quick proto tcp from 10.0.0.0/8 to any port 9000
    block in quick proto tcp from any to any port 9000

The serve loop can report its live peers (each session's upstream
and downstream) with the kernel's smoothed TCP round-trip time, read
from `TCP_INFO` (Linux, FreeBSD) or `TCP_CONNECTION_INFO` (macOS)
with no probe of its own (#21); `locus-stage --peer-log N` prints
them every N seconds. `locus-stage --metrics-listen HOST:PORT` serves
them as Prometheus text on GET `/metrics`, from the same loop through
`http::LoopConnection`, with the session, token, byte and downstream
dial counters (#22). Each stage serves its own page, so a scraper
reads every stage directly; the driver does not gather them. The
`--allow` list covers that port too, since the page lists every peer's
address.
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
single-host run on hosts of one architecture, and token-exact output
(within the kernels' tolerance) across architectures; at concurrency
at or above the pipeline depth, throughput approaches the single-box
ceiling, with only the hop latency added.

`locus-server` serves the same way from several engines
(`server::EngineGroup`, #929): each an `EngineLoop` with its own
engine, KV pool and worker over the one loaded model, on its own
Vulkan device (its pool made with the caller bound there, its worker
bound to it) or its own share of the CPU cores
(`sys::ThreadPool::share_threads`, as the stage executors). Each engine
has its own prefix cache, so a request goes to the engine that last
served its longest prefix of whole 16-token blocks (a chained hash per
block, up to 64 blocks: chat prompts share their template's first
block), unless that engine has more than one unreleased request over
the least-loaded one; otherwise (and for a shorter prompt) to the
least-loaded engine (#940, #988). It stays
there; its id is `local * engines + engine`, so one engine keeps
today's ids. `/metrics` sums the engines. The disk prefix tier serves
one engine only; CUDA serves one engine (#924).

## 15. 8-bit activations (R16)

llama.cpp does not dot quantized weights against f32 activations. For
QK_K weights it quantizes the activation row to Q8_K (an f32 scale,
256 int8 values, 16 int16 block sums) and integer-dots the two; for
block-32 weights (Q8_0, Q4_0, IQ4_NL) it uses a Q8_0 activation.
locus ports those dots (`backend::matvec_q8k`):

| Backend | 8-bit activation dots                                         |
|---------|---------------------------------------------------------------|
| scalar  | all 15 QK_K types and the block-32 types                      |
| neon    | all 15 QK_K types and the block-32 types                      |
| sse4    | all 15 QK_K types and the block-32 types                      |
| cuda    | all 15 QK_K types                                             |

Today the sse4 and CUDA dots, and the neon ones but Q4_K and Q6_K,
replay the scalar accumulation order, so they are bit-identical to it
(a property of these kernels, not a requirement: section 16). The
neon Q4_K and Q6_K dots do not (below). The build turns off implicit
fused multiply-add (`-ffp-contract=off`), so the scalar path rounds
the same on x86-64 and arm64, and a stage chain that mixes the two is
exact (#458).

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

### Numerics: speed over exact bits

A kernel is judged by its error and its speed, not by matching another
kernel's bits (the owner's call, #1240). A kernel change passes:
- its tests against the scalar reference within a stated tolerance
  (matmuls 1e-4 relative, 1e-3 absolute; a looser bound only with a
  reason in the PR);
- the goldens: greedy output token for token, or a near-tie
  teacher-forced or re-pinned, named in the PR;
- the batch-invariant mode, which stays bit-exact when it is on.

What no longer binds: a batched forward equal to the per-token one,
CPU variants equal to each other, llama.cpp's non-repacked float
order, f32 staging on the GPU, an f32 KV cache by default. Each of
those moves only with a measured speed gain. Kernel variants are
picked per GPU family or CPU feature set from a table filled by
measurements (as `q6k_kernel()` is), and weights may be reordered once
at load into a kernel's layout where that measures faster. The build
keeps `-ffp-contract=off` for now, so the scalar reference rounds the
same on every compiler; allowing fused multiply-adds is a perf item
like any other, taken when it measures faster.
