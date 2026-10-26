# locus vs other inference engines

locus is a full inference server: continuous batching, paged KV, the
OpenAI and Anthropic APIs, sampling, constrained decoding, prompt
caching and speculative decoding, over CPU, CUDA and Vulkan (which
reaches Apple GPUs through MoltenVK, so there is no separate Metal
backend). Its distinctive lever is running models that do not fit. On
one host it **streams weights when the model exceeds RAM or VRAM and
batches inputs**, so weight reads are shared across concurrent
requests and the page cache carries the hot working set. Across hosts,
pipeline stages (experimental) pool the memory of several machines to
run a model too big for any one of them.

The engines below are grouped by how they relate to that "run bigger
than memory, fast" thesis. These projects move fast: read the cells as
architecture-level positioning, not a feature audit, and check
specifics upstream before quoting them.

## Capability matrix

| Engine       | Bigger-than-RAM/VRAM streaming            | Continuous batching (paged KV) | Vendor-neutral GPU        | Serving API             | Stack / deps             | Multi-host scaling                          |
|--------------|-------------------------------------------|--------------------------------|---------------------------|-------------------------|--------------------------|---------------------------------------------|
| locus        | yes: mmap + page-cache working set,       | yes                            | yes: Vulkan + CUDA + CPU  | OpenAI + Anthropic      | C++20, minimal vendored  | experimental: pipeline stages; expert       |
|              | routed-expert streaming                   |                                |                           |                         |                          | parallel planned                            |
| llama.cpp    | partial: mmap weights, -ngl offload split | limited: server parallel slots | yes: Metal/Vulkan/CUDA/   | OpenAI-ish (llama-      | C/C++, minimal           | partial: RPC + tensor-split                 |
|              | (no bigger-than-VRAM GPU streaming)       |                                | ROCm/SYCL                 | server)                 |                          |                                             |
| Ollama       | via llama.cpp                             | via llama.cpp (limited)        | via llama.cpp             | own + OpenAI-compat     | Go + llama.cpp           | via llama.cpp                               |
| Apple MLX    | no (resident; scales via distribution)    | yes (mlx_lm.server)            | no: Apple Silicon only    | OpenAI-compat           | Python + MLX (C++ core)  | yes: pipeline + tensor (Ring/JACCL)         |
| vLLM         | no (model resident; some CPU swap)        | yes (originated PagedAttention)| mostly NVIDIA (+ROCm)     | OpenAI                  | heavy Python/CUDA        | yes: tensor + pipeline parallel, multi-node |
| LightLLM     | no                                        | yes                            | mostly NVIDIA             | OpenAI                  | Python                   | yes: tensor parallel                        |
| SGLang       | no                                        | yes (+ radix/prefix cache)     | mostly NVIDIA             | OpenAI                  | Python                   | yes: tensor + pipeline parallel             |
| TGI          | no                                        | yes                            | NVIDIA (+ some AMD)       | own + OpenAI-compat     | Rust + Python            | yes: tensor-parallel sharding               |
| AirLLM       | yes (extreme): one layer/expert on GPU    | no                             | CUDA + Apple MLX + CPU    | none (library only)     | Python + PyTorch         | no (single device)                          |
|              | at a time                                 |                                | (no AMD)                  |                         |                          |                                             |
| kimi-k3-in-c | yes (extreme): routed 4-bit experts       | no (single sequence)           | no: CPU-only (AVX2+FMA)   | none (CLI + C library)  | C99 + OpenMP, no BLAS    | no (single CPU)                             |
|              | streamed from disk per token (LRU)        |                                |                           |                         |                          |                                             |

Two families. The Python GPU-serving stacks (vLLM, LightLLM, SGLang,
TGI) assume the model fits in VRAM and optimize throughput on top of
that. The portable C/C++ local runtimes (llama.cpp, Ollama) optimize
footprint and reach. locus sits in the gap, a llama.cpp-class
footprint with a continuous-batching serving core, and pushes on the
axis neither family targets: running a model that does not fit, fast,
by sharing the cost of streaming it.

## Multi-host scaling

The matrix above is single-host. A separate axis spreads one model
across several machines to fit a model too big for any one of them
(capacity scaling), which differs from replicating a resident model
for throughput. locus does this with pipeline parallelism: it splits
the layers across hosts, hands the residual stream (a few KB per
token) from stage to stage, and keeps each host's weights resident.
The design is in DESIGN.md section 14; the stage tools work, but are
experimental and not benchmarked.

The Python GPU stacks (vLLM, SGLang, TGI, LightLLM) already do tensor
and pipeline parallelism, aimed at throughput on a model that fits the
cluster's VRAM, over a fast (NVLink-class) interconnect. Apple's MLX
does the same across Macs (pipeline and tensor over its Ring backend,
or JACCL on Thunderbolt 5), the closest analog to locus's approach but
Apple Silicon only.

locus's angle is the streaming niche again: pool the memory of cheap
commodity hosts over a plain LAN to run a model none of them could
hold alone. It tolerates network latency, because only the small
activation vector crosses the wire (not the weights) and concurrency
hides the hop. Tensor parallelism (an all-reduce every layer) is not
the target, since it needs a fast interconnect; pipeline and expert
parallelism suit a LAN. This is the distributed sibling of the
multi-GPU weight-sharding pager, and composes with it: a stage can
shard its layers across that host's GPUs.

## AirLLM: the closest neighbor on the streaming thesis

AirLLM is the one project whose central idea overlaps locus's: run a
model far larger than your GPU by not keeping it resident. It shares
the goal and diverges sharply on method.

How it works: the model lives **on disk** as per-layer shard files
(the checkpoint is split into about 80 to 100 shards on first run).
AirLLM memory-maps them and loads **one transformer layer at a time**
onto the GPU, runs that layer's forward pass, frees it, and moves to
the next; for sparse MoE models it streams **one expert at a time**
rather than a whole layer. Only about one layer is ever resident, so
the whole model need not fit in CPU RAM (RAM and page cache just speed
up the reloads). Peak GPU memory drops to roughly one layer's worth
(about 1.6 GB for a 70B), so it advertises 70B on a 4 GB GPU and 405B
(Llama 3.1) on 8 GB.

- The one-time split needs about twice the model size in disk (the
  original checkpoint and the shards coexist during conversion;
  reclaimable afterwards with `delete_original`, and smaller still with
  compression).
- Optional 4- or 8-bit block-wise compression of the weights shrinks
  the load (a claimed 3x speed-up), and prefetch overlaps the next
  layer's load with the current layer's compute (about 10%).
- Backends: CUDA primarily, Apple (MLX and torch), and CPU; no AMD.
- It is a Python / PyTorch library on the Hugging Face `AutoModel`
  interface, with no server, no OpenAI API and no batching. License:
  Apache-2.0.

The decisive difference is **amortization**, not disk streaming as
such. **Both** engines mmap weights from disk and pull in only what a
step needs; neither needs the whole model in RAM. What separates them
is what happens to that streaming cost:

- AirLLM re-streams the model's layers to the GPU on **every forward
  pass**, with no reuse across tokens or requests, and runs one
  sequence at a time. Every token pays the full weight-read and
  host-to-device bill, so community runs of very large models sit well
  below 1 token per second: it is I/O-bound by construction. It is a
  "make it run at all" tool, trading speed for reach on purpose.
- locus targets exactly that bill. It keeps the hot working set
  resident across tokens (the OS page cache on the CPU; an LRU GPU
  weight pool on the Vulkan and CUDA pagers, so hot weights are not
  re-uploaded per token), streams only the routed experts for MoE,
  and, the core lever, **batches many concurrent requests through one
  weight pass** (continuous batching and the R11 cache-blocked batched
  matvec), so one stream of the weights serves N tokens at once. Same
  "bigger than memory" reach, but the streaming cost is divided across
  the batch instead of paid per token.

Put differently: AirLLM proves the demand (people will trade a lot of
speed to run a 405B on a laptop GPU) and picks the least
throughput-friendly way to meet it: per-layer reload, a single
sequence, no server. That is the opening locus's "stream **and**
batch, behind a real serving API" thesis is built for. AirLLM is the
right tool for a one-off local generation on tiny hardware; locus aims
at serving a bigger-than-RAM model to concurrent clients at a usable
rate.

To keep it honest: locus has no published head-to-head wall-clock
against AirLLM on the same host and model. The GLM-5.2 (216 GB) test
in DESIGN.md section 9 measures locus against llama.cpp (1.82x on
sse4, 3.08x on CUDA), not against AirLLM; an AirLLM data point on that
model would make the streaming claim concrete.

Sources: AirLLM GitHub (github.com/lyogavin/airllm); "AirLLM: Layered
Inference for Low-Memory Hardware" (B. Marie); "AirLLM and '70B on a
4GB GPU'" (R. Shirke).

## kimi-k3-in-c: the streaming thesis in portable C

kimi-k3-in-c (FareedKhan-dev) is the other project built on locus's
core idea, and it pushes it to an extreme: it runs the 2.78-trillion-
parameter Kimi K3 on a single CPU in about 8.24 GB of RAM. Like
AirLLM, it keeps almost nothing resident. The 93-layer dense trunk
(bf16) can stream from a packed file with a tunable pinned depth, and
the about 82k routed experts (1.45 TB, packed to about 4 bits) never
load: each token activates 16 of 896 experts per layer, and the rest
"sit asleep on disk", multiplied straight out of their 4-bit form on
demand, with an LRU cache for reuse within a run. It is portable C99,
with no BLAS, no framework and no GPU (AVX2 and FMA, OpenMP); license
Apache-2.0.

On stack it is the closest neighbor of all: minimal C with mmap
streaming, like locus. It diverges on the same axis as AirLLM,
amortization:

- It runs one sequence at a time (an optional `--incremental` KV flag;
  without it each step recomputes the prefix).
- It has no continuous batching and no server: a CLI plus a C library,
  with results written to a JSON file.
- It moves on the order of 100 GB or more per token at small memory
  budgets, so it is I/O-bound by construction: a "make it run at all"
  tool.
- It is specific to Kimi K3, where locus loads arbitrary GGUF models.

locus takes the same bigger-than-memory reach and divides the
streaming cost across a batch of concurrent requests behind an OpenAI
and Anthropic server, trading kimi-k3-in-c's single-CPU simplicity for
throughput.

Sources: kimi-k3-in-c GitHub (github.com/FareedKhan-dev/kimi-k3-in-c).
