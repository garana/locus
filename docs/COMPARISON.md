# locus vs other inference engines

Positioning snapshot (2026-08-02). locus's niche is narrow and
deliberate: **stream weights when the model does not fit in RAM/VRAM,
and batch inputs**, so weight reads are amortized across concurrent
requests and the page cache carries the hot working set. The engines
below are grouped by how they relate to that niche. Fast-moving
projects -- treat the cells as architecture-level positioning, not a
feature audit; verify specifics against upstream before quoting.

## Capability matrix

| Engine    | Bigger-than-RAM/VRAM streaming            | Continuous batching (paged KV) | Vendor-neutral GPU        | Serving API             | Stack / deps             |
|-----------|-------------------------------------------|--------------------------------|---------------------------|-------------------------|--------------------------|
| locus     | yes: mmap + page-cache working set,       | yes                            | yes: Vulkan + CUDA + CPU  | OpenAI + Anthropic      | C++20, minimal vendored  |
|           | routed-expert streaming                   |                                |                           |                         |                          |
| llama.cpp | partial: mmap weights, -ngl offload split | limited: server parallel slots | yes: Metal/Vulkan/CUDA/   | OpenAI-ish (llama-      | C/C++, minimal           |
|           | (no bigger-than-VRAM GPU streaming)       |                                | ROCm/SYCL                 | server)                 |                          |
| Ollama    | via llama.cpp                             | via llama.cpp (limited)        | via llama.cpp             | own + OpenAI-compat     | Go + llama.cpp           |
| vLLM      | no (model resident; some CPU swap)        | yes (originated PagedAttention)| mostly NVIDIA (+ROCm)     | OpenAI                  | heavy Python/CUDA        |
| LightLLM  | no                                        | yes                            | mostly NVIDIA             | OpenAI                  | Python                   |
| SGLang    | no                                        | yes (+ radix/prefix cache)     | mostly NVIDIA             | OpenAI                  | Python                   |
| TGI       | no                                        | yes                            | NVIDIA (+ some AMD)       | own + OpenAI-compat     | Rust + Python            |
| AirLLM    | yes (extreme): one layer/expert on GPU    | no                             | CUDA + Apple MLX + CPU    | none (library only)     | Python + PyTorch         |
|           | at a time                                 |                                | (no AMD)                  |                         |                          |

Two families: the Python GPU-serving stacks (vLLM/LightLLM/SGLang/
TGI) assume the model fits in VRAM and optimize throughput on top of
that; the portable C/C++ local runtimes (llama.cpp/Ollama) optimize
footprint and reach. locus sits in the gap -- a llama.cpp-class
footprint with a continuous-batching serving core -- and pushes on
the axis neither family targets: running a model that does not fit,
fast, by amortizing the streaming.

## AirLLM -- the closest neighbor on the streaming thesis

AirLLM is the one project whose central idea overlaps locus's: run a
model far larger than your GPU by not keeping it resident. It is
worth a direct look because it shares the goal and diverges sharply
on method.

How it works: the model lives **on disk** as per-layer shard files
(the checkpoint is split into ~80-100 shards on first run); AirLLM
memory-maps them and loads **one transformer layer at a time** onto
the GPU, runs that layer's forward pass, frees it, and moves to the
next -- and for sparse MoE models it streams **one expert at a time**
rather than a whole layer. Only ~one layer is ever resident, so the
whole model need not fit in CPU RAM (RAM/page cache just speeds the
reloads). Peak GPU memory drops to roughly one layer's worth (~1.6 GB
for a 70B), so it advertises 70B on a 4 GB GPU and 405B (Llama 3.1)
on 8 GB. The one-time split needs ~2x the model size in disk (the
original checkpoint plus the shards coexist during conversion;
reclaimable afterward via `delete_original`, and smaller still with
compression). It adds optional 4/8-bit block-wise compression of the
weights (to shrink load size, ~3x speed claim) and prefetch that
overlaps the next layer's load with the current layer's compute
(~10%). Backends: CUDA primarily, Apple (MLX + torch), and CPU; no
AMD. It is a Python/PyTorch library on the Hugging Face `AutoModel`
interface -- no server, no OpenAI API, and no batching. License
Apache-2.0.

The decisive architectural difference is **amortization**, not
disk-streaming per se -- **both** engines mmap weights from disk and
pull in only what a step needs; neither requires the whole model in
RAM. What separates them is what happens to that streaming cost:

- AirLLM re-streams the model's layers to the GPU on **every forward
  pass** with no cross-token or cross-request reuse, and runs one
  sequence at a time. Every token pays the full weight-read /
  host-to-device bill, so community runs of very large models sit
  well below 1 token/sec -- it is I/O-bound by construction. It is a
  "make it run at all" tool, explicitly trading speed for reach.
- locus targets exactly that bill. It keeps the hot working set
  resident across tokens (OS page cache on CPU; an LRU GPU weight
  pool on the Vulkan/CUDA pager, so hot weights are not re-uploaded
  per token), streams only the routed experts for MoE, and -- the
  core lever -- **batches many concurrent requests through one weight
  pass** (continuous batching + the R11 cache-blocked batched
  matvec), so a single stream of the weights serves N tokens at once.
  Same "bigger than memory" reach, but the streaming cost is divided
  across the batch instead of paid per token.

Put differently: AirLLM proves the demand (people will trade a lot of
speed to run a 405B on a laptop GPU) and picks the least
throughput-friendly implementation of it -- per-layer reload, single
sequence, no server. That is precisely the opening locus's "stream
**and** batch, behind a real serving API" thesis is built for. Where
AirLLM is the right tool for a one-off local generation on tiny
hardware, locus aims at serving a bigger-than-RAM model to
concurrent clients at a usable rate.

Caveat worth keeping honest: locus has not yet published a
head-to-head wall-clock vs AirLLM on the same host/model. The GLM-5.2
216 GB exit test (DESIGN.md R11/R12) measures locus against llama.cpp
(1.82x sse4 / 3.08x CUDA), not AirLLM; an AirLLM data point on that
model would make the streaming-niche claim concrete.

Sources: AirLLM GitHub (github.com/lyogavin/airllm);
"AirLLM: Layered Inference for Low-Memory Hardware" (B. Marie);
"AirLLM and '70B on a 4GB GPU'" (R. Shirke).
