# locus

A pure C/C++ LLM inference server. It runs on the CPU, on any Vulkan
GPU (Apple GPUs through MoltenVK included) and on CUDA, and it can run
models larger than RAM by streaming weights from disk. It serves the
OpenAI and Anthropic HTTP APIs, with a small, vendored dependency set.

- Design and rationale: [docs/DESIGN.md](docs/DESIGN.md)
- Supported clients and endpoints: [docs/CLIENTS.md](docs/CLIENTS.md)
- How locus compares to other engines:
  [docs/COMPARISON.md](docs/COMPARISON.md)
- How a transformer model works, as locus implements it:
  [docs/MODELS.md](docs/MODELS.md)

## Status

### What works

- **Architectures:** llama (dense and Mixtral-style MoE), deepseek2
  (MLA), glm-dsa, qwen2, qwen2vl (Qwen2.5-VL's language model, with
  M-RoPE for image tokens on the CPU), qwen2moe, qwen3moe, dbrx,
  muse-spark, deepseek41, kolibri1 (Kolibri-1).
- **Weight formats:** GGUF with F32, F16, BF16, the legacy quants,
  K-quants, IQ quants, TQ1_0 and TQ2_0.
- **Tokenizers:** SentencePiece, byte-level BPE (gpt2, llama3 and
  qwen2 pretokenizers), the T5 unigram model and BERT's WordPiece,
  with chat templates.
- **Backends:** neon, avx2, sse4 and scalar on the CPU; vulkan (the
  whole forward pass on the GPU); cuda (matvec on the GPU).
- **Serving:** continuous batching; a paged KV cache with
  copy-on-write prefix sharing; f32, f16, q8 or q4 KV.
- **API:** `/v1/completions`, `/v1/chat/completions`, `/v1/messages`,
  `/v1/responses` (stateless), `/v1/embeddings`, `/v1/rerank`,
  `/v1/classify`, `/v1/models`, `/metrics`, `/health`.
- **Embedding models:** GGUFs of arch `bert` (BERT with WordPiece, or
  XLM-R with SentencePiece) or `nomic-bert`, with `--embedding-model`,
  beside the chat model; tested with bge-small-en-v1.5, e5-small-v2,
  bge-m3 and nomic-embed-text v1.5. A reranker (an XLM-R cross-encoder
  such as bge-reranker-v2-m3) serves `/v1/rerank` with
  `--rerank-model`. Both run on the CPU, and a large request's passes
  on Vulkan where the GPU shares host memory (Apple), or on CUDA.
- **Decoding:** sampling (temperature, top-k, top-p, min-p,
  penalties, seed), structured output (JSON mode, a JSON Schema, a
  regex or a GBNF grammar), prompt caching (in RAM, and on disk with
  `--prefix-cache-dir DIR`, read back also after a restart), logprobs,
  prompt-lookup speculative decoding with a draft length that adapts
  to the accept rate and the measured verify cost.
- **Large models:** weights stream from disk with readahead, so a
  model can be larger than RAM.
- **Speech:** `/v1/audio/speech` with Orpheus-TTS and the SNAC codec;
  24 kHz WAV or PCM, whole or streamed.
- **Image output:** `/v1/images/generations` with Janus-Pro-1B and
  the VQ-16 decoder (one 384 x 384 PNG), with PixArt-alpha 512
  (T5-XXL encoder, DiT, DPM-Solver++, SD VAE; one 512 x 512 PNG), or
  with FLUX.1 dev, schnell or Kontext dev as text to image (CLIP-L and
  T5-XXL, the MMDiT, flow-matching Euler, the FLUX VAE; up to 1024 x
  1024 pixels). `/v1/images/edits` runs Qwen-Image-Edit
  (instruction-based edits with true CFG) or FLUX.1 Kontext dev on a
  reference image (its VAE-encoded tokens join the image tokens), for
  the same character or scene in a new picture. On a Vulkan GPU with
  unified memory, whatever backend the language model uses, the
  PixArt and FLUX DiTs run each whole step on the GPU with their
  activations resident, and the VAE convolutions run on the GPU
  (T5's attention stays on the CPU). On an M4 Max a
  256 x 256 FLUX image takes about 35 s; elsewhere FLUX runs on the
  CPU, minutes per image. Request fields:
  docs/CLIENTS.md.
- **Video output:** `/v1/videos` with LTX-Video 2B distilled (T5-XXL,
  the 3D DiT, rectified flow, the causal 3D VAE), text to video or
  image to video from a first frame. A clip is a job, in the shape of
  OpenAI's Videos API: POST starts it, GET reports it, and `/content`
  returns the MP4. locus does not encode video itself: ffmpeg runs as
  a helper process (`--ffmpeg`). Request fields: docs/CLIENTS.md.
- **Image input:** PNG and JPEG in chat requests (OpenAI `image_url`
  data URLs, Anthropic image blocks) with Janus-Pro-1B. On a Vulkan GPU
  with unified memory the vision tower runs on the GPU with its
  activations resident, whatever backend the language model uses (on
  an M4 Max, about 0.1 s per image; an image question about 0.9 s with
  the language model on Vulkan).
- **Tools:** `locus-server`; `locus-run` (one prompt); `locus-gguf`
  (inspect a GGUF); `locus-stage` and `locus-driver` (multi-server
  pipeline, experimental; a stage runs on the CPU, CUDA or Vulkan, and
  can serve its own Prometheus `/metrics`);
  `locus-router` (or `locus-server
  --router`: one URL in front of several locus-server instances, given
  as host:port, found by DNS SRV, by mDNS / DNS-SD or by scanning an
  address range, or registering themselves with a key (`locus-server
  --register`), each request sent to an instance that serves its kind
  and model; experimental, #388);
  `locus-auth-helper` (a reference auth helper,
  see docs/CLIENTS.md).

### Media: from what to what

| From \ To | Text                                   | Image                                               | Video                                    |
|-----------|----------------------------------------|-----------------------------------------------------|------------------------------------------|
| Text      | completions, chat, messages, responses | `/v1/images/generations`: Janus-Pro, PixArt, FLUX.1 | `/v1/videos`: LTX-Video                  |
| Image     | chat with an image: Janus-Pro-1B       | `/v1/images/edits`: Qwen-Image-Edit, FLUX.1 Kontext | `/v1/videos` with `input_reference`: LTX |
| Video     | no                                     | no                                                  | no                                       |

Speech (text to audio) is `/v1/audio/speech`. How these run together:

1. Text to text has no lock: requests share continuous batching. Every
   other kind runs alongside it and slows it, since they share the CPU
   cores and the GPU; a FLUX step on the GPU holds the Vulkan lock, so
   text on the same GPU waits for each step.
2. Image to text runs one request at a time (the next one waits), on
   its own KV cache, outside the batch (#257).
3. Text to image and image to image share one lock with video
   generation: one image at a time, and no queue. A request that finds
   the lock taken gets a 503 at once, with `Retry-After` (60 s for
   PixArt and FLUX, 30 s for Janus).
4. Text and image to video are jobs: up to 4 wait or run, and one runs
   at a time. It holds the image lock while it denoises and decodes
   (not during the ffmpeg encode), so image requests get 503s
   meanwhile, and a waiting clip starts after an image in progress. A
   fifth clip gets a 503 with `Retry-After` 60 s.
5. Image and video generation never run two at once. The weights are
   shared, but each generation has its own working memory, and one
   generation already keeps the GPU or the CPU cores busy.
6. Speech runs one request at a time; the next one waits.

Greedy decoding is the default; sampling and the other decoding
features are opt-in per request. A request whose client disconnects
is cancelled. On the CPU, quantized weights use 8-bit activations
where the backend's int8 dot is the fast path (neon, sse4, scalar),
as llama.cpp does; `LOCUS_F32_ACTIVATIONS=1` selects f32 activations.
`LOCUS_CPU_FAST=1` opts in to CPU kernels with a different float order
(today every CPU softmax: attention, MoE routing and the image
encoders, about 2x faster): close to the default results but not
bit-identical, so the goldens are checked without it.

### Validated models

Each model below has a check in the test suite, against a reference:
llama.cpp's greedy output, or the torch reference for the multimedia
models. "Teacher-forced"
means each reference token must be within a small margin of the top
logit, for prompts where greedy decoding tips at a near-tie.

| Model                     | Arch       | Weights            | Check                                                      |
|---------------------------|------------|--------------------|------------------------------------------------------------|
| stories260K               | llama      | F32                | token-exact, every backend                                 |
| Llama-3.2-1B              | llama      | Q8_0, Q4_K_M       | token-exact; Q4_K_M on every backend                       |
| DeepSeek-V2-Lite          | deepseek2  | Q4_K_M             | token-exact, CPU and Vulkan                                |
| DeepSeek-Coder-V2-Lite    | deepseek2  | Q8_0               | Vulkan batched prefill tracks per-token prefill            |
| Qwen1.5-MoE-A2.7B         | qwen2moe   | Q4_K_M             | token-exact, CPU and Vulkan                                |
| Qwen3-Coder-30B-A3B       | qwen3moe   | Q4_K_M             | token-exact, CPU and Vulkan                                |
| DBRX-Instruct 16x12B      | dbrx       | IQ2_XXS, 32 GB     | token-exact                                                |
| Muse Spark Open 816B-A42B | muse-spark | UD-IQ1_S, 176 GB   | one prompt token-exact, one teacher-forced                 |
| DeepSeek-V4.1-Flash-1017  | deepseek41 | UD-IQ1_S, 233 GB   | teacher-forced, against llama.cpp#30817                    |
| Kolibri-1 384x2.8B        | kolibri1   | APEX I-Nano, 24 GB | English token-exact, German teacher-forced, CPU and Vulkan |
| Orpheus-TTS 3B            | llama      | Q4_K_M             | first audio frame teacher-forced; SNAC RMS < 1e-6          |
| Janus-Pro-1B              | llama      | BF16               | 576 image codes exact; vision tower RMS < 1e-4 rel         |
| PixArt-XL-2-512x512       | pixart     | F32, T5 F16        | each stage vs diffusers; 8-step latent RMS 9e-5            |

Two goldens run with f32 activations, because 8-bit activations tip
their greedy text at a near-tie: Qwen1.5-MoE and DeepSeek-V2-Lite.

Checked by hand, with no test in the suite: GLM-5.2 744B (glm-dsa,
UD-IQ1_S, 203 GB, streamed on a 32 GB machine) and Moonlight-16B-A3B
(deepseek2), both token-exact against llama.cpp.

### Speed

| Workload                                   | Hardware          | Speed                                  |
|--------------------------------------------|-------------------|----------------------------------------|
| Llama-3.2-1B Q4_K_M decode, Vulkan         | M4 Max (MoltenVK) | 299 tok/s (llama.cpp Metal: 312)       |
| Llama-3.2-1B Q8_0 decode, Vulkan           | M4 Max (MoltenVK) | 214 tok/s (llama.cpp Metal: 223)       |
| Orpheus 3B Q4_K_M decode, Vulkan           | M4 Max (MoltenVK) | 131 tok/s (llama.cpp Metal: 135)       |
| Llama-3.2-1B Q8_0 prefill, 2.5k tokens     | M4 Max (MoltenVK) | 2048 tok/s                             |
| DeepSeek-Coder-V2-Lite Q8_0 decode, Vulkan | M4 Max (MoltenVK) | 88 tok/s                               |
| Kolibri-1 APEX I-Nano decode, Vulkan       | M4 Max (MoltenVK) | 37 tok/s (llama.cpp CPU: 24 to 31)     |
| Kolibri-1 prefill, 560 tokens, Vulkan      | M4 Max (MoltenVK) | 383 tok/s (llama.cpp CPU: 73)          |
| Llama-3.2-1B Q8_0 decode, CPU              | M4 Max (neon)     | 89 tok/s                               |
| Llama-3.2-1B Q8_0 prefill, 8k tokens, CPU  | M2 Pro (neon)     | 124 tok/s                              |
| Llama-3.2-1B Q8_0 prefill, 2k tokens, CPU  | i5-2400 (sse4)    | 18 tok/s (llama.cpp: 26)               |
| Janus-Pro-1B, one image                    | M2 Pro (MoltenVK) | 22 s, decode included                  |
| Orpheus speech, 3.8 s clip                 | M2 Pro (MoltenVK) | 12 s; first audio at 1.7 s             |
| Qwen-Image-Edit, 512 x 512, 20 steps       | M4 Max (MoltenVK) | 344 s denoise with `auto_resize` false |

### Pending

- Multimedia (#143): real-time speech on the M2 Pro (#173; the M4 Max
  runs at 1.34x real time), Qwen-Image-Edit's 65 s CPU encode (#671,
  parked: the edit's working set does not fit a 36 GiB host with the
  language model on the GPU).
- Performance: prefill on Apple GPUs at 44% of Metal's (#10:
  MoltenVK exposes no cooperative matrix), CPU prefill on x86 (#618:
  an AVX path, #706, behind the opt-in `LOCUS_CPU_FAST` flag, as it
  changes the float order), batch-invariant decode at full
  speed on V3D (#511), the MoE hybrid on Vulkan (#679).
- Multi-server inference, pipeline and expert parallel (#3, #24).
- Waiting on hardware: the AVX2 kernel's validation needs an AVX2 host
  (#61); the multi-GPU weight pager needs several GPUs (#62).

## Build and test

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    ./build/tests/locus_tests

Tests that need model files skip when the file is absent;
`scripts/fetch-test-model.sh` gets the small ones (`bge`, `e5`,
`bge-m3` or `nomic` as its argument gets that embedding model, `qwen2`
the qwen2 golden's model, and `qwen2.5-vl-mmproj` the Qwen2.5-VL vision
tower). Tests tagged as heavy also need `LOCUS_HEAVY_TESTS=1`.

## Run

    locus-server model.gguf 8080
    locus-run model.gguf "Once upon a time" 32

The server listens on every interface; `--host 127.0.0.1` keeps it
on loopback. `locus-server --help` lists the options and environment
variables;
`--capabilities` lists the architectures, backends, weight formats
and tokenizers of the build. Speech needs `--audio-decoder
<snac.gguf>`, image generation `--image-decoder <vq16.gguf>` (Janus)
or `--image-model <dir>` (PixArt: a directory with the GGUFs from
`scripts/convert-t5-encoder.py`, `convert-pixart.py` and
`convert-sd-vae.py`; FLUX.1: the GGUFs from `scripts/convert-flux.py`
and `convert-t5-encoder.py`, plus `--lora-dir <dir>` for LoRAs from
`scripts/convert-lora.py`). Video needs `--video-model <dir>` (the
GGUFs from `scripts/convert-ltxv.py` and `convert-t5-encoder.py`) and
an ffmpeg on PATH or at `--ffmpeg <path>`. A dedicated embedding model
for `/v1/embeddings` needs `--embedding-model <bert.gguf>`; task
prefixes such as "query: " are the client's job. `/v1/rerank` needs
`--rerank-model <reranker.gguf>`, made from a transformers XLM-R
reranker by `scripts/convert-xlmr-reranker.py`. `/v1/classify` and
image inputs to `/v1/embeddings` need `--image-classifier <cnn.gguf>`,
a ResNet-50 or Inception-v3 from `scripts/convert-cnn.py`.

## Backends

Every backend is compiled into the one binary, and the best one the
machine supports is picked at startup (on x86, avx2 before sse4). Pick
one with `--backend NAME` or `LOCUS_BACKEND=NAME`, and with `--backend
vulkan` or `cuda` the GPU with `--device N` (`--backends` lists the
Vulkan devices; the first is the default).

| Backend | Runs                                                                                     |
|---------|------------------------------------------------------------------------------------------|
| neon    | arm64; vectorized matvec and int8 dots, the rest scalar                                  |
| avx2    | x86-64 with AVX2; vectorized F32 / Q8_0 matvec, f32 activations; not yet validated (#61) |
| sse4    | x86-64; vectorized matvec and int8 dots, the rest scalar                                 |
| scalar  | portable reference, every weight type                                                    |
| vulkan  | the whole forward pass on the GPU: matmuls, norms, RoPE, paged attention, MoE routing    |
| cuda    | matvec and attention on the GPU (paged weight pool, KV cache copy); the rest on the CPU  |

Every selectable backend must reproduce the golden outputs. A GPU does
not always beat the CPU: on a tile GPU streaming a model larger than
its weight pool (a Raspberry Pi 5 running DeepSeek-V2-Lite) Vulkan
runs, but slower than neon.

## Dependencies

No runtime dependencies beyond the C and C++ libraries and the OS.
Vendored, pinned and checksummed (see DESIGN.md section 6): Catch2
and cpp-httplib (tests only), nlohmann/json, and ggml's IQ lookup
tables. Image input (PNG, JPEG) uses locus's own decoders. Optional,
found at configure time: the Vulkan loader and a shader compiler, the
CUDA toolkit.
