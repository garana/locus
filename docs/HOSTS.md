# Recommended settings per host type

Which backend and flags to run locus with, by hardware, each choice
backed by a measurement on the named part. A section applies to other
hardware only by the matching rule it states; the numbers hold for the
named part alone. Speeds are tokens a second unless said otherwise.

Some choices are automatic:

- The CPU backend: the best variant the CPU supports, read at startup
  (`--backend` or `LOCUS_BACKEND` overrides it).
- The image, image-edit and video models pick their own engine,
  whatever `--backend` says: the GPU through Vulkan where the Vulkan
  device maps host memory (Apple's unified memory), else the CPU.
  `locus-server` prints one `engine:` line per kind of request.
- On Apple GPUs, the Metal kernels pick their layout by GPU family
  (the Q6_K matvec, #1060; the decode norm fold, #1082).

The text model is not automatic yet: without `--backend` it runs on
the CPU, and `locus-server` warns when a GPU is usable.

How each was measured: `locus-run --warmup --timing` (decode: 127
generated tokens after a short prompt; prefill: a prompt of the length
given), against `llama-bench` (`tg128`, `pp<N>`) on the same host.
llama.cpp's figure counts the graph alone, locus's the engine's whole
step, so the gap is a little smaller than the ratio.

## Apple M2 Pro

Match rule: Apple GPU family apple8 (M2). Measured on an M2 Pro: arm64
with neon, dotprod and i8mm; 32 GB unified memory; macOS. Vulkan runs
on MoltenVK.

Recommended: `--backend metal` for text, the default since #1042
(metal for the models its forward takes, else vulkan). Images and
video take the GPU through Vulkan by themselves. Concurrent clients
decode together since #1184; locus-server, Llama 1B, 128 tokens a
client, tok/s in all:

| model  | clients | Metal before #1184 | Metal, invariant | Metal, not invariant | Vulkan |
|--------|---------|--------------------|------------------|----------------------|--------|
| Q8_0   | 1       | 106.4              | 108.5            | 107.3                | 95.8   |
| Q8_0   | 4       | 103.9              | 149.4            | 152.0                | 166.1  |
| Q8_0   | 8       | 102.7              | 157.2            | 159.3                | 165.9  |
| Q4_K_M | 1       | 162.8              | 161.2            | 162.2                | 112.7  |
| Q4_K_M | 4       | 158.2              | 187.1            | 217.0                | 127.3  |
| Q4_K_M | 8       | 156.7              | 197.3            | 243.1                | 119.3  |

Q8_0's default is the invariant column, Q4_K_M's the other (as on the
M4 Max below). Several Q8_0 clients still do 4 to 9% better on
`--backend vulkan`; Q4_K_M runs 1.4 to 2x Vulkan at every count.

| workload                         | command                        | locus Metal     | locus Vulkan | llama.cpp Metal                 | source       |
|----------------------------------|--------------------------------|-----------------|--------------|---------------------------------|--------------|
| decode, Llama 3.2 1B Q4_K_M      | `--backend metal`              | 164             | 111          | 172                             | #1069, #1071 |
| decode, Llama 3.2 1B Q8_0        | `--backend metal`              | 105             | 94           | 121                             | #1082        |
| decode, Qwen2.5-0.5B Q8_0        | `--backend metal`              | 196             | 153          | not measured                    | #1082        |
| prefill 2.3k, Llama 1B Q4_K_M    | `--backend metal`              | 1063            | 674          | 2109 (pp2048)                   | #1066, #1070 |
| prefill 2.3k, Llama 1B Q8_0      | `--backend metal`              | 1144            | 705          | 2347 (pp2048)                   | #1066        |
| prefill 2.3k, Qwen2.5-0.5B Q8_0  | `--backend metal`              | 2437            | 1556         | not measured                    | #1066        |
| speech, Orpheus 3B Q4_K_M + SNAC | `locus-server --backend metal` | 15.5 ms a token | not measured | 15.9 ms a token (with sampling) | #784, #1060  |

- Prefill trails llama.cpp at about 0.5x here (#1074, Backlog).
- Speech: 17.7 ms a token on Metal before #1060's Q6_K kernel; the
  same request, 784 tokens with sampling, in both.
- PixArt's VAE decode, 128 x 128: CPU 27.2 s, Vulkan 3.09 s, Metal's
  convs 2.83 s (#1055). The server takes Vulkan for it today (#1057).

## Apple M4 Max

Match rule: Apple GPU family apple9 (M3, M4), 32 GB or more of unified
memory. Measured on an M4 Max: 14 CPU cores (10 performance, 4
efficiency), 32 GPU cores; arm64 with neon, dotprod and i8mm; 36 GB;
macOS.

Recommended: `--backend metal` for text, the default since #1042: it
leads Vulkan on every prefill and on most decode rows. On the decode
of Orpheus and Qwen3-Coder the two are within 1%. Concurrent clients
decode together since #1184; locus-server, Llama 1B, 128 tokens a
client, tok/s in all:

| model  | clients | Metal before #1184 | Metal, invariant | Metal, not invariant | Vulkan |
|--------|---------|--------------------|------------------|----------------------|--------|
| Q8_0   | 1       | 213.0              | 212.3            | 213.0                | 208.0  |
| Q8_0   | 4       | 206.2              | 500.8            | 512.5                | 434.3  |
| Q8_0   | 8       | 201.9              | 527.9            | 540.1                | 431.1  |
| Q4_K_M | 1       | 279.8              | 276.2            | 280.1                | 285.4  |
| Q4_K_M | 4       | 273.2              | 372.6            | 397.4                | 403.5  |
| Q4_K_M | 8       | 265.2              | 392.8            | 430.8                | 414.8  |

Batch-invariant mode (a request decodes to the same bits alone or
batched) is the default for Q8_0, at no cost to a lone request and
about 2% with several; `LOCUS_METAL_BATCH_INVARIANT=0` turns it off. A
Q4_K or Q6_K model runs the fastest kernels by default instead, as on
Vulkan, and `LOCUS_METAL_BATCH_INVARIANT=1` makes it invariant at the
cost above.

| workload, model                    | locus Metal | locus Vulkan | llama.cpp Metal | source |
|------------------------------------|-------------|--------------|-----------------|--------|
| decode, Llama 3.2 1B Q4_K_M        | 293.7       | 291.9        | 311.7           | #1064  |
| decode, Llama 3.2 1B Q8_0          | 221.1       | 213.3        | 223.7           | #1064  |
| decode, Orpheus 3B Q4_K_M          | 126.7       | 127.4        | 134.4           | #1064  |
| decode, Qwen2.5-0.5B Q8_0          | 345.6       | 304.0        | 337.7           | #1064  |
| decode, Qwen1.5-MoE-A2.7B Q4_K_M   | 132.6       | 117.0        | 156.3           | #1064  |
| decode, Qwen3-Coder-30B-A3B Q4_K_M | 97.2        | 98.0         | 108.0           | #1064  |
| prefill 1.7k, Llama 1B Q8_0        | 3442        | 2142         | 4939 (pp1681)   | #1063  |
| prefill 1.7k, Llama 1B Q4_K_M      | 3302        | 2107         | 4737            | #1063  |
| prefill 1.7k, Orpheus 3B Q4_K_M    | 1177        | 797          | 1653            | #1063  |
| prefill 1.7k, Qwen2.5-0.5B Q8_0    | 8051        | 4620         | 11101           | #1063  |
| prefill 1.7k, Qwen1.5-MoE Q4_K_M   | 1534        | 939          | 1917            | #1063  |
| prefill 1.7k, Qwen3-Coder Q4_K_M   | 939         | 567          | 1195            | #1063  |

- The CPU alone (no GPU): `--backend neon` with `LOCUS_CPU_FAST=1`
  prefills 1.7k tokens through Accelerate: Qwen2.5-0.5B Q8_0 1231
  against 875 by default, Llama 1B Q8_0 701 against 401 (#1027). The
  results are close to the default's but not bit-identical.
- FLUX.1 Kontext Q8_0, one DiT step: 512 x 512 Vulkan 4.88 s, Metal
  4.30 s; 1024 x 1024 Vulkan 17.6 s, Metal 16.4 s (#1044). The server
  serves images on Vulkan today (#1057).

## Intel Core i5-2400 with an NVIDIA GTX 750 Ti

Match rule: x86-64 with SSE4.1 and AVX, no AVX2 and no FMA; an NVIDIA
GPU of compute capability 5.0. Measured on an Intel Core i5-2400
(Sandy Bridge, 4 cores), 32 GB RAM without ECC, and a GTX 750 Ti (2
GB; driver 580.178.04, CUDA 12.6); Ubuntu 24.04, GCC 13.

Automatic: the CPU backend `sse4` (its AVX build). locus does not move
to CUDA by itself: pass `--backend cuda`.

Recommended: `--backend cuda` for dense models that fit in 2 GB (1.7x
to 2.2x the CPU's decode, 2.3x to 3.9x its prefill); the CPU default
for larger ones.

| workload                         | default (CPU) | llama.cpp CPU | `--backend cuda` | llama.cpp CUDA | source      |
|----------------------------------|---------------|---------------|------------------|----------------|-------------|
| prefill 512, Llama 3.2 1B Q4_K_M | 31.9          | 46.2          | 125.8            | 499.9          | #412, #411  |
| decode, Llama 3.2 1B Q4_K_M      | 16.1          | 17.8          | 27.9             | 31.7           | #412, #411  |
| prefill 512, Llama 3.2 1B Q8_0   | 38.4          | 44.0          | 90.6             | 500.1          | #412, #411  |
| decode, Llama 3.2 1B Q8_0        | 11.6          | 11.9          | 25.8             | 36.2           | #412, #411  |
| prefill 512, Qwen2.5-0.5B Q8_0   | 86.7          | 115.3         | 196.7            | 1195.6         | #412, #1196 |
| decode, Qwen2.5-0.5B Q8_0        | 25.9          | 27.9          | 53.1             | 80.9           | #412, #1196 |
| prefill 512, Orpheus 3B Q4_K_M   | 12.4          | 15.9          | 19.3             | does not load  | #412, #1196 |
| decode, Orpheus 3B Q4_K_M        | 6.5           | 7.0           | 1.5              | does not load  | #412, #1196 |
| prefill 512, Qwen1.5-MoE Q4_K_M  | 14.2          | 20.1          | does not fit     | not run        | #412        |
| decode, Qwen1.5-MoE Q4_K_M       | 8.1           | 9.1           | does not fit     | not run        | #412        |
| LTX-Video 2B, one DiT step       | 75.3 s        | PyTorch 85 s  | not run          | not run        | #982        |

- Measured on main at 4fb4f9d8 against llama.cpp e8e6c7a (a CPU build,
  4 threads, and a CUDA build for sm_50), median of 3 runs. Decode
  follows the prompt "1, 2, 3, 4, 5,", on which every model decodes all
  127 tokens.
- On the CPU, decode is at 0.89x to 0.98x of llama.cpp and prefill at
  0.69x to 0.87x (#1198). Decode is bound by memory bandwidth: with 16
  GB (one module out) it ran 32% to 40% slower on both. The MoE rows
  have the 9.5 GB file in the page cache; a cold first prefill runs at
  8.5.
- On CUDA, prefill is at 0.16x to 0.25x of llama.cpp (#1200). Decode
  is at 0.88x on Q4_K_M, 0.71x on Q8_0 (#1197) and 0.66x on Qwen2.5.
  The Qwen2.5 and Orpheus CUDA rows are of #1196 (Qwen2 on the
  one-submit forward; Orpheus's two runs, not three).
- Orpheus (1.94 GB of weights) does not fit the 2 GB card: llama.cpp
  fails to load it, and locus pages its weights (slower than the CPU).
- `LOCUS_CPU_FAST=1` gained 1 to 7% on the dense rows (#1059), near the
  noise: the default stays, as it reproduces the goldens. On the DiT
  step it cuts the self attention (12.2 to 10.4 s, #1062) but not the
  whole step (one run each).
- Vulkan on this card has 2 GB of device memory and no host import:
  an MoE whose experts do not fit routes on the host and runs the
  picked experts on the CPU (i#1174), the rest on the GPU. Qwen1.5-MoE
  Q4_K_M, 48 tokens, page cache warm: decode 3.9 tok/s, against 2.6
  on the CPU backend and 0.1 when the picks were uploaded each token
  (`LOCUS_VK_HOST_EXPERTS=0` still does that).

## Raspberry Pi 5

Match rule: arm64 with dotprod, no i8mm; a V3D 7.x GPU (Mesa V3DV).
The CPU rows hold for any arm64 CPU with dotprod and no i8mm. Measured
on a Raspberry Pi 5 (BCM2712, 4 Cortex-A76 at 2.4 GHz), 4 GB RAM,
V3D 7.1 (Mesa 24.2.8, Vulkan 1.2); Debian 12, GCC 12.

Automatic: `neon` (it uses its dotprod kernels on this CPU).
Recommended: no flags; leave `--kv-type` at f32 and `--engines` at 1.

| workload                         | command               | number                                                                  | source     |
|----------------------------------|-----------------------|-------------------------------------------------------------------------|------------|
| text, Llama 3.2 1B Q4_K_M        | default               | prefill 31 to 33, decode 7.6 to 7.7 (llama.cpp: 66, 9.6)                | #1059      |
| text, Qwen2.5-0.5B Q8_0          | default               | prefill 63 to 75, decode 14.5 to 14.9 (llama.cpp: 229, 13.0)            | #1059      |
| text, Qwen2.5-0.5B Q8_0          | `LOCUS_CPU_FAST=1`    | prefill 124 to 127, decode 15.4 to 15.8 (llama.cpp: 226, 13.6)          | #1076      |
| text on V3D                      | do not                | 1B Q4_K_M prefill 1.2, decode 0.7 (neon: 24.6, 10.6)                    | #607, #612 |
| MoE, DeepSeek-V2-Lite Q4_K_M     | default               | 15 decode tokens in 34.6 s; Vulkan 4x slower                            | #679, #252 |
| speech, Orpheus 3B Q4_K_M + SNAC | default, `--ctx 2048` | real-time factor 27 to 29, first audio after 56 to 70 s                 | #245, #984 |
| speech on V3D                    | does not run          | the weights exceed V3D's 1.98 GiB heap                                  | #245, #976 |
| embeddings, bge-small Q8_0       | default               | 60 tokens in 151 ms (V3D: 974 ms)                                       | #692       |
| image VAE, FLUX 128 x 128 decode | default               | 16.5 s (V3D: 55.6 s)                                                    | #393       |
| image DiT (PixArt, FLUX)         | does not fit          | not enough memory in 4 GB                                               | #245       |
| KV cache, Llama 1B Q4_K_M        | `--kv-type f32`       | prefill 33.7, decode 8.1 (q8: 25.2, 8.3; q4: 23.6, 8.1; f16: 20.6, 7.4) | #1059      |
| engine core pinning              | not offered           | no gain on 4 cores with 2 engines                                       | #1015      |

- Prefill trails llama.cpp at 0.3x to 0.5x by default; on Q8_0,
  `LOCUS_CPU_FAST=1` brings it to 0.56x (#1076). The decode columns
  differ in setup: locus decodes after its prompt, llama-bench's tg128
  from an empty context.
- `--kv-type q8` only to fit a longer context: it keeps the decode
  speed and costs about 25% of the prefill.
- `LOCUS_CPU_FAST`: Q8_0 takes the dotprod fast path (four
  interleaved rows per `sdot` lane, the i8mm path's float order): for
  prompt-heavy Q8_0 work, at a decode cost of about 10%. Its faster
  softmax applies too, but is not measured here.
