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

Each section's "Against llama.cpp" table is `scripts/bench/vs-llama.sh`
(#1242): the first 2200 and 8600 bytes of docs/DESIGN.md as prompts,
then 128 decoded tokens, three rounds (locus's median, llama-bench's
mean), the miner stopped. Rerun it when a perf change lands on that
host's path, and replace the table.

## Apple M2 Pro

Match rule: Apple GPU family apple8 (M2). Measured on an M2 Pro: arm64
with neon, dotprod and i8mm; 32 GB unified memory; macOS. Vulkan runs
on MoltenVK.

Recommended: `--backend metal` for text, the default since #1042
(metal for the models its forward takes, else vulkan). Images and
video take the GPU through Vulkan by themselves. Its KV cache stays
f32 by default: f16 (`--kv-type f16`, the M4's default, #1245) is no
gain on this GPU family. Decode tok/s after the prompt, main 6e13558f,
f32 against f16, 3 rounds:

| model, prompt         | f32               | f16               |
|-----------------------|-------------------|-------------------|
| Llama 1B Q8_0, 501    | 105.6 105.7 105.7 | 105.1 104.9 105.2 |
| Llama 1B Q8_0, 2101   | 94.3 82.9 93.8    | 94.8 92.4 94.3    |
| Llama 1B Q4_K_M, 501  | 153.9 155.5 155.8 | 152.3 155.0 153.9 |
| Llama 1B Q4_K_M, 2101 | 131.1 129.0 129.7 | 131.6 133.6 133.0 |
| Qwen2.5 Q8_0, 501     | 184.6 186.8 183.6 | 183.1 184.6 183.5 |
| Qwen2.5 Q8_0, 2123    | 163.5 161.3 164.1 | 159.2 144.4 154.5 |

Concurrent clients
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

- Speech: 17.7 ms a token on Metal before #1060's Q6_K kernel; the
  same request, 784 tokens with sampling, in both.
- PixArt's VAE decode, 128 x 128: CPU 27.2 s, Vulkan 3.09 s, Metal's
  convs 2.83 s (#1055). The server takes Vulkan for it today (#1057).

Against llama.cpp (Homebrew, ggml 1.1.1; 2026-11-27, on AC power):
Metal on main 5443ecf5, the CPU on main 6e13558f; `llama-bench -ngl 0`
still uses Accelerate for its CPU prefill. tok/s:

| model                      | phase                   | Metal  | llama.cpp Metal | share | CPU (neon) | llama.cpp CPU | share |
|----------------------------|-------------------------|--------|-----------------|-------|------------|---------------|-------|
| llama-3.2-1b-q8_0          | prefill 512 (501 tok)   | 1764.8 | 2557.1          | 0.69x | 451.0      | 668.7         | 0.67x |
| llama-3.2-1b-q8_0          | decode 128              | 105.9  | 120.0           | 0.88x | 70.5       | 81.8          | 0.86x |
| llama-3.2-1b-q8_0          | prefill 2048 (2101 tok) | 1642.0 | 2352.9          | 0.70x | 323.9      | 355.0         | 0.91x |
| llama-3.2-1b-q4_k_m        | prefill 512 (501 tok)   | 1608.4 | 2280.4          | 0.71x | 377.6      | 258.3         | 1.46x |
| llama-3.2-1b-q4_k_m        | decode 128              | 155.9  | 171.0           | 0.91x | 102.2      | 107.0         | 0.96x |
| llama-3.2-1b-q4_k_m        | prefill 2048 (2101 tok) | 1505.4 | 1913.6          | 0.79x | 274.9      | 206.3         | 1.33x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 512 (501 tok)   | 4210.8 | 5949.4          | 0.71x | 1072.4     | 1243.7        | 0.86x |
| qwen2.5-0.5b-instruct-q8_0 | decode 128              | 187.7  | 191.7           | 0.98x | 154.8      | 176.2         | 0.88x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 2048 (2123 tok) | 3690.0 | 5264.1          | 0.70x | 722.0      | 652.1         | 1.11x |

Metal prefill is the gap (#1074): the prefill matmuls run at about
3.7 TFLOP/s against llama.cpp's about 4.8.

## Apple M4 Max

Match rule: Apple GPU family apple9 (M3, M4), 32 GB or more of unified
memory. Measured on an M4 Max: 14 CPU cores (10 performance, 4
efficiency), 32 GPU cores; arm64 with neon, dotprod and i8mm; 36 GB;
macOS.

Recommended: `--backend metal` for text, the default since #1042: it
leads Vulkan on every prefill and on most decode rows. Its KV cache is
f16 by default (#1245): Llama 1B decode after a 2101-token prompt,
Q8_0 191.2 -> 200.8 and Q4_K_M 244.8 -> 260.8 tok/s against f32
(llama.cpp at depth 2048: 206.2 and 278.9), prefill and short-context
decode level. On the decode of Orpheus and Qwen3-Coder the two are
within 1%. Concurrent clients decode together since #1184;
locus-server, Llama 1B, 128 tokens a client, tok/s in all:

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
  runs the FLUX DiT on Metal, CLIP, T5 and the VAE on Vulkan (#1057):
  a served 4-step image, warm, 256 x 256 10.7 s against 25.2 s with
  the DiT on Vulkan, 512 x 512 19.8 against 31.2 s, 1024 x 1024 64.8
  against 80.7 s. PixArt the same way: a served 512 x 512, 20-step
  image, warm, denoise 7.7 s against 11.3 s, whole image 9.4 s against
  12.8 s.

Against llama.cpp (Homebrew, ggml 1.1.1; 2026-11-27), main 5443ecf5,
so the CPU rows are before #1232's block-32 change (its Q8_0 prefill
gain is not in them); `llama-bench -ngl 0` uses Accelerate for its CPU
prefill. tok/s:

| model                      | phase                   | Metal   | llama.cpp Metal | share | CPU (neon) | llama.cpp CPU | share |
|----------------------------|-------------------------|---------|-----------------|-------|------------|---------------|-------|
| llama-3.2-1b-q8_0          | prefill 512 (501 tok)   | 4315.5  | 5311.9          | 0.81x | 520.5      | 925.4         | 0.56x |
| llama-3.2-1b-q8_0          | decode 128              | 206.7   | 211.3           | 0.98x | 139.1      | 164.2         | 0.85x |
| llama-3.2-1b-q8_0          | prefill 2048 (2101 tok) | 3946.1  | 3704.9          | 1.07x | 393.7      | 510.5         | 0.77x |
| llama-3.2-1b-q4_k_m        | prefill 512 (501 tok)   | 4118.0  | 5098.4          | 0.81x | 550.7      | 387.1         | 1.42x |
| llama-3.2-1b-q4_k_m        | decode 128              | 269.9   | 288.1           | 0.94x | 172.7      | 177.8         | 0.97x |
| llama-3.2-1b-q4_k_m        | prefill 2048 (2101 tok) | 3877.6  | 4245.0          | 0.91x | 413.5      | 286.0         | 1.45x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 512 (501 tok)   | 10061.8 | 12743.0         | 0.79x | 1119.7     | 1762.7        | 0.64x |
| qwen2.5-0.5b-instruct-q8_0 | decode 128              | 312.6   | 337.5           | 0.93x | 271.3      | 309.5         | 0.88x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 2048 (2123 tok) | 8889.9  | 10755.4         | 0.83x | 847.3      | 838.4         | 1.01x |

llama.cpp's Q8_0 prefill at 2048 is below its own 512 row here, so
that 1.07x is its dip, not a locus gain.

## Intel Core i5-2400 with an NVIDIA GTX 750 Ti

Match rule: x86-64 with SSE4.1 and AVX, no AVX2 and no FMA; an NVIDIA
GPU of compute capability 5.0. Measured on an Intel Core i5-2400
(Sandy Bridge, 4 cores), 32 GB RAM without ECC, and a GTX 750 Ti (2
GB; driver 580.178.04, CUDA 12.6); Ubuntu 24.04, GCC 13.

Automatic: the CPU backend `sse4` (its AVX build). locus does not move
to CUDA by itself: pass `--backend cuda`.

Recommended: `--backend cuda` for dense models that fit in 2 GB (1.7x
to 3.2x the CPU's decode, 9.7x to 12x its prefill); the CPU default
for larger ones.

Against llama.cpp (e8e6c7a, built here: CUDA for sm_50 with `-ngl 99`,
and CPU-only, 4 threads; 2026-11-27), main 6e13558f, `--backend cuda`
and `--backend sse4`. tok/s:

| model                      | phase                   | CUDA  | llama.cpp CUDA | share | CPU (sse4) | llama.cpp CPU | share |
|----------------------------|-------------------------|-------|----------------|-------|------------|---------------|-------|
| llama-3.2-1b-q8_0          | prefill 512 (501 tok)   | 406.5 | 503.3          | 0.81x | 41.0       | 42.6          | 0.96x |
| llama-3.2-1b-q8_0          | decode 128              | 35.9  | 36.1           | 0.99x | 9.7        | 11.0          | 0.88x |
| llama-3.2-1b-q8_0          | prefill 2048 (2101 tok) | 238.2 | 449.4          | 0.53x | 34.6       | 39.6          | 0.87x |
| llama-3.2-1b-q4_k_m        | prefill 512 (501 tok)   | 387.2 | 496.0          | 0.78x | 37.7       | 44.3          | 0.85x |
| llama-3.2-1b-q4_k_m        | decode 128              | 27.1  | 31.5           | 0.86x | 13.5       | 17.0          | 0.79x |
| llama-3.2-1b-q4_k_m        | prefill 2048 (2101 tok) | 229.7 | 448.7          | 0.51x | 32.1       | 41.6          | 0.77x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 512 (501 tok)   | 885.6 | 1186.4         | 0.75x | 98.7       | 107.1         | 0.92x |
| qwen2.5-0.5b-instruct-q8_0 | decode 128              | 54.2  | 80.8           | 0.67x | 21.2       | 25.2          | 0.84x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 2048 (2123 tok) | 428.3 | 994.4          | 0.43x | 77.5       | 97.1          | 0.80x |

On CUDA the long prompt is the gap (the attention over the prefix);
Qwen2.5's decode leaves its small matmuls' launch overhead bare.

| workload                         | default (CPU) | llama.cpp CPU | `--backend cuda` | llama.cpp CUDA | source      |
|----------------------------------|---------------|---------------|------------------|----------------|-------------|
| prefill 512, Llama 3.2 1B Q4_K_M | 31.9          | 46.2          | 381.5            | 499.9          | #412, #1200 |
| decode, Llama 3.2 1B Q4_K_M      | 16.1          | 17.8          | 27.9             | 31.7           | #412, #411  |
| prefill 512, Llama 3.2 1B Q8_0   | 38.4          | 44.0          | 400.9            | 500.1          | #412, #1200 |
| decode, Llama 3.2 1B Q8_0        | 11.6          | 11.9          | 37.5             | 36.2           | #412, #1197 |
| prefill 512, Qwen2.5-0.5B Q8_0   | 86.7          | 115.3         | 843.9            | 1195.6         | #412, #1200 |
| decode, Qwen2.5-0.5B Q8_0        | 25.9          | 27.9          | 55.7             | 80.9           | #412, #1197 |
| prefill 512, Orpheus 3B Q4_K_M   | 12.4          | 15.9          | 32.1             | does not load  | #412, #1200 |
| decode, Orpheus 3B Q4_K_M        | 6.5           | 7.0           | 1.6              | does not load  | #412, #1196 |
| prefill 512, Qwen1.5-MoE Q4_K_M  | 14.2          | 20.1          | does not fit     | not run        | #412        |
| decode, Qwen1.5-MoE Q4_K_M       | 8.1           | 9.1           | does not fit     | not run        | #412        |
| LTX-Video 2B, one DiT step       | 75.3 s        | PyTorch 85 s  | not run          | not run        | #982        |

- Measured on main at 4fb4f9d8 against llama.cpp e8e6c7a (a CPU build,
  4 threads, and a CUDA build for sm_50), median of 3 runs. Decode
  follows the prompt "1, 2, 3, 4, 5,", on which every model decodes all
  127 tokens.
- On the CPU (the "Against llama.cpp" table is the current ratio),
  decode is bound by memory bandwidth: with 16 GB (one module out) it
  ran 32% to 40% slower on both. The MoE rows have the 9.5 GB file in
  the page cache; a cold first prefill runs at 8.5.
- On CUDA, prefill is at 0.71x to 0.80x of llama.cpp: chunks of 32
  tokens or more run as a cuBLAS GEMM (#1200; its rows on main
  c65a89b9 plus that change). Decode is at 0.88x on Q4_K_M, 1.04x on
  Q8_0 and 0.69x on Qwen2.5: the Q8_0 decode kernel takes two rows a
  warp (#1197; its rows on main 098a2590 plus that change). Qwen2.5
  runs on the one-submit forward since #1196; Orpheus per op (below).
- Orpheus (1.94 GB of weights) does not fit the 2 GB card: llama.cpp
  fails to load it, and locus runs it per op, paging its weights
  (slower than the CPU). The one-submit forward declines a model whose
  weights exceed the weight pool, as paging through it on every step
  is slower still (Orpheus prefill 19.3, decode 1.5).
- Tests on this host run with the Vulkan drivers hidden
  (`VK_LOADER_DRIVERS_DISABLE='*'`): Vulkan compute on this card
  faulted (Xid 13) and froze the host twice.
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

Against llama.cpp (a956a85, CPU build, `-ngl 0`, 4 threads;
2026-11-27, started from under 60 C, 69 C at the end, not
throttled), main 6e13558f,
`--backend neon`. llama.cpp has no Vulkan build here. tok/s:

| model                      | phase                   | CPU (neon) | llama.cpp CPU | share |
|----------------------------|-------------------------|------------|---------------|-------|
| llama-3.2-1b-q8_0          | prefill 512 (501 tok)   | 45.1       | 52.7          | 0.86x |
| llama-3.2-1b-q8_0          | decode 128              | 6.1        | 5.8           | 1.06x |
| llama-3.2-1b-q8_0          | prefill 2048 (2101 tok) | 33.6       | 46.1          | 0.73x |
| llama-3.2-1b-q4_k_m        | prefill 512 (501 tok)   | 39.2       | 67.1          | 0.58x |
| llama-3.2-1b-q4_k_m        | decode 128              | 8.5        | 8.1           | 1.05x |
| llama-3.2-1b-q4_k_m        | prefill 2048 (2101 tok) | 32.3       | 58.2          | 0.55x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 512 (501 tok)   | 113.0      | 190.2         | 0.59x |
| qwen2.5-0.5b-instruct-q8_0 | decode 128              | 15.1       | 10.7          | 1.41x |
| qwen2.5-0.5b-instruct-q8_0 | prefill 2048 (2123 tok) | 84.9       | 158.0         | 0.54x |

llama.cpp's load-time repack is most of its prefill lead here (its
pp512 with `--repack 0` against `1`: Q8_0 1.40x, Q4_K_M 1.68x, Q4_0
2.99x, Qwen2.5 Q8_0 2.16x; #1243).

| workload                         | command               | number                                                                  | source     |
|----------------------------------|-----------------------|-------------------------------------------------------------------------|------------|
| text, Llama 3.2 1B Q4_K_M        | default               | prefill 34 to 38, decode 10.7 to 11.0 (llama.cpp: 66, 8.1)              | #412       |
| text, Llama 3.2 1B Q8_0          | default               | prefill 34 to 35, decode 6.9 (llama.cpp: 52, 6.0)                       | #412       |
| text, Qwen2.5-0.5B Q8_0          | default               | prefill 71 to 89, decode 16.6 to 17.5 (llama.cpp: 205, 12.1)            | #412       |
| text, Qwen2.5-0.5B Q8_0          | `LOCUS_CPU_FAST=1`    | prefill 124 to 127, decode 15.4 to 15.8 (llama.cpp: 226, 13.6)          | #1076      |
| text on V3D                      | do not                | 1B Q4_K_M prefill 1.3, decode 0.7 (neon: 35, 10.7)                      | #412       |
| MoE, DeepSeek-V2-Lite Q4_K_M     | default               | 15 decode tokens in 34.6 s; Vulkan 4x slower                            | #679, #252 |
| speech, Orpheus 3B Q4_K_M + SNAC | default, `--ctx 2048` | real-time factor 27 to 29, first audio after 56 to 70 s                 | #245, #984 |
| speech on V3D                    | does not run          | the weights exceed V3D's 1.98 GiB heap                                  | #245, #976 |
| embeddings, bge-small Q8_0       | default               | 60 tokens in 151 ms (V3D: 974 ms)                                       | #692       |
| image VAE, FLUX 128 x 128 decode | default               | 16.5 s (V3D: 55.6 s)                                                    | #393       |
| image DiT (PixArt, FLUX)         | does not fit          | not enough memory in 4 GB                                               | #245       |
| KV cache, Llama 1B Q4_K_M        | `--kv-type f32`       | prefill 33.7, decode 8.1 (q8: 25.2, 8.3; q4: 23.6, 8.1; f16: 20.6, 7.4) | #1059      |
| engine core pinning              | not offered           | no gain on 4 cores with 2 engines                                       | #1015      |

- The text rows: main at 2f96456b against llama.cpp a956a85 (a CPU
  build, 4 threads), the range of 3 runs; prefill is a prompt of 505
  or 506 tokens, decode 127 tokens after "1, 2, 3, 4, 5,". bitcoind and
  other processes left about 1.8 GB free, and the prefill runs vary up
  to 20% (Qwen2.5 71 to 89).
- The text rows above are older (main 2f96456b); the "Against
  llama.cpp" table is the current comparison. The decode columns
  differ in setup there too: locus decodes after its prompt,
  llama-bench's tg128 from an empty context.
- On V3D, the 1B Q8_0 prefill is killed out of memory (4 GB shared with
  the CPU), and Orpheus 3B fails to allocate its heap, as speech on
  V3D does.
- Orpheus 3B as text ran at prefill 2.3 to 2.8 and decode 0.4 to 1.8
  here, against llama.cpp's 18.0 and 3.65 on the same host: not yet
  explained, so it has no row.
- `--kv-type q8` only to fit a longer context: it keeps the decode
  speed and costs about 25% of the prefill.
- The `LOCUS_CPU_FAST=1` Q8_0 row above was measured when the four
  interleaved row `sdot` prefill was opt-in; it is the default now
  (#1076), while decode keeps the one-row kernel, in the same float
  order. The flag still turns on the faster softmax, not measured
  here.
