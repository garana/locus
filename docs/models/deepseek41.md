# DeepSeek-V4.1-Flash (deepseek41): forward-pass reference

This is the per-token decode math locus implements in
src/locus/model/deepseek41.cpp. It was written from the draft llama.cpp
implementation (ggml-org/llama.cpp#30817, branch `deepseek41` of
danielhanchen/llama.cpp at `b72050d03ae14bb3cb5138ecefab3f3f65157d31`).
Citations `file:N` refer to a line in that tree.

The GGUF facts come from the UD-IQ1_S GGUF header
(unsloth/DeepSeek-V4.1-Flash-1017-GGUF): the KV metadata and the full tensor
list (1043 tensors with types).

Notation. Math is row-major and 0-based. `W·x` means `ggml_mul_mat(W, x)`: a
GGUF tensor with ggml shape `[in, out]` maps an `in` vector to an `out` vector.
`rms(x, w) = x / sqrt(mean(x^2) + eps) * w` uses eps = `f_norm_rms_eps` =
1e-20, the same eps for every RMS norm in the model (`norm_rms_eps`). Plain
`rms(x)` has no weight. `D = 5120` (n_embd), `HC = 4`, `H = 64` heads,
`Dh = 512` (head dim), `R = 64` (rope dims), `Dn = Dh - R = 448` (nope dims),
`Dq = 1280` (q_lora_rank), `Hi = 32` (indexer heads), `Di = 128` (indexer
head dim).

---------------------------------------------------------------------------

## 1. Hyperparameters, tensors and layer classification

### 1.1 GGUF keys (prefix `deepseek41.`) and the values in UD-IQ1_S

In the key column, `a.` stands for `attention.`.

| key                                  | type        | value                   |
|--------------------------------------|-------------|-------------------------|
| block_count                          | u32         | 40                      |
| context_length                       | u32         | 1048576                 |
| embedding_length                     | u32         | 5120                    |
| embedding_length_out                 | u32         | 20480 (=4*D, unused)    |
| a.head_count / head_count_kv         | u32         | 64 / 1                  |
| a.key_length / value_length          | u32         | 512 / 512               |
| rope.dimension_count                 | u32         | 64                      |
| rope.freq_base                       | f32         | 10000                   |
| rope.scaling.type                    | str         | yarn                    |
| rope.scaling.factor                  | f32         | 16                      |
| rope.scaling.original_context_length | u32         | 65536                   |
| rope.scaling.yarn_beta_fast / _slow  | f32         | 32 / 1                  |
| a.layer_norm_rms_epsilon             | f32         | 1e-20 (9.9999997e-21)   |
| a.q_lora_rank                        | u32         | 1280                    |
| a.sliding_window                     | u32         | 128                     |
| a.compress_ratios                    | i32[43]     | see 1.3                 |
| a.compress_rope_freq_base            | f32         | 160000                  |
| a.kv_source_layers                   | u32[]       | [2, 8, 14, 20]          |
| a.indexer.source_layers              | u32[]       | [2,8,14,20,24,28,32,36] |
| a.indexer.head_count                 | u32         | 32                      |
| a.indexer.key_length                 | u32         | 128                     |
| a.indexer.top_k                      | u32         | 512                     |
| a.indexer.candidate_source_layer     | i32         | 20                      |
| a.indexer.candidate_top_k_blocks     | u32         | 2048                    |
| a.indexer.candidate_block_size       | u32         | 8                       |
| a.output_group_count                 | u32         | 8                       |
| a.output_lora_rank                   | u32         | 1024                    |
| expert_count / expert_used_count     | u32         | 384 / 6                 |
| expert_shared_count                  | u32         | 1                       |
| expert_feed_forward_length           | u32         | 2304                    |
| expert_gating_func                   | u32         | 4 (sqrt-softplus)       |
| expert_weights_scale                 | f32         | 1.5                     |
| expert_weights_norm                  | bool        | true                    |
| swiglu_clamp_exp / _shexp            | f32[40]     | all 10.0                |
| hyper_connection.count               | u32         | 4                       |
| hyper_connection.sinkhorn_iterations | u32         | 20                      |
| hyper_connection.epsilon             | f32         | 1e-6                    |
| hash_layer_count                     | u32         | 0                       |
| ple.layers                           | u32[2]      | [1, 14]                 |
| ple.ngram_size                       | u32         | 4                       |
| ple.heads_per_ngram                  | u32         | 8                       |
| embedding_length_per_layer_input     | u32         | 256                     |
| ple.layer_multipliers                | u64[8]      | 2 layers x 4            |
| ple.head_offsets                     | u64[48]     | 2 layers x 3 x 8        |
| ple.head_vocab_sizes                 | u64[48]     | primes ~16000057..      |
| engram.pad_token_id                  | u32         | 2                       |
| engram.token_map                     | i32[129280] | token -> compressed     |
| engram.vocab_size                    | u32         | 16000000 (not read)     |
| engram.compressed_vocab_size         | u32         | 99092 (not read)        |
| dspark.*                             | various     | ignore (section 9)      |

Readers: `src/models/deepseek4.cpp:11-72` (base V4 keys) and
`src/models/deepseek41.cpp:25-120` (V4.1 keys). Notes:

- `expert_gating_func` must be 4; the loader rejects anything else
  (`deepseek4.cpp:55-58`).
- `swiglu_clamp_shexp` falls back to `swiglu_clamp_exp` when absent
  (`deepseek4.cpp:29-31`).
- `n_embd_out_impl = n_embd`: V4.1 has no `hc_head` (`deepseek41.cpp:28`).
- The hyper-connection count must be 4 (`deepseek4.cpp:370`).
- The loader requires `indexer.top_k > 0`.
- Every Engram bucket count must be at least 2, and offset + count must fit in
  an int32 (`deepseek41.cpp:110-115`).
- `rope.freq_scale = 1/factor = 1/16`. `ext_factor = 1.0` because the scaling
  type is yarn (`llama-context.cpp:259-260`).
- llama.cpp's own `yarn_attn_factor` is not used. The model passes its own
  attn_factor (section 4.1).

### 1.2 Global tensors

Types are those in UD-IQ1_S.

| name                        | ggml shape       | type   | notes             |
|-----------------------------|------------------|--------|-------------------|
| token_embd.weight           | [5120, 129280]   | Q6_K   | row lookup        |
| output_norm.weight          | [5120]           | F32    |                   |
| output.weight               | [5120, 129280]   | Q6_K   | lm head           |
| per_layer_token_embd.weight | [256, 768022850] | IQ4_NL | all Engram tables |

There are no `output_hc_*` tensors (no hc_head). Loader:
`deepseek41.cpp:137-156`. `per_layer_token_embd` is created `TENSOR_READ_LAZY`
and must have at least `max(offset+size)` = 752021961 + 16000889 = 768022850
rows. In this GGUF it is the whole of shard 5 (110.6 GB).

### 1.3 Per-layer arrays and the layer classes

`compress_ratios` (43 entries; only the first 40 are used, the last 3 belong
to the absent MTP/DSpark stages):

    [0,0, 2 x18 (layers 2..19), 1 x20 (layers 20..39), 0,0,0]

The derived per-layer `kv_src[il]` and `topk_src[il]` come from
`deepseek41.cpp:46-68`. Each compressed layer reads the KV, or the top-k, of
the latest source layer at or before it:

```
last_kv = last_idx = -1
for il in 0..n_layer-1:
    if il in kv_source_layers:      last_kv  = il
    if il in indexer_source_layers: last_idx = il
    if ratio[il] == 0: continue                 # kv_src = topk_src = -1
    require ratio <= 2, last_kv >= 0, last_idx >= last_kv,
            ratio[last_kv] == ratio[il] == ratio[last_idx]
    kv_src[il] = last_kv; topk_src[il] = last_idx
```

The loader also rejects any source layer whose ratio is 0
(`deepseek41.cpp:54-56`). The resulting classes are below; `r` is the ratio,
`kv` is `kv_src` and `topk` is `topk_src`:

| layers | r | kind              | kv | topk | owns                            |
|--------|---|-------------------|----|------|---------------------------------|
| 0,1    | 0 | SWA only          | -1 | -1   | window cache                    |
| 2      | 2 | CSA, kv+idx src   | 2  | 2    | comp cache+state, indexer q/w   |
| 3..7   | 2 | CSA consumer      | 2  | 2    | window only                     |
| 8      | 2 | CSA, kv+idx src   | 8  | 8    | as layer 2                      |
| 9..13  | 2 | CSA consumer      | 8  | 8    |                                 |
| 14     | 2 | CSA, kv+idx src   | 14 | 14   | as layer 2                      |
| 15..19 | 2 | CSA consumer      | 14 | 14   |                                 |
| 20     | 1 | HCA, kv+idx src   | 20 | 20   | comp cache+state, idx, cand src |
| 21..23 | 1 | HCA consumer      | 20 | 20   |                                 |
| 24     | 1 | idx src, cand use | 20 | 24   | indexer q/w only                |
| 25..27 | 1 | consumer          | 20 | 24   |                                 |
| 28     | 1 | idx src           | 20 | 28   |                                 |
| 29..31 | 1 | consumer          | 20 | 28   |                                 |
| 32     | 1 | idx src           | 20 | 32   |                                 |
| 33..35 | 1 | consumer          | 20 | 32   |                                 |
| 36     | 1 | idx src           | 20 | 36   |                                 |
| 37..39 | 1 | consumer          | 20 | 36   |                                 |

"CSA" means the ratio-2 cache slot and "HCA" the ratio-1 slot; these are
llama.cpp names (`llama-kv-cache-dsv4.cpp:1334-1337`). Every layer, compressed
or not, also has its own sliding-window cache of raw tokens.

The candidate source is layer 20. It must be an index source of itself
(`topk_src[20] == 20`), and every later index source must share its ratio
(`deepseek41.cpp:70-84`).

The Engram layers are 1 and 14 (`ple.layers`). The Engram index is
`e = position in ple.layers`, so e=0 is layer 1 and e=1 is layer 14
(`deepseek41.cpp:221-224`).

### 1.4 Per-layer tensors (`blk.N.`)

Loader: `deepseek41.cpp:158-209`. Each row gives the ggml shape
`[ne0, ne1, ...]` and the type in UD-IQ1_S.

All 40 layers:

| tensor                | shape             | IQ1_S type                  |
|-----------------------|-------------------|-----------------------------|
| attn_norm.weight      | [5120]            | F32                         |
| attn_sinks.weight     | [64]              | F32                         |
| attn_q_a.weight       | [5120, 1280]      | Q6_K (Q5_K L27-39)          |
| attn_q_a_norm.weight  | [1280]            | F32                         |
| attn_q_b.weight       | [1280, 32768]     | Q6_K / Q5_K                 |
| attn_kv.weight        | [5120, 512]       | Q6_K / Q5_K                 |
| attn_kv_a_norm.weight | [512]             | F32                         |
| attn_output_a.weight  | [4096, 8192]      | Q6_K / Q5_K                 |
| attn_output_b.weight  | [8192, 5120]      | Q6_K / Q5_K                 |
| hc_attn_fn.weight     | [20480, 24]       | F32                         |
| hc_attn_base.weight   | [24]              | F32                         |
| hc_attn_scale.weight  | [3]               | F32                         |
| hc_ffn_fn.weight      | [20480, 24]       | F32                         |
| hc_ffn_base.weight    | [24]              | F32                         |
| hc_ffn_scale.weight   | [3]               | F32                         |
| ffn_norm.weight       | [5120]            | F32                         |
| ffn_gate_inp.weight   | [5120, 384]       | F32                         |
| exp_probs_b.bias      | [384]             | F32                         |
| exp_probs_b_vl.bias   | [384] (optional)  | F32                         |
| ffn_gate_exps.weight  | [5120, 2304, 384] | IQ1_S / IQ2_XXS             |
| ffn_up_exps.weight    | [5120, 2304, 384] | IQ1_S / IQ2_XXS             |
| ffn_down_exps.weight  | [2304, 5120, 384] | IQ1_S L0-19, IQ2_XXS L20-39 |
| ffn_gate_shexp.weight | [5120, 2304]      | Q6_K / Q8_0                 |
| ffn_up_shexp.weight   | [5120, 2304]      | Q6_K / Q8_0                 |
| ffn_down_shexp.weight | [2304, 5120]      | Q6_K / Q8_0                 |

`attn_output_a` is stored 2-D as `[4096, 8192]` and loaded as
`[4096, 1024, 8]` with `TENSOR_ALLOW_RESHAPE` (`deepseek41.cpp:168`). Group g
is file rows `1024*g .. 1024*g+1023`.

Only when `kv_src[il] == il` (layers 2, 8, 14, 20; `deepseek41.cpp:178-186`):

| tensor                      | shape       | type | condition      |
|-----------------------------|-------------|------|----------------|
| attn_compressor_kv.weight   | [5120, 512] | BF16 |                |
| attn_compressor_norm.weight | [512]       | F32  |                |
| attn_compressor_gate.weight | [5120, 512] | BF16 | ratio > 1 only |
| indexer.attn_k.weight       | [512, 128]  | BF16 |                |
| indexer.k_norm.weight       | [128]       | F32  |                |

`attn_compressor_gate` therefore exists for layers 2, 8 and 14, but not for 20.

Only when `topk_src[il] == il` (layers 2, 8, 14, 20, 24, 28, 32, 36;
`deepseek41.cpp:187-190`):

| tensor                  | shape        | type |
|-------------------------|--------------|------|
| indexer.attn_q_b.weight | [1280, 4096] | BF16 |
| indexer.proj.weight     | [5120, 32]   | BF16 |

Only on the Engram layers 1 and 14 (`deepseek41.cpp:192-195`):

| tensor             | shape         | type | notes                           |
|--------------------|---------------|------|---------------------------------|
| engram_kv.weight   | [6144, 25600] | Q6_K | 6144 = 24 cols*256, 25600 = 5*D |
| engram_gate.weight | [5120, 4]     | F32  | = q_weight*k_weight, folded     |

The full set of types in this GGUF is F32, BF16, Q5_K, Q6_K, Q8_0, IQ1_S,
IQ2_XXS and IQ4_NL. It contains no MXFP8 and no MXFP4 (section 8).

---------------------------------------------------------------------------

## 2. Embedding and Engram (n-gram hashed embeddings)

### 2.1 Token embedding

`x0 = dequant(token_embd[token])`, a 5120-vector with no scaling
(`build_inp_embd(model.tok_embd, 1.0f, ...)`, `deepseek41.cpp:786`). The
hyper-connection streams start as 4 identical copies:
`X[h] = x0` for h = 0..3 (`deepseek41.cpp:834`).

### 2.2 Engram hash (host side, per Engram layer e, per token)

Constants: `N = ngram_size = 4`, `NH = heads_per_ngram = 8`,
`ncols = (N-1)*NH = 24`, and `hd = 256`. The slices for layer e are
`mult = layer_multipliers[e*N .. e*N+3]`, `prime = head_vocab_sizes[e*24 ..]`
and `off = head_offsets[e*24 ..]`. The offsets are absolute rows in
`per_layer_token_embd`. Layer 1 uses rows 0..384M and layer 14 uses the rest.

Code: `deepseek41.cpp:226-264`.

```
cid(t) = (t < 0 || t >= len(token_map)) ? DEAD : token_map[t]   # DEAD = -1
pad    = token_map[engram_pad_id]          # pad_id = 2; token_map[2] == 2
toks[0] = cid(token at pos p)              # embedding input -> DEAD
blocked = (toks[0] == DEAD); if blocked: toks[0] = pad
for s in 1..N-1:
    # from the sequence history
    c = (p - s < 0) ? DEAD : cid(token at pos p - s)
    blocked = blocked || c == DEAD
    toks[s] = blocked ? pad : c
rolling = toks[0] * mult[0]                            # uint64, wraps mod 2^64
for o in 1..N-1:                                     # n-gram order o+1 = 2,3,4
    rolling ^= toks[o] * mult[o]
    for h in 0..NH-1:
        col = (o-1)*NH + h
        row[col] = (int32)(rolling % prime[col] + off[col])
```

Once a shift is blocked, every longer shift is also padded, so at sequence
start pos 0 gives `[c0, pad, pad, pad]`, pos 1 gives `[c1, c0, pad, pad]`, and
so on. The token at `pos - s` comes from the raw-window cache cells, which keep
token ids (`llama-kv-cache.cpp:1854-1908`;
`prev[i*n + j]` = token at `pos - (n-j)`). In the hash,
`prev[i*n_prev + (n_prev - s)]` is the token at `pos - s`. A paged cache must
therefore keep the last `N-1 = 3` token ids per sequence.

The multipliers are odd and bounded by `INT64_MAX / compressed_vocab / 2`
(`conversion/deepseek41.py:49-57`), so a product never overflows int64 and the
uint64 math is exact. The token map and multipliers are stored in the GGUF
because they need HF tokenizer normalizers and numpy PCG64
(`conversion/deepseek41.py:20-21, 60-88`).

### 2.3 Engram lookup and residual injection (layers 1 and 14)

Engram runs at the start of layer il, before the attention hyper-connection
mixes (`deepseek41.cpp:843-847`). It acts on all 4 streams.
`build_engram`, `deepseek41.cpp:369-410`:

```
# 6144, col-major blocks of 256
E    = concat_{col=0..23} dequant(per_layer_token_embd[row[col]])
kv   = engram_kv · E                                  # 25600
key[h] = kv[h*D : (h+1)*D]        for h = 0..3         # 4 keys
val    = kv[4*D : 5*D]                                 # one value
for h in 0..3:
    dot_h  = sum_j rms(X[h])_j * rms(key[h])_j * engram_gate[h][j]  / sqrt(D)
    gate_h = sigmoid( sign(dot_h) * sqrt(max(|dot_h|, 1e-6)) )   # sign(0) = +1
    X[h]  += gate_h * val
```

Quoted:

```
deepseek41.cpp:390  key   = view_3d(kv, n_embd, hc, nt, ..., 0)
deepseek41.cpp:391  value = view_3d(kv, n_embd, 1,  nt, ...,
                                    row_size(hc*n_embd))
deepseek41.cpp:396  dot = mul(mul(rms_norm(x, eps), rms_norm(key, eps)), egate)
deepseek41.cpp:397  dot = scale(sum_rows(dot), 1.0f/sqrtf(n_embd))
deepseek41.cpp:401  mag  = sqrt(clamp(abs(dot), 1e-6f, INFINITY))
# dot<0 -> -1 else +1
deepseek41.cpp:402  sign = scale_bias(step(neg(dot)), -2.0f, 1.0f)
deepseek41.cpp:403  gate = sigmoid(mul(mag, sign))
deepseek41.cpp:409  return add(x, mul(repeat(value, n_embd, hc, nt), gate))
```

Details:

- The rms norms have no weight. They use eps = 1e-20 and normalize each
  5120-wide stream separately.
- `engram_gate` (ggml `[5120, 4]`) holds 4 rows of 5120, one per stream.
- The table type in UD-IQ1_S is IQ4_NL, which dequantizes to F32 through
  get_rows.
- Image or embedding tokens get gate 0 (`engram_keep`). This is not needed for
  text.

---------------------------------------------------------------------------

## 3. Hyper-connections (mHC, 4 streams, "single-pass")

The state is `X[4][D]`. `flat(X)` is the stream-major concatenation
`flat[h*D + j] = X[h][j]` (20480 values).

### 3.1 Mixes (one set per sublayer, computed from that sublayer's input)

`build_hc_mixes`, `deepseek4.cpp:356-416`, with `f32_src = true`:

```
# hc_fn [20480, 24] F32 -> m[24]; rms no weight, eps 1e-20
m     = hc_fn · rms(flat(X))
# j = 0..3     (hc_eps = 1e-6)
pre[j]  = sigmoid(m[j]   * scale[0] + base[j])   + hc_eps
# j = 0..3
post[j] = 2 * sigmoid(m[4+j] * scale[1] + base[4+j])
# k = 0..15
C[k]    = m[8+k] * scale[2] + base[8+k]
          interpreted as C[dst][src] with k = dst + 4*src
comb = sinkhorn(C)
```

Quoted:

```
deepseek4.cpp:373-375
    flat = reshape(x, hc_dim, nt); flat_norm = rms_norm(flat, eps);
    mixes = mul_mat(hc_fn, flat_norm)
deepseek4.cpp:376-379
    F32 src precision requested ("model.py hc_pre runs in fp32")
deepseek4.cpp:388-391
    pre = sigmoid(mixes[0:4]*scale[0] + base[0:4]); pre = pre + hc_eps
deepseek4.cpp:394-397
    post = 2*sigmoid(mixes[4:8]*scale[1] + base[4:8])
deepseek4.cpp:405-411
    comb = reshape_3d(mixes[8:24]*scale[2] + base[8:24], hc, hc, nt);
    sinkhorn
```

Precision: keep the whole HC path in F32. That covers the 20480-wide RMS norm,
the F32 weights and F32 activations.

### 3.2 Sinkhorn (`deepseek4.cpp:308-343`)

`comb` has ggml shape `[dst = ne0, src = ne1]`. Write it as `M[dst][src]`:

```
# ggml_soft_max on ne0, per src column
M[dst][src] = softmax over dst of C[.][src]
M += eps                                               # eps = hc_eps = 1e-6
# "norm_cols"
normalize over src:  M[d][s] /= (sum_s' M[d][s'] + eps)
repeat (iters - 1) = 19 times:
    # "norm_rows"
    normalize over dst: M[d][s] /= (sum_d' M[d'][s] + eps)
    # "norm_cols"
    normalize over src: M[d][s] /= (sum_s' M[d][s'] + eps)
```

In total that is one softmax, one eps add, 20 src-normalizations and 19
dst-normalizations. Quoted:

```
deepseek4.cpp:315  comb = ggml_soft_max(ctx0, comb);
deepseek4.cpp:320  comb = ggml_add(ctx0, comb, eps);
deepseek4.cpp:322-328
    norm_cols: permute(1,0) -> sum_rows -> +eps -> permute back -> div
deepseek4.cpp:330-334
    norm_rows: sum_rows(comb) + eps -> div
deepseek4.cpp:336-340
    norm_cols(); for (i = 1; i < iters; ++i) { norm_rows(); norm_cols(); }
```

Here `sum_rows` sums over ne0, which is the dst axis. "norm_rows" therefore
divides each src column by its sum over dst. "norm_cols" permutes first, so it
divides each dst row by its sum over src.

### 3.3 Collapse (pre) and write-back (post)

```
# deepseek4.cpp:297-305
hc_pre(X, w)       = sum_{h=0..3} w[h] * X[h]
# deepseek4.cpp:436-452
hc_post(y, Xres, post, M)[dst] = post[dst]*y + sum_{src} M[dst][src] * Xres[src]
```

In `hc_post`, `comb_src_dst` is the view at `dst*nb0 + src*nb1`, which is
element `C[dst + 4*src]` = `M[dst][src]`.

### 3.4 Lagged ("single-pass") mHC order: differs from V4

In V4.1 the pre-mix that collapses a sublayer's input comes from the previous
sublayer's mixes. The post and comb weights still come from the sublayer's own
input. `deepseek41.cpp:837-935` (trailing numbers are line numbers):

```
pre_prev = None
for il in 0..39:
    if il in {1,14}: X = engram(X)                                 # 843-847
    Xres = X
    (attn_pre, post_a, M_a) = hc_mixes(X, hc_attn_fn/scale/base)   # 860
    h = (pre_prev is None) ? X[0] : hc_pre(X, pre_prev)            # 863
    h = rms(h, attn_norm)                                          # 866
    a = attention(h)                                               # 869
    X = hc_post(a, Xres, post_a, M_a)                              # 871
    Xres = X
    (pre_prev, post_f, M_f) = hc_mixes(X, hc_ffn_fn/scale/base)    # 875
    # attention mixes, post-attention streams
    h = hc_pre(X, attn_pre)                                        # 877
    h = rms(h, ffn_norm)                                           # 884
    f = moe(h) + shared(h)                                         # 888-922
    X = hc_post(f, Xres, post_f, M_f)                              # 925
# last layer's FFN pre-mix, no hc_head
out = hc_pre(X, pre_prev)                                          # 935
```

The FFN of layer il is collapsed with `attn_pre`, computed from the
pre-attention streams of layer il. The attention of layer il+1 is collapsed
with `pre_prev`, computed from the pre-FFN streams of layer il. Layer 0's
attention uses copy 0 with weight `[1,0,0,0]`, not the sum
(`deepseek41.cpp:862-863`: "the first layer's attention collapses with the
identity pre-mix: copy 0"). The DSpark converter documents this as
`hyper_connection.single_pass` ("lagged pre-mix, no hc_head";
`llama-hparams.h:329`).

---------------------------------------------------------------------------

## 4. Attention (`build_attention41`, `deepseek41.cpp:412-768`)

The input `h` is the attn_norm output (5120). `p` is the token position.

### 4.1 RoPE parameters per layer (`deepseek41.cpp:433-446`)

| layer ratio    | freq_base | freq_scale | ext_factor | n_ctx_orig | beta f/s |
|----------------|-----------|------------|------------|------------|----------|
| 0 (L0, L1)     | 10000     | 1.0        | 0          | 0          | 0/0      |
| != 0 (L2..L39) | 160000    | 1/16       | 1.0        | 65536      | 32/1     |

`attn_factor = dsv4_rope_attn_factor(freq_scale, ext_factor)`
(`models.h:1286-1293`). It is 1 when `ext_factor == 0` and otherwise
`1/(1+0.1*ln(1/freq_scale))` = 0.78293. That exactly cancels ggml's YaRN
mscale, so the net cos/sin magnitude is 1 ("the DeepSeek-V4 reference applies
none"). The same per-layer parameters apply to all of: q, the window kv, the
compressed rows, the index q, the index k, and the inverse rope of the output.

Rope type is NORM: adjacent pairs `(x[2i], x[2i+1])` (`llama-model.cpp`
rope_type case for DEEPSEEK41 returns `LLAMA_ROPE_TYPE_NORM`).
`ggml_rope_set_offset(t, n_off)` rotates the `R = 64` dims starting at
`n_off`, so only the last 64 dims rope and the rest pass through
(`ops.cpp:6505-6533`):

```
for pair i in 0..31:   (dims n_off+2i, n_off+2i+1)
    theta_extrap = p * base^(-2i/64)
    theta_interp = freq_scale * theta_extrap
    if ext_factor != 0:
        # ops.cpp:6252-6255, i = i0/2
        ramp = 1 - clamp((i - lo)/max(0.001, hi - lo), 0, 1)
        theta = theta_interp*(1-ramp) + theta_extrap*ramp
    else theta = theta_interp
    # magnitude 1 (see above)
    (x0, x1) -> (x0 cos - x1 sin, x0 sin + x1 cos)
```

`lo` and `hi` come from `ggml_rope_yarn_corr_dims` (`ggml.c:4498-4510`):

    corr(b) = 64 * ln(65536/(b*2*pi)) / (2*ln(base))
    lo = max(0, floor(corr(32))), hi = min(63, ceil(corr(1)))

For base 160000 this gives lo = 15 (15.45) and hi = 25 (24.71). Pairs 0..15
keep the original frequency, pairs 25 and up are divided by 16, and pairs in
between are linearly mixed.

Inverse rope (`ggml_rope_ext_back`, `deepseek41.cpp:754-756`) uses the same
cache with the sign of sin flipped (`ops.cpp`: `sin_sign = forward ? 1 : -1`).

### 4.2 Q, KV and window projections

Trailing numbers are `deepseek41.cpp` line numbers.

```
qr = rms(attn_q_a · h, attn_q_a_norm)              # 1280                  448
# head-major: q[hd*512 + d]
q  = attn_q_b · qr  -> q[64][512]                                         # 451
# last 64 dims of each head
q  = rope(q, p, offset 448)                                               # 452
kv = rms(attn_kv · h, attn_kv_a_norm)              # 512, single MQA head  455
kv = rope(kv, p, offset 448)                                              # 456
```

V4.1 has no per-head RMS norm on q. V4 has one
(`deepseek4.cpp:926 q = ggml_rms_norm(q)`); V4.1 dropped it (open question
Q1). K and V are the same tensor (`kv`), MQA with 1 kv head and 512 dims.
`kq_scale = 1/sqrt(512)` (`deepseek41.cpp:430`).

`kv` is appended to this layer's own window cache (`cpy_k`,
`deepseek41.cpp:650`, or `build_raw_attention` for ratio 0). Every layer keeps
its own window.

The window mask (`llama-hparams.h:525`, `swa_type STANDARD`, `n_swa = 128`)
masks a key at position p0 for a query at p1 when `p1 - p0 >= 128`, plus the
usual causal mask. The visible window is therefore the 128 positions
`p-127 .. p`, including the current token.

### 4.3 Ratio-0 layers (0, 1)

`out = MHA(q, K=V=window rows, mask=window, sinks, kq_scale)`
(`deepseek41.cpp:524-526`, `build_raw_attention`, `deepseek4.cpp:831-865`).

### 4.4 Compressed KV production (kv sources 2, 8, 14 with r=2; 20 with r=1)

This is `deepseek41.cpp:465-515`. Plan semantics come from
`llama-kv-cache-dsv4.cpp:557-611`.

Per token at position p, every kv source computes:

```
s_kv[p]    = attn_compressor_kv   · h                 # 512, F32 state
s_score[p] = attn_compressor_gate · h  (r = 2)        # 512
           = 0                         (r = 1)        # ggml_fill(.., 0)   470
```

Both are persisted in a per-sequence F32 ring of `r` slots at index
`p % r` (`state_size = r`, `llama-kv-cache-dsv4.cpp:1436-1449`,
`state_persist_*`).

A compressed row completes when `(p+1) % r == 0` (`kv-cache-dsv4.cpp:586`).
The block is `b = p / r` and covers source tokens `t = p+1-r .. p`
(`source_start`):

```
# softmax over the r tokens, per channel
for each channel c in 0..511:
    w_t   = softmax_t( s_score[t][c] )                           # 489-492
    comp[c] = sum_t w_t * s_kv[t][c]
comp = rms(comp, attn_compressor_norm)                           # 494
# 128, from the UNROPED comp                                       498
ik   = rms(indexer.attn_k · comp, indexer.k_norm)
ik   = rope(ik,   pos = b*r, offset 128-64 = 64)                 # 499
comp = rope(comp, pos = b*r, offset 448)                         # 500
# -> compressed cache row b (640 wide)                             502-503
row[b] = [comp (512) | ik (128)]
```

The rope position is the block's first token, `state_write_pos = source_start`
(`kv-cache-dsv4.cpp:590-594`; graph comment "a latent stands for the first
token of its group", `deepseek41.cpp:497`). V4.1 has no APE (V4's
`attn_compressor_ape` is absent) and no overlapping windows, so the state is
`r` rows of 512, not 2r rows of 1024 (`kv-cache-dsv4.cpp:1334-1337`).

For r=1 (layer 20), every token immediately yields row `b = p`, equal to
`rms(attn_compressor_kv·h, norm)` roped at p, with an index key.

For r=2 at decode:

- At even p: store `s_kv` and `s_score` in slot 0. No row is produced.
- At odd p: read slot 0 (token p-1) together with the current token's values
  (slot 1, which is also persisted), and write row p/2.

Visibility: a query at position p sees compressed rows `j < n_visible =
(p+1) / r` (`kv-cache-dsv4.cpp:566-567`; mask `llama-graph.cpp:912-917`, 0 if
`j < n_visible` else -inf). A row completed by the current token is visible to
it.

Sharing: a consumer layer reads `kv_src[il]`'s compressed cache. Layers 3..7
read layer 2's rows, and so on (`deepseek41.cpp:463`, `530`). The row values
come from the source layer's own `h`, meaning its attn_norm output. Consumers
never compute compressed rows.

### 4.5 Indexer: top-k over compressed rows (index-source layers only)

`topk_src[il] == il` (`deepseek41.cpp:535-639`). Layers 2, 8 and 14 index
their own CSA cache. Layers 20, 24, 28, 32 and 36 index layer 20's HCA cache.
Below, `n_comp` is the compressed row count and `ik_j = row[j][512:640]`:

```
iq = indexer.attn_q_b · qr  -> iq[32][128]                 # qr from 4.2   547
iq = rope(iq, p, offset 64)                                              # 548
iw = (indexer.proj · h) * (1/sqrt(128*32))  -> iw[32]   (= /64)          # 551
# 584-592
score[j] = sum_{hd=0..31} relu( dot(ik_j, iq[hd]) ) * iw[hd]  + mask[j]
           mask[j] = 0 if j < n_visible else -inf
```

The qk dot is computed with F32 precision (`ggml_prec_set_src(kq, F32)`,
`deepseek41.cpp:585-587`). It has no extra 1/sqrt(Di). The `iw` scale is the
only scale.

Candidate blocks: two-level selection with block size `bs = 8`, 2048 blocks,
and candidate source layer 20 (`deepseek41.cpp:561-619`).

- Level 1, layer 20 itself (`pick_blk`). Compute
  `bmax[k] = max_{j in block k} score[j]` with padding to a multiple of 8 using
  -inf (`ggml_pool_1d MAX`, 598-602). Add `cand_pin[k] = 1e30` to the block
  `(n_visible-1)/8` holding the newest compressed row, and 0 elsewhere (only
  when `n_visible > 0`; `deepseek41.cpp:331-345`). Then
  `cand_sel = top_k(bmax, min(2048, n_blk))` (605). Layer 20's own top-k comes
  from the unrestricted `score` (619).
- Level 2, index sources after layer 20 (24, 28, 32, 36; `use_blk`). Add
  `score[j] += 0` if `block(j) in cand_sel`, else -inf (606-617), then take
  the top-k. `cand_sel` is the one computed at layer 20 in the same forward
  pass, for the same token.
- Layers 2, 8 and 14 come before the candidate source and use no block
  restriction.

The final selection is `top_k = indices of the min(n_comp, 512) largest score`
(`deepseek41.cpp:619`). Set semantics are enough: order does not matter, and
tie-break is CPU `std::partial_sort` (`ops.cpp:8891`). When fewer than 512
rows are visible, top-k also returns masked indices, which the attention mask
then removes (section 4.6). The net effect is to attend all visible
compressed rows if `n_visible <= 512`, and otherwise the 512 best visible
ones.

Consumers reuse `topk_of[topk_src[il]]` (`deepseek41.cpp:534`). For example,
layers 21..23 use layer 20's top-k and layers 25..27 use layer 24's.

Implementation shortcut (math-equivalent): llama.cpp pads `n_comp` to a
multiple of 256 (at least 256) and masks the padding
(`kv-cache-dsv4.cpp:708`). A from-scratch engine can score only the
`n_visible` rows, and at level 2 only the rows inside the 2048 candidate
blocks.

### 4.6 Attention for compressed layers (2..39)

The attention is joint over window plus selected compressed rows, with one
softmax that includes the sinks (`deepseek41.cpp:652-746`; the dense
reference path is at 718-746, and the gather path at 664-717 computes the same
keys and mask in a different order):

```
keys   = [ window rows (512) | comp rows row[j][0:512] for j in top_k ]
mask   = [ window mask       | (0 if j < n_visible else -inf) for j in top_k ]
for each head hd in 0..63:
    l_k   = dot(q[hd], key_k) * (1/sqrt(512)) + mask_k
    m     = max(max_k l_k, sink[hd])
    Z     = sum_k exp(l_k - m) + exp(sink[hd] - m)
    o[hd] = sum_k exp(l_k - m)/Z * key_k            # V = K (512 dims)
```

The compressed part of the mask is `build_top_k_mask`
(`deepseek4.cpp:685-712`): -inf everywhere, 0 at the top-k indices, plus the
visibility mask (`kq_mask`).

Sinks: `attn_sinks[hd]` is a raw logit with no kq_scale. It joins the
max-subtraction and the denominator and carries no value (CPU softmax
`ops.cpp:5970-5981`: `max = MAX(max, sk[h])`, then
`sum += expf(sk[h] - max)`; the FA path is the same, `ops.cpp:9119-9122`). It
applies to every layer type.

The compressed rows use only `row[0:512]`, which is the already-roped `comp`.
The index key part is not attended. Rows are stored unrotated (no Hadamard).
`k_rot` applies only to quantized K caches and is mathematically transparent.
It can be skipped.

### 4.7 Output: inverse rope plus grouped output LoRA

`deepseek41.cpp:753-767` (trailing numbers are line numbers):

```
# 754-756
o[hd] = rope_inverse(o[hd], pos = p, offset 448, same per-layer params)
o_flat = concat_hd o[hd]                                          # 32768
# 759
group g = 0..7 : o_g = o_flat[g*4096 : (g+1)*4096]   (= heads 8g..8g+7)
# 760
a_g = W_a[g] · o_g
    W_a[g] = attn_output_a rows g*1024..g*1024+1023 (each a 4096 row)
# 762
a   = concat_g a_g                                    # 8192, index g*1024 + r
out = attn_output_b · a                               # 5120               764
```

---------------------------------------------------------------------------

## 5. MoE FFN (`deepseek41.cpp:884-922`)

The routed experts go through `build_moe_ffn` (`llama-graph.cpp:2208-2575`).
The input is `h = rms(hc_pre(X, attn_pre), ffn_norm)`. Bare trailing numbers
are `llama-graph.cpp` line numbers unless another file is named:

```
# [384], F32 weights, F32 activations+accum   888-892 (deepseek41.cpp)
logits = ffn_gate_inp · h
# softplus(x) = x > 20 ? x : log1p(exp(x))
#   llama-graph.cpp:2272, unary-ops.cpp:80-82
probs  = sqrt(softplus(logits))
# text: exp_probs_b; exp_probs_b_vl only when the ubatch is embedding
# input (image)  deepseek41.cpp:896-899, llama-graph.cpp:2283
sel    = probs + exp_probs_b
E      = argsort_desc(sel)[0:6]    # llama-graph.cpp:2326  (no expert groups)
w_i    = probs[E_i]                # UNBIASED probs   2340
# norm only because n_used = 6 > 1   deepseek41.cpp:909, llama-graph.cpp:2359
w_i    = w_i / (sum_i w_i + 1e-20)
w_i   *= 1.5                       # expert_weights_scale   2372
for each i:
    g = gate_exps[E_i] · h ; u = up_exps[E_i] · h          # 2304 each
    a = silu(min(g, L)) * clamp(u, -L, L)    # L = swiglu_clamp_exp[il] = 10
    y_i = down_exps[E_i] · a
moe = sum_i w_i * y_i                                       # 2553 then adds
shared: g = gate_shexp·h, u = up_shexp·h,
        # Ls = swiglu_clamp_shexp[il] = 10
        a = silu(min(g, Ls)) * clamp(u, -Ls, Ls)
        # build_ffn, llama-graph.cpp:2020-2045
        s = down_shexp·a
f = moe + s                                             # deepseek41.cpp:922
```

The swiglu clamp formula is `ggml_swiglu_clamp`, which is what the code uses
for DEEPSEEK41 in both the routed path (`llama-graph.cpp:2454`) and the shared
path (`llama-graph.cpp:2028`):

```
// raw gate, upper clamp only
ops.cpp:3685  const float gate = std::min(src0_p[k], limit);
ops.cpp:3686  const float up   = std::clamp(src1_p[k], -limit, limit);
ops.cpp:3687  dst_p[k]         = gate / (1.f + expf(-gate)) * up;
```

This is not the generic path, which clamps `silu(gate)` instead
(`llama-graph.cpp:2030-2036`). It clamps the pre-activation gate from above
(no lower bound) and clamps up symmetrically.

No hash-routed layers exist (`hash_layer_count = 0`). The shared expert is
added unweighted. The router bias `exp_probs_b` affects selection only.

---------------------------------------------------------------------------

## 6. Output

Trailing numbers are `deepseek41.cpp` line numbers.

```
# pre-mix of layer 39's FFN mixes (3.4)
out    = hc_pre(X, pre_prev)                                         # 935
out    = rms(out, output_norm)                                       # 938
# no softcap, no scaling
logits = output · out                 # [129280]                       942
```

---------------------------------------------------------------------------

## 7. KV cache contents per sequence (for a paged design)

Everything is K-only: V equals K. Each store is listed as layers; width per
entry; entries kept.

- Window K (roped, normed kv): all 40 layers; 512; the last 128 tokens.
- Token-id history (Engram): 1 per sequence; 1 int32; the last 3 tokens.
- CSA compressed rows: layers 2, 8, 14 (r=2); 640 = 512 comp + 128 ik; 1 per
  2 tokens, ALL history.
- HCA compressed rows: layer 20 (r=1); 640; 1 per token, ALL history.
- Compressor state (F32): layers 2, 8, 14; kv 512 + score 512; ring of 2.
- Compressor state (F32): layer 20; kv 512 + score 512; ring of 1 (optional).

Growth per token: compressed `3*640/2 + 640 = 1600` elements, which is 3.2 KB
at F16. Fixed per sequence: windows `40*128*512` = 2.6M elements (5.2 MB F16)
plus about 16 KB of F32 state. Window size is 128. llama.cpp's SWA cache is
sized for window + ubatch. For one-token decode a 128-slot ring per layer
works. The window must contain the current token before attention runs.

Compressed rows can never be evicted, because top-k attends anywhere in
history. Row `b` of the CSA caches corresponds to tokens `2b, 2b+1`, and row
`b` of the HCA cache to token `b`. The index key lives in the same row, so a
paged block of compressed rows serves both the indexer and the attention.
llama.cpp stores the window and the compressed rows in `type_k` (F16 by
default; `llama-kv-cache-dsv4.cpp:1339-1347, 1413-1423`). Compressor state is
always F32 (`kv-cache-dsv4.cpp:1033-1034`).

Shift and rollback are not supported in llama.cpp: `get_can_shift`, and
`llama_model_n_swa` returns 0 for DSV4. The `state_snapshot` and
`state_restore` planes exist for speculative rollback only.

---------------------------------------------------------------------------

## 8. MXFP8 (type 43) and the other quant types

Block layout (`ggml-common.h:221-227`): 33 bytes per 32 values.

```
struct block_mxfp8 { uint8_t e; /* E8M0 */ uint8_t qs[32]; /* E4M3 (fn) */ };
```

Dequant (`ggml-quants.c:618-633`, `ggml-impl.h:451-485, 567-581`):

```
d    = (e == 0xFF) ? NaN : (e == 0 ? 2^-127 : 2^(e-127))
e4m3(x): s = x&0x80, E = (x>>3)&0xF, m = x&7
         E==15 && m==7 -> NaN ; E==0 -> m * 2^-9 ; else (8+m) * 2^(E-10) ;
         negate if s
y[j] = e4m3(qs[j]) * d
```

The type is "gather-only" (`ggml.h:433`): it supports get_rows, not matmul.
The converter writes it only for the Engram tables, bit-exact from the
checkpoint's e4m3 rows plus one e8m0 scale per 32. The default is
`DSV41_ENGRAM_TYPE=mxfp8`, or bf16 (`conversion/deepseek41.py:232-262`). The
converter also writes the routed experts bit-exact as MXFP4 and turns FP8
linears into BF16 (`conversion/deepseek41.py:120-123, 220-230`).

The UD-IQ1_S GGUF contains no MXFP8 or MXFP4. Unsloth requantized the Engram
table to IQ4_NL and the experts to IQ1_S/IQ2_XXS. The types needed for this
file are F32, BF16 (compressor and indexer weights), Q5_K, Q6_K, Q8_0, IQ1_S,
IQ2_XXS and IQ4_NL. For the Engram table only a row dequant
(get_rows) is needed.

---------------------------------------------------------------------------

## 9. Not needed for plain text decode

- DSpark / MTP. `build_arch_graph` throws for `LLM_GRAPH_TYPE_DECODER_MTP`
  (`deepseek41.cpp:212-218`). The draft is a separate DFLASH GGUF, and the
  target GGUF holds only `dspark.*` metadata and no DSpark tensors. The
  `embeddings_layer_inp` / `build_hc_mean` feature taps (`849-854`) feed
  DSpark only.
- Vision. The tower is in a separate mmproj (`deepseek41v`).
  `exp_probs_b_vl` and `engram_keep` apply only to embedding-input ubatches
  (`896-899`, `811-817`). For text, always use `exp_probs_b` and gate keep 1.
- Hash-routed MoE (`ffn_gate_tid2eid`, V4 only), V4's `hc_head`
  (`output_hc_*`), APE, overlap compressor and LID cache: all absent in V4.1.
- Performance-only paths: `LLAMA_DSV41_SPARSE` (fused lightning indexer,
  gather path, `n_kv_max`), the host gather, KV mirrors across devices
  (`llama_dsv41_kv_read_layer`, `kv-cache-dsv4.cpp:18-39`), Hadamard `k_rot`,
  query chunking, and the fused HC ops (`ggml_dsv4_hc_pre/post/comb`). Each is
  equivalent to the dense path above.

---------------------------------------------------------------------------

## 10. Mapping in locus

The paged cache uses `kv_dim = head_dim + idx_dim` = 512 + 128 = 640, and
`n_layers + (number of kv-source layers)` cache layers (40 + 4 = 44 here).

- Cache layers `0 .. n_layers-1` hold each layer's roped `kv` at its position
  in the K half. The window reads the last `window` (128) positions.
- One extra cache layer per kv-source layer holds that source's compressed
  rows in its K half: row `b` at position `b`, 512 latent values then the
  128-dim index key. Its V half holds the compressor kv state at the token's
  position.
- The compressor score state lives in the source layer's own V half.
- Layer 0's V half, slot 0, stores each position's token id for the Engram
  hash.

The cache is F32 (llama.cpp uses F16 by default), so small logit differences
versus the reference are expected.

---------------------------------------------------------------------------

## 11. Open questions and ambiguities

1. No q head-norm. V4's graph applies `rms_norm(q)` per head before rope
   (`deepseek4.cpp:926`). The V4.1 graph does not (`deepseek41.cpp:451-452`),
   and the window `kv` keeps its weighted norm. This presumably follows the
   V4.1 `model.py`, but it was not checked against the upstream model.py; the
   `inference/model.py` of the HF repo was not fetched.
2. The lagged mHC order (3.4) is unusual. It is deliberate according to the
   comments and the DSpark `single_pass` flag. A first-token logits check
   against llama.cpp is the cheapest way to confirm a re-implementation.
3. Compressed-row and window precision. llama.cpp stores both in F16 by
   default, including the 128-dim index key, and runs the window/compressed
   attention through F16 K. An F32 cache in locus is more accurate, but small
   logit differences against llama.cpp are expected.
4. Tie-breaking in the indexer top-k, the candidate blocks and the MoE argsort
   is implementation-defined. It only matters for exact parity on ties.
5. Engram history for tokens shared by several sequences is rejected in
   llama.cpp (`dsv41_shared_history_ok`). With one history per sequence this
   is moot.
6. `n_visible` uses `(p+1)/r` with absolute positions. Positions must be
   contiguous from 0 per sequence (llama.cpp throws otherwise,
   `kv-cache-dsv4.cpp:655-656`).
7. The candidate pin uses `+1e30` where the reference uses `+inf`. That only
   changes how fully masked blocks behave. For 2048 blocks of 8 rows, the
   restriction matters only past 16384 compressed rows, which is 16384 tokens
   for the ratio-1 layers.
