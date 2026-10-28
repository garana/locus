# PixArt-α 512 (pixart-alpha): text-to-image reference

Draft spec for i#307. It describes the math locus must run to make a
512 x 512 image with PixArt-α 512 (`PixArt-alpha/PixArt-XL-2-512x512`,
Hugging Face commit `50f7021`). Four parts work together: a T5 text encoder
turns the prompt into vectors, a diffusion transformer (the "DiT", or
denoiser) predicts the noise in a latent image, a sampler uses that
prediction to step the latent from pure noise to a clean latent, and a VAE
decoder turns the clean latent into pixels.

Sources:

- diffusers 0.31.0. A citation `file:Name` names a class or a function in
  that file. File names drop the `src/diffusers/` prefix and the folder:
  `pixart_transformer_2d.py`, `attention.py`, `attention_processor.py`,
  `embeddings.py`, `normalization.py`, `resnet.py`, `unet_2d_blocks.py`,
  `upsampling.py`, `autoencoder_kl.py`, `vae.py` (all under models/),
  `scheduling_dpmsolver_multistep.py` (schedulers/) and
  `pipeline_pixart_alpha.py` (pipelines/pixart_alpha/).
- The tensor list of the three checkpoints at `50f7021` (name, dtype,
  shape) and the transformer and VAE `config.json` files.
- The T5 config values (section 3.1). The T5 forward math is NOT in the
  files above. Section 3 states it from transformers' `modeling_t5.py`
  (`T5EncoderModel`) as known to the author; it must be checked against
  the transformers source in the reference venv before code depends on it.
- The golden script `scripts/make-pixart-goldens.py` (section 8).

Anything else is marked "(assumption)".

Notation. Shapes are PyTorch shapes, outermost dimension first, row-major.
A Linear weight is `[out, in]` and maps an `in` vector to an `out` vector:
`W·x + b` means `y[o] = b[o] + sum_i W[o][i] x[i]`. In the code blocks, a
`+ b` right after a Linear is that Linear's own bias. A conv weight is
`[out, in, kh, kw]`. The GGUF `ne` order is the reverse (section 7.1).

- `ln(x)` is a LayerNorm over the last axis with no weight and no bias:
  `(x - mean(x)) / sqrt(var(x) + eps)`, where `var` is the biased variance
  (divide by the count). PyTorch `nn.LayerNorm` semantics (assumption).
- `gn(x; w, b)` is a GroupNorm with 32 groups: the channels split into 32
  equal groups, each group is normalized like `ln` over (its channels x
  all pixels), then each channel `c` gets `* w[c] + b[c]`.
- `rms(x; w) = w * x / sqrt(mean(x^2) + eps)`: no mean, no bias (T5).
- `silu(x) = x * sigmoid(x)`.
- `gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`. This
  is PyTorch's `gelu(approximate="tanh")` and transformers' `gelu_new`
  (assumption: the formulas come from PyTorch and transformers, not from
  the inputs). No model here uses the erf GELU.
- `D = 1152` (DiT width), `H = 16` (DiT heads), `Dh = 72` (DiT head dim),
  `L = 28` (DiT blocks), `F = 4608` (DiT FFN width), `T = 4096` (image
  tokens), `S = 120` (text tokens), `C = 4` (latent channels).

---------------------------------------------------------------------------

## 1. Overview

### 1.1 The three models

Parameter counts are summed from the tensor list. Sizes are at the
checkpoint dtype.

| part         | subfolder    | class                    | params  | dtype | size    |
|--------------|--------------|--------------------------|---------|-------|---------|
| text encoder | text_encoder | T5EncoderModel           | 4.762 B | F16   | 9.52 GB |
| denoiser     | transformer  | PixArtTransformer2DModel | 611.3 M | F32   | 2.45 GB |
| VAE decoder  | vae          | AutoencoderKL            | 49.5 M  | F32   | 0.20 GB |

- T5: 4,762,310,656 parameters. 131,596,288 of them are the token
  embedding `shared.weight`.
- DiT: 611,347,616 parameters. That count includes
  `caption_projection.y_embedding` (491,520), which diffusers does not use
  (section 4.5).
- VAE: 49,490,199 parameters in `post_quant_conv` and `decoder.*`. The
  checkpoint also holds the encoder and `quant_conv`; text-to-image never
  runs them.

### 1.2 What runs per image

For one 512 x 512 image with 8 steps and guidance scale 4.5. FLOP counts
are estimates from the shapes, with one multiply-add counted as 2 FLOP.

1. Clean and tokenize the prompt to 120 token ids plus a 0/1 mask. The
   empty negative prompt `""` goes through the same steps (section 2).
2. Run T5 twice, once for the prompt and once for `""`. Each pass gives
   `[120, 4096]`. A full 120-token pass is about 1.1 TFLOP. The negative
   output never changes, so locus can compute it once per model load.
3. Draw the start latent: `x ~ N(0, 1)` of shape `[1, 4, 64, 64]`
   (section 5.6).
4. Run 8 steps. Each step runs the DiT once on a batch of 2 (row 0 is the
   negative prompt, row 1 is the prompt). Each row has 4096 image tokens
   of width 1152 and cross-attends to its 120 text tokens. One row costs
   about 6.5 TFLOP; the 16 rows of the 8 steps cost about 104 TFLOP. The
   step then combines the two rows (CFG, classifier-free guidance), keeps
   the noise half of the 8 output channels, and updates `x` with the
   sampler.
5. Decode `x / 0.18215` with the VAE to `y`, of shape `[1, 3, 512, 512]`
   and in about [-1, 1]. About 2.5 TFLOP.
6. Map `y` to pixels: `clamp(y / 2 + 0.5, 0, 1) * 255`.

Per DiT block and batch row, the FLOP split is: self-attention
projections 43.5 G, self-attention scores and weighted sum 77.3 G,
cross-attention 24.6 G, FFN 87.0 G; 232 G in total.

The self-attention has `16 x 4096 x 4096` scores per batch row and block
(1 GiB in F32 if stored at once). locus needs an attention that streams
over keys instead of keeping the full score matrix.

### 1.3 Layouts

- Latents and every VAE tensor are NCHW: `[batch, channel, row, col]`,
  with `col` fastest.
- DiT hidden states are token-major: `[batch, token, 1152]`. Token
  `n = 32 i + j`, where `i` is the patch row (0..31) and `j` the patch
  column (0..31).
- T5 hidden states are token-major: `[token, 4096]`.

---------------------------------------------------------------------------

## 2. Prompt to token ids

`pipeline_pixart_alpha.py:PixArtAlphaPipeline.encode_prompt` with
`max_sequence_length = 120`.

1. `_text_preprocessing(text, clean_caption)`. `__call__` defaults
   `clean_caption = True`. If BeautifulSoup (`bs4`) or `ftfy` is not
   installed, it logs a warning and sets `clean_caption = False`. With
   cleaning on, it runs `_clean_caption` twice; with cleaning off, it
   runs `text.lower().strip()`.
2. `_clean_caption` does: URL unquoting, strip and lower case,
   `<person>` to `person`, URL removal, HTML to text (BeautifulSoup),
   `@name` removal, removal of seven CJK ranges, every dash kind to `-`,
   quote kinds to `"` and `'`, removal of `&quot;` and `&amp`, IP
   addresses, some id and file-name patterns, the `bad_punct_regex` set
   (`#®•©™&@·º½¾¿¡§~)(][}{|\/*`) to a space, `-` and `_` to spaces when
   there are more than 3, `ftfy.fix_text`, `html.unescape` twice, more
   id patterns, "free shipping" and "download" phrases, whitespace
   collapse, and leading or trailing quote and punctuation removal.
3. Tokenize with `padding="max_length"`, `max_length=120`,
   `truncation=True`, `add_special_tokens=True`. The result is
   `input_ids [1, 120]` and `attention_mask [1, 120]`.
4. The negative prompt (`""`) goes through the same preprocessing and the
   same tokenizer call, with `max_length` equal to the prompt's (120).

Tokenizer facts that are not in the inputs (assumption): `T5Tokenizer` is
a SentencePiece unigram model (`spiece.model`). It appends `</s>` (id 1),
pads with `<pad>` (id 0), and the mask is 1 on the real tokens and the
`</s>`, 0 on the pads. So `""` gives ids `[1, 0, 0, ...]` and mask
`[1, 0, 0, ...]`: one valid token.

For the golden prompt `A small red apple on a wooden table, soft morning
light.` both cleaning paths give `a small red apple on a wooden table,
soft morning light.` (checked by hand against each regex in
`_clean_caption`; the `t5_ids` golden settles it).

---------------------------------------------------------------------------

## 3. T5 v1.1 XXL encoder

Section 3.1 lists config values given for this task. Sections 3.2 to 3.6
describe transformers' `T5EncoderModel`; they are not backed by the input
files and must be checked against `modeling_t5.py`.

### 3.1 Config

| key                             | value                 |
|---------------------------------|-----------------------|
| d_model                         | 4096                  |
| d_ff                            | 10240                 |
| d_kv (head dim)                 | 64                    |
| num_heads                       | 64                    |
| num_layers                      | 24                    |
| relative_attention_num_buckets  | 32                    |
| relative_attention_max_distance | 128                   |
| feed_forward_proj               | gated-gelu (gelu_new) |
| layer_norm_epsilon              | 1e-6                  |
| vocab_size                      | 32128                 |

### 3.2 Tensors

All F16, no biases anywhere. `N = 0..23`; `P = encoder.block.N.layer`.

| tensor                                                               | shape         | meaning                               |
|----------------------------------------------------------------------|---------------|---------------------------------------|
| shared.weight                                                        | [32128, 4096] | token embedding, row lookup           |
| P.0.layer_norm.weight                                                | [4096]        | rms weight before attention           |
| P.0.SelfAttention.q.weight                                           | [4096, 4096]  | query; head h is rows 64h..64h+63     |
| P.0.SelfAttention.k.weight                                           | [4096, 4096]  | key                                   |
| P.0.SelfAttention.v.weight                                           | [4096, 4096]  | value                                 |
| P.0.SelfAttention.o.weight                                           | [4096, 4096]  | output projection                     |
| encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight | [32, 64]      | bias per (bucket, head); layer 0 only |
| P.1.layer_norm.weight                                                | [4096]        | rms weight before the FFN             |
| P.1.DenseReluDense.wi_0.weight                                       | [10240, 4096] | gate input (the GELU side)            |
| P.1.DenseReluDense.wi_1.weight                                       | [10240, 4096] | linear input                          |
| P.1.DenseReluDense.wo.weight                                         | [4096, 10240] | FFN output                            |
| encoder.final_layer_norm.weight                                      | [4096]        | final rms weight                      |

### 3.3 Forward

`ids` and `mask` are the 120 token ids and the 0/1 mask from section 2.

```
x = shared[ids]                                  # [120, 4096], no scale
# position bias, from layer 0's table, shared by all 24 layers
B[h][i][j] = rel_bias[bucket(j - i)][h] + mbias[j]
mbias[j] = 0 if mask[j] == 1 else finfo(dtype).min
for l in 0..23:
    y = rms(x; P.0.layer_norm)                   # eps 1e-6
    q = Wq·y ; k = Wk·y ; v = Wv·y               # [120, 4096] each
    for head h in 0..63:                         # dims 64h..64h+63
        s[i][j] = dot(q_h[i], k_h[j]) + B[h][i][j]   # no 1/sqrt(64)
        p[i][:] = softmax over j of s[i][:]
        o_h[i]  = sum_j p[i][j] v_h[j]
    x = x + Wo·concat_h(o_h)
    y = rms(x; P.1.layer_norm)
    x = x + wo·( gelu_tanh(wi_0·y) * (wi_1·y) )
out = rms(x; encoder.final_layer_norm)           # [120, 4096]
```

- T5 has no attention scale: the query-key dot is not divided by
  `sqrt(64)`.
- The softmax runs in F32 and casts back to the activation dtype.
- `finfo(dtype).min` is about -3.39e38 for bf16. Any value that makes the
  weight exactly 0 works; locus can skip masked keys.
- Dropout layers are no-ops at inference.

### 3.4 Relative position buckets

The bias is bidirectional. `d = j - i` is the key position minus the query
position. Half of the 32 buckets are for `d > 0`:

```
bucket(d) = (16 if d > 0 else 0) + b(|d|)
b(n) = n                                            if n < 8
b(n) = min(15, 8 + floor(ln(n/8) / ln(128/8) * 8))  otherwise
```

With 120 tokens, `n = abs(d)` is at most 119, so `b` takes these values:

| n    | 0..7 | 8..11 | 12..15 | 16..22 | 23..31 | 32..45 | 46..63 | 64..90 | 91..119 |
|------|------|-------|--------|--------|--------|--------|--------|--------|---------|
| b(n) | n    | 8     | 9      | 10     | 11     | 12     | 13     | 14     | 15      |

The edges at 16, 32 and 64 are where `ln(n/8)/ln(16)*8` is an exact
integer. A float32 recomputation gave the same table, but an `ln` that is
one ulp low would move one of these distances down a bucket. locus should
use this integer table, not the float formula.

### 3.5 Masked rows

Every query sees only the valid keys, and the bias depends only on
`j - i`. So the output rows of the valid tokens do not depend on the
padded rows at all. locus can run T5 on the `n` valid tokens only (`n = 1`
for `""`) and leave the padded rows of the `[120, 4096]` output as zeros.
The DiT masks those rows (section 4.6), so its output does not change.
The padded rows of the reference `t5_out` are real numbers, so a test must
compare only the rows where `mask == 1`.

### 3.6 Precision of the reference

The golden script loads the fp16 weights with `torch_dtype=bfloat16`. The
cast from fp16 (10 mantissa bits) to bf16 (7 bits) loses precision.
transformers 4.46.3 keeps `wo` in F32 for T5 (`_keep_in_fp32_modules =
["wo"]`, checked) and casts the FFN input to `wo`'s dtype, so the
residual stream turns F32 after the first FFN.
The rms norm computes in F32 and casts to the weight dtype (bf16). locus
runs F16 weights with F32 math, so it will differ from `t5_out` by about
bf16 rounding.

---------------------------------------------------------------------------

## 4. PixArt DiT (PixArtTransformer2DModel)

### 4.1 Config and derived values

From `transformer/config.json` and `PixArtTransformer2DModel.__init__`:

| name                      | value            | where it comes from                            |
|---------------------------|------------------|------------------------------------------------|
| inner_dim D               | 1152             | num_attention_heads 16 * attention_head_dim 72 |
| num_layers L              | 28               | config                                         |
| in_channels C             | 4                | config                                         |
| out_channels              | 8                | config; 4 noise + 4 learned variance           |
| patch_size                | 2                | config                                         |
| sample_size               | 64               | config; latent is 64 x 64                      |
| use_additional_conditions | false            | `__init__`: true only if sample_size == 128    |
| interpolation_scale       | 1                | `__init__`: max(sample_size // 64, 1)          |
| pos-embed base_size       | 32               | `PatchEmbed.__init__`: height // patch_size    |
| caption_channels          | 4096             | config                                         |
| cross_attention_dim       | 1152             | config                                         |
| norm_eps                  | 1e-6             | config; all LayerNorms, no affine              |
| activation_fn             | gelu-approximate | config; gelu_tanh                              |
| attention_bias            | true             | config; q, k, v and out all have biases        |

`norm_num_groups` (32), `num_embeds_ada_norm` (1000), `dropout` (0) and
`upcast_attention` (false) do not change the math on this path.

### 4.2 Tensors

All F32. `N = 0..27`; `B = transformer_blocks.N`; `ada =
adaln_single.emb.timestep_embedder`.

| tensor                             | shape                 | meaning                               |
|------------------------------------|-----------------------|---------------------------------------|
| pos_embed.proj.weight              | [1152, 4, 2, 2]       | patch embed conv, stride 2            |
| pos_embed.proj.bias                | [1152]                |                                       |
| ada.linear_1.weight                | [1152, 256]           | timestep MLP, first layer             |
| ada.linear_1.bias                  | [1152]                |                                       |
| ada.linear_2.weight                | [1152, 1152]          | timestep MLP, second layer            |
| ada.linear_2.bias                  | [1152]                |                                       |
| adaln_single.linear.weight         | [6912, 1152]          | shared modulation, 6 rows of 1152     |
| adaln_single.linear.bias           | [6912]                |                                       |
| caption_projection.linear_1.weight | [1152, 4096]          | T5 width to DiT width                 |
| caption_projection.linear_1.bias   | [1152]                |                                       |
| caption_projection.linear_2.weight | [1152, 1152]          |                                       |
| caption_projection.linear_2.bias   | [1152]                |                                       |
| caption_projection.y_embedding     | [120, 4096]           | not used by diffusers (4.5)           |
| B.scale_shift_table                | [6, 1152]             | per-block modulation offsets          |
| B.attn1.to_q.weight / .bias        | [1152, 1152] / [1152] | self-attention query                  |
| B.attn1.to_k.weight / .bias        | [1152, 1152] / [1152] | self-attention key                    |
| B.attn1.to_v.weight / .bias        | [1152, 1152] / [1152] | self-attention value                  |
| B.attn1.to_out.0.weight / .bias    | [1152, 1152] / [1152] | self-attention output                 |
| B.attn2.to_q.weight / .bias        | [1152, 1152] / [1152] | cross-attention query (image side)    |
| B.attn2.to_k.weight / .bias        | [1152, 1152] / [1152] | cross-attention key (text side)       |
| B.attn2.to_v.weight / .bias        | [1152, 1152] / [1152] | cross-attention value (text side)     |
| B.attn2.to_out.0.weight / .bias    | [1152, 1152] / [1152] | cross-attention output                |
| B.ff.net.0.proj.weight / .bias     | [4608, 1152] / [4608] | FFN up, then gelu_tanh                |
| B.ff.net.2.weight / .bias          | [1152, 4608] / [1152] | FFN down                              |
| scale_shift_table                  | [2, 1152]             | final-layer modulation offsets        |
| proj_out.weight                    | [32, 1152]            | final linear, 2*2*8 outputs per token |
| proj_out.bias                      | [32]                  |                                       |

The position table is not in the checkpoint: `PatchEmbed` registers it as
a non-persistent buffer. locus computes it (section 4.3).

### 4.3 Patch embed and positions

`embeddings.py:PatchEmbed.forward`. The input is one latent
`x [4, 64, 64]`. The conv has kernel 2, stride 2 and no padding, then
`flatten(2).transpose(1, 2)` makes it token-major:

```math
e_n[d] = b[d] + \sum_{c=0}^{3} \sum_{p=0}^{1} \sum_{q=0}^{1}
  W[d][c][p][q] \cdot x[c][2i+p][2j+q], \quad n = 32 i + j
```

Then `h_n = e_n + pe_n`. The table `pe` is
`get_2d_sincos_pos_embed(1152, 32, base_size=32, interpolation_scale=1)`.
The grid coordinate is `index / (32 / 32) / 1`, which is just the index.

Watch the names: `get_2d_sincos_pos_embed` builds the grid with
`np.meshgrid(grid_w, grid_h)`, so `grid[0]` holds the column `j`, and
`get_2d_sincos_pos_embed_from_grid` puts `grid[0]` in the first half
(the half it calls `emb_h`). So the first 576 dims encode the column and
the last 576 dims encode the row.
`get_1d_sincos_pos_embed_from_grid(576, pos)` gives `[sin(pos w_k) for
k in 0..287, cos(pos w_k) for k in 0..287]`, with:

```math
w_k = 10000^{-k/288}, \quad k = 0, \dots, 287
```

So for token `n = 32 i + j` and `k = 0..287`:

```
pe[n][k]       = sin(j * w_k)
pe[n][288 + k] = cos(j * w_k)
pe[n][576 + k] = sin(i * w_k)
pe[n][864 + k] = cos(i * w_k)
```

numpy computes it in float64 and `PatchEmbed` casts it to float32.

### 4.4 Timestep embedding and adaLN-single

`normalization.py:AdaLayerNormSingle` with
`embeddings.py:PixArtAlphaCombinedTimestepSizeEmbeddings`. adaLN-single
means one modulation vector per step, shared by all 28 blocks, computed
from the timestep alone (no size or aspect-ratio terms here, because
`use_additional_conditions` is false).

The timestep `t` is the integer scheduler timestep (for example 999). The
pipeline passes the same `t` for both batch rows; the embedding casts it
to float. `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`
calls `get_timestep_embedding`:

```math
f_k = \exp(-\ln(10000) \cdot k / 128), \quad k = 0, \dots, 127
```

```
tf[k]       = cos(t * f_k)        # flip_sin_to_cos puts cos first
tf[128 + k] = sin(t * f_k)
```

`downscale_freq_shift = 0` makes the divisor `128 - 0`. The exponent
and `exp` run in float32. Then `TimestepEmbedding` (act `silu`) and
`AdaLayerNormSingle.forward`:

```
c = ada.linear_2 · silu(ada.linear_1 · tf + b) + b        # [1152]
m = adaln_single.linear · silu(c) + b                     # [6912]
```

`c` is `embedded_timestep`; the final layer uses it (4.7). `m` is the
`timestep` that every block receives. Row `r` of `m` is
`m[1152 r .. 1152 r + 1151]` (`timestep.reshape(batch_size, 6, -1)`).

### 4.5 Caption projection

`embeddings.py:PixArtAlphaTextProjection` (act `gelu_tanh`), applied to
each of the 120 rows of the T5 output `y`, padded rows included:

```
u = caption_projection.linear_2 · gelu_tanh(linear_1 · y + b) + b
```

`u` is `[120, 1152]`. `PixArtAlphaTextProjection` has no `y_embedding`
attribute, so diffusers never uses that checkpoint tensor; CFG uses T5 of
`""` instead (section 5.5).

### 4.6 Transformer block

`attention.py:BasicTransformerBlock.forward` with `norm_type =
"ada_norm_single"`. The block adds its own table to the shared vector:
`M = B.scale_shift_table + m.reshape(6, 1152)`, then `chunk(6, dim=1)`:

| row | name      | use                                      |
|-----|-----------|------------------------------------------|
| 0   | shift_msa | shift after norm1, before self-attention |
| 1   | scale_msa | scale after norm1: `* (1 + scale_msa)`   |
| 2   | gate_msa  | multiplies the self-attention output     |
| 3   | shift_mlp | shift after norm2, before the FFN        |
| 4   | scale_mlp | scale after norm2: `* (1 + scale_mlp)`   |
| 5   | gate_mlp  | multiplies the FFN output                |

Each row is a 1152-vector that broadcasts over the 4096 tokens:

```
a = ln(h) * (1 + scale_msa) + shift_msa      # norm1, eps 1e-6
h = h + gate_msa * SelfAttn(a)               # attn1
h = h + CrossAttn(h, u, mask)                # attn2: no norm, no gate
f = ln(h) * (1 + scale_mlp) + shift_mlp      # norm2, eps 1e-6
h = h + gate_mlp * FFN(f)
```

- The cross-attention takes `h` as it is. The code says "For PixArt norm2
  isn't applied here" and uses `norm2` for the FFN instead. There is no
  `norm3` on this path.
- `SelfAttn(a)` (`attention_processor.py:AttnProcessor2_0`): `q = to_q·a +
  b`, `k = to_k·a + b`, `v = to_v·a + b`, each `[4096, 1152]`. Head `g`
  uses dims `72g .. 72g+71`. The scores are `dot(q_g[i], k_g[j]) /
  sqrt(72)`, softmax over the 4096 keys, no mask (the pipeline passes no
  `attention_mask`). The output is `to_out.0 · concat_g(o_g) + b`.
  `F.scaled_dot_product_attention` uses its default scale `1/sqrt(72)`;
  `Attention.scale = dim_head**-0.5` is the same value.
- `CrossAttn(h, u, mask)`: `q = to_q·h + b` (4096 rows), `k = to_k·u + b`
  and `v = to_v·u + b` (120 rows). The scores are `dot(q_g[i], k_g[j]) /
  sqrt(72) + bias[j]`, where `bias[j] = (1 - mask[j]) * -10000`
  (`PixArtTransformer2DModel.forward` turns the `[batch, 120]` mask into
  this bias; `Attention.prepare_attention_mask` repeats it per head). Each
  batch row uses its own mask. `exp(-10000)` is exactly 0 in F32, so
  skipping masked keys gives the same result.
- `FFN(f) = ff.net.2 · gelu_tanh(ff.net.0.proj · f + b) + b`, 1152 to
  4608 to 1152. `FeedForward` builds `GELU(dim, inner_dim,
  approximate="tanh")` for "gelu-approximate". The `GELU` class
  (`activations.py`) is a Linear named `proj` followed by
  `F.gelu(approximate="tanh")` (checked). `net.1` is a dropout.

### 4.7 Final layer and unpatchify

`PixArtTransformer2DModel.forward`, step 3:

```
shift = scale_shift_table[0] + c             # c = embedded_timestep (4.4)
scale = scale_shift_table[1] + c
o = ln(h) * (1 + scale) + shift              # norm_out, eps 1e-6, no affine
y = proj_out · o + b                         # [4096, 32]
```

The final layer uses `c` itself: not `silu(c)` and not `m`.

Unpatchify: `reshape(-1, 32, 32, 2, 2, 8)` then
`einsum("nhwpqc->nchpwq")`. For channel `k` in 0..7 and `p, q` in 0..1:

```
out[k][2i + p][2j + q] = y[32 i + j][(2p + q) * 8 + k]
```

The output is `[batch, 8, 64, 64]`. Channels 0..3 are the predicted noise
(epsilon). Channels 4..7 are the learned variance, which the sampler does
not use.

---------------------------------------------------------------------------

## 5. Sampler (DPM-Solver++ multistep, order 2, midpoint)

### 5.1 Scheduler config

From the checkpoint's `scheduler/scheduler_config.json`; keys it does
not set (`final_sigmas_type`) take the `DPMSolverMultistepScheduler`
defaults:

| key                 | value       |
|---------------------|-------------|
| num_train_timesteps | 1000        |
| beta_start          | 0.0001      |
| beta_end            | 0.02        |
| beta_schedule       | linear      |
| solver_order        | 2           |
| prediction_type     | epsilon     |
| algorithm_type      | dpmsolver++ |
| solver_type         | midpoint    |
| lower_order_final   | true        |
| euler_at_final      | false       |
| final_sigmas_type   | zero        |
| lambda_min_clipped  | -inf        |
| timestep_spacing    | linspace    |
| steps_offset        | 0           |
| thresholding        | false       |
| variance_type       | None        |
| use_karras_sigmas   | false       |

### 5.2 Noise schedule

From `__init__`, all in float32:

```
beta[k]  = linspace(0.0001, 0.02, 1000)[k]          # k = 0..999
abar[k]  = prod_{m=0..k} (1 - beta[m])              # torch.cumprod
sigma[k] = sqrt((1 - abar[k]) / abar[k])
```

`sigma` is the scheduler's "sigma": the noise-to-signal ratio, not the
noise level. `_sigma_to_alpha_sigma_t` turns one sigma `s` into the signal
and noise levels, and the update code takes `lambda` from them:

```math
\alpha(s) = \frac{1}{\sqrt{s^2 + 1}}, \quad
\sigma(s) = s \cdot \alpha(s), \quad
\lambda(s) = \ln \alpha(s) - \ln \sigma(s) = -\ln s
```

So `alpha^2 = abar` and `sigma^2 = 1 - abar` at a training timestep.

### 5.3 Timesteps for 8 steps

`set_timesteps(8)`: `lambda_min_clipped = -inf`, so `clipped_idx = 0` and
`last_timestep = 1000`. `np.linspace(0, 999, 9)` is `[0, 124.875,
249.75, 374.625, 499.5, 624.375, 749.25, 874.125, 999]`. numpy rounds
half to even, so 499.5 becomes 500. Reversed, with the last entry (0)
dropped:

    timesteps = [999, 874, 749, 624, 500, 375, 250, 125]

`np.interp` at integer points returns `sigma[t]` exactly, and
`final_sigmas_type = "zero"` appends 0, so there are 9 sigmas (float32).
The timesteps and the sigma column were checked against the scheduler
itself (diffusers 0.31.0, `set_timesteps(8)`); the other columns follow
from the sigma.

| i | t   | sigma    | alpha(s)   | sigma(s) | lambda    |
|---|-----|----------|------------|----------|-----------|
| 0 | 999 | 157.407  | 0.00635282 | 0.99998  | -5.05884  |
| 1 | 874 | 48.1668  | 0.0207567  | 0.999785 | -3.87467  |
| 2 | 749 | 17.247   | 0.0578839  | 0.998323 | -2.84764  |
| 3 | 624 | 7.18614  | 0.137829   | 0.990456 | -1.97215  |
| 4 | 500 | 3.44297  | 0.278921   | 0.960314 | -1.23633  |
| 5 | 375 | 1.80124  | 0.485388   | 0.874299 | -0.588473 |
| 6 | 250 | 0.958034 | 0.722096   | 0.691793 | 0.0428722 |
| 7 | 125 | 0.429943 | 0.918688   | 0.394984 | 0.844102  |
| 8 | -   | 0        | 1          | 0        | +inf      |

### 5.4 One step

`step(eps, t_i, x)` at step index `i` (0..7) does:

1. `convert_model_output`: turn the noise prediction into a prediction of
   the clean latent, at the current sigma `s_i`:

```math
x_0 = \frac{x - \sigma(s_i) \cdot \epsilon}{\alpha(s_i)}
```

2. Shift the list of the last two `x_0` (`model_outputs`): `m1` is the
   previous one, `m0` the new one.
3. Upcast `x` to float32 and pick the update:

| i    | update                 | why                                                                                         |
|------|------------------------|---------------------------------------------------------------------------------------------|
| 0    | first order            | `lower_order_nums` is 0 (no previous x0)                                                    |
| 1..6 | second order, midpoint | `solver_order` is 2                                                                         |
| 7    | first order            | last step: `final_sigmas_type` is "zero" (and `lower_order_final` with fewer than 15 steps) |

4. Write the result to `x` and add 1 to the step index.

First order (`dpm_solver_first_order_update`), from `s = s_i` to
`t = s_{i+1}`:

```math
h = \lambda(t) - \lambda(s), \quad
x' = \frac{\sigma(t)}{\sigma(s)} x - \alpha(t) \left( e^{-h} - 1 \right) x_0
```

Second order (`multistep_dpm_solver_second_order_update`), with
`s0 = s_i`, `s1 = s_{i-1}` and `t = s_{i+1}`:

```math
h = \lambda(t) - \lambda(s_0), \quad
h_0 = \lambda(s_0) - \lambda(s_1), \quad
r_0 = h_0 / h
```

```math
D_0 = m_0, \quad D_1 = \frac{m_0 - m_1}{r_0}
```

```math
x' = \frac{\sigma(t)}{\sigma(s_0)} x
  - \alpha(t) \left( e^{-h} - 1 \right) D_0
  - \frac{1}{2} \alpha(t) \left( e^{-h} - 1 \right) D_1
```

The last step goes to sigma 0: `alpha(t) = 1`, `sigma(t) = 0`,
`lambda(t) = +inf` (torch computes `log(0) = -inf`), `exp(-h) = 0`, so
`x' = x_0` exactly. locus should special-case it: the final latent is the
last `x_0`.

The coefficients are float32 scalars (the sigmas are a float32 tensor).

### 5.5 CFG and the learned-variance split

`PixArtAlphaPipeline.__call__`, per step:

```
X   = [x, x]                                  # torch.cat([latents] * 2)
E   = DiT(X, t, [u_neg, u_pos], [mask_neg, mask_pos])   # [2, 8, 64, 64]
e   = E[0] + 4.5 * (E[1] - E[0])              # all 8 channels
eps = e[0:4]                                  # out_channels // 2 == 4
x   = step(eps, t, x)
```

- Row 0 is the negative prompt: `torch.cat([negative_prompt_embeds,
  prompt_embeds])`, and the masks are joined in the same order.
- CFG is on because `guidance_scale > 1`.
- `scale_model_input` returns its input unchanged, so the DiT sees `x`.
- The split takes the first 4 channels (`noise_pred.chunk(2, dim=1)[0]`).

### 5.6 Start latent

`prepare_latents`: the shape is `[1, 4, 512 / 8, 512 / 8] = [1, 4, 64,
64]` (`vae_scale_factor = 2^(4 - 1) = 8`). The noise comes from
`randn_tensor` with a CPU `torch.Generator` seeded with 146, then is
multiplied by `init_noise_sigma = 1.0`. locus cannot reproduce torch's
normal sampler, so the tests take the start latent from the golden
`dit_in` (both of its rows equal the start latent).

`use_resolution_binning` (default true) maps 512 x 512 to the `"1.0"`
entry of `ASPECT_RATIO_512_BIN`, which is `[512, 512]`, so it changes
nothing here.

---------------------------------------------------------------------------

## 6. VAE decoder (AutoencoderKL)

### 6.1 Config

From `vae/config.json` and `vae.py:Decoder.__init__`:

- `block_out_channels = [128, 256, 512, 512]`. The decoder walks them in
  reverse: up blocks 0..3 output 512, 512, 256 and 128 channels.
- `layers_per_block = 2`, so each up block has `2 + 1 = 3` resnets.
- `norm_num_groups = 32` everywhere. Every GroupNorm uses eps 1e-6 (the
  resnets and the mid attention get `resnet_eps = 1e-6`; `conv_norm_out`
  sets eps 1e-6).
- `act_fn = silu`, `latent_channels = 4`, `scaling_factor = 0.18215`,
  `mid_block_add_attention = true` (the default).
- The resnets get no time embedding (`temb_channels = None`), so they have
  no `time_emb_proj`.
- `force_upcast` is in the config, but this pipeline never reads it.

### 6.2 Tensors

All F32. Only `post_quant_conv` and `decoder.*` are used. `R` stands for
`resnets.k`, with the norm and conv tensors of `ResnetBlock2D`:
`norm1.{weight,bias} [Cin]`, `conv1.weight [Cout, Cin, 3, 3]`,
`conv1.bias [Cout]`, `norm2.{weight,bias} [Cout]`,
`conv2.weight [Cout, Cout, 3, 3]`, `conv2.bias [Cout]`, and
`conv_shortcut.weight [Cout, Cin, 1, 1]` with `conv_shortcut.bias [Cout]`
only when `Cin != Cout`.

| tensor                                                   | shape                  |
|----------------------------------------------------------|------------------------|
| post_quant_conv.weight / .bias                           | [4, 4, 1, 1] / [4]     |
| decoder.conv_in.weight / .bias                           | [512, 4, 3, 3] / [512] |
| decoder.mid_block.resnets.{0,1} (R)                      | Cin = Cout = 512       |
| decoder.mid_block.attentions.0.group_norm.weight / .bias | [512] / [512]          |
| decoder.mid_block.attentions.0.to_{q,k,v}.weight / .bias | [512, 512] / [512]     |
| decoder.mid_block.attentions.0.to_out.0.weight / .bias   | [512, 512] / [512]     |
| decoder.conv_norm_out.weight / .bias                     | [128] / [128]          |
| decoder.conv_out.weight / .bias                          | [3, 128, 3, 3] / [3]   |

The tensor list prints the first block's shape for each folded family.
The real channels of `decoder.up_blocks.U` are:

| U | resnets.0 Cin to Cout | conv_shortcut on resnets.0 | resnets.1, .2 | upsamplers.0.conv | size out  |
|---|-----------------------|----------------------------|---------------|-------------------|-----------|
| 0 | 512 to 512            | none                       | 512           | [512, 512, 3, 3]  | 128 x 128 |
| 1 | 512 to 512            | none                       | 512           | [512, 512, 3, 3]  | 256 x 256 |
| 2 | 512 to 256            | [256, 512, 1, 1]           | 256           | [256, 256, 3, 3]  | 512 x 512 |
| 3 | 256 to 128            | [128, 256, 1, 1]           | 128           | none              | 512 x 512 |

### 6.3 Forward

`autoencoder_kl.py:AutoencoderKL._decode` (tiling and slicing are off by
default) and `vae.py:Decoder.forward`:

```
z = x / 0.18215                          # the pipeline divides first
z = post_quant_conv(z)                   # 1x1, 4 -> 4, 64 x 64
s = conv_in(z)                           # 3x3 pad 1, 4 -> 512
s = res(s); s = attn(s); s = res(s)      # mid block, 512 ch, 64 x 64
for U in 0..3:
    s = res(s) three times               # resnets.0 changes the channels
    if U < 3: s = up(s)                  # nearest x2, then 3x3 conv
s = conv_out(silu(gn(s; conv_norm_out))) # 128 -> 3, 512 x 512
```

All 3x3 convs have stride 1, zero padding 1 and a bias. The 1x1 convs
have no padding and a bias.

`res(x)` (`resnet.py:ResnetBlock2D.forward`; no time embedding,
`output_scale_factor = 1`, dropout is a no-op):

```
h = conv1(silu(gn(x; norm1)))
h = conv2(silu(gn(h; norm2)))
r = conv_shortcut(x) if Cin != Cout else x
return r + h
```

`attn(x)` (`unet_2d_blocks.py:UNetMidBlock2D` builds `Attention` with
`heads = 512 // 512 = 1`, `dim_head = 512`, `norm_num_groups = 32`,
`residual_connection = True`, `bias = True`; the processor is
`AttnProcessor2_0`):

```
t = gn(x; group_norm) as [4096, 512]     # token n = 64 r + k (row r, col k)
q = to_q·t + b ; k = to_k·t + b ; v = to_v·t + b
o = softmax(q kᵀ / sqrt(512)) v          # one head, no mask
o = to_out.0·o + b
return x + o, back to [512, 64, 64]      # rescale_output_factor = 1
```

The residual adds the input from before the GroupNorm.

`up(x)` (`upsampling.py:Upsample2D` with `use_conv=True`):
`y[c][r][k] = x[c][r / 2][k / 2]` (integer division; `F.interpolate`
with `scale_factor=2.0, mode="nearest"`), then `upsamplers.0.conv`, a
3x3 conv with padding 1.

The largest activation is the 256-channel 512 x 512 map after up block 2
(256 MiB in F32).

### 6.4 To pixels

The golden script computes `clamp(y / 2 + 0.5, 0, 1) * 255` and permutes
it to `[512, 512, 3]` (HWC). It writes the PNG with
`image_processor.postprocess`, which is not in the inputs (assumption: it
rounds to uint8).

---------------------------------------------------------------------------

## 7. Proposed GGUF layout

### 7.1 Conventions

- One GGUF per model, so each one loads and streams on its own. The T5
  GGUF can then serve other models that use the same encoder (SD3, FLUX).
- Tensor names are the checkpoint names, unchanged, as the `vq16` GGUF
  does.
- The GGUF `ne` order is the reverse of the PyTorch shape: a Linear
  `[out, in]` has `ne = [in, out]`, a conv `[out, in, kh, kw]` has
  `ne = [kw, kh, in, out]`. The data bytes keep the checkpoint's
  row-major order.
- Types: the first version keeps the checkpoint dtype (F16 for T5, F32
  for the DiT and the VAE). Quantized types come later.
- Each loader checks `general.architecture`, every key it reads, and every
  tensor shape against the keys, and rejects a layout it does not run.

### 7.2 T5 encoder

`general.architecture = "t5encoder"` (the name llama.cpp uses for T5
encoder GGUFs, assumption).

| key                                         | type   | value             |
|---------------------------------------------|--------|-------------------|
| t5encoder.block_count                       | u32    | 24                |
| t5encoder.embedding_length                  | u32    | 4096              |
| t5encoder.feed_forward_length               | u32    | 10240             |
| t5encoder.attention.head_count              | u32    | 64                |
| t5encoder.attention.key_length              | u32    | 64                |
| t5encoder.attention.layer_norm_rms_epsilon  | f32    | 1e-6              |
| t5encoder.attention.relative_buckets_count  | u32    | 32                |
| t5encoder.attention.relative_max_distance   | u32    | 128               |
| t5encoder.feed_forward_activation           | str    | gated_gelu_tanh   |
| tokenizer.ggml.model                        | str    | t5 (unigram)      |
| tokenizer.ggml.tokens / scores / token_type | arrays | from spiece.model |
| tokenizer.ggml.eos_token_id                 | u32    | 1                 |
| tokenizer.ggml.padding_token_id             | u32    | 0                 |
| tokenizer.ggml.unknown_token_id             | u32    | 2                 |
| tokenizer.ggml.add_eos_token                | bool   | true              |

The tokenizer values are assumptions (section 2). Tensors:
`shared.weight`, `encoder.block.N.layer.0.layer_norm.weight`,
`encoder.block.N.layer.0.SelfAttention.{q,k,v,o}.weight`,
`encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`,
`encoder.block.N.layer.1.layer_norm.weight`,
`encoder.block.N.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight`,
`encoder.final_layer_norm.weight`.

### 7.3 PixArt DiT

`general.architecture = "pixart"`.

| key                                  | type | value       |
|--------------------------------------|------|-------------|
| pixart.block_count                   | u32  | 28          |
| pixart.embedding_length              | u32  | 1152        |
| pixart.feed_forward_length           | u32  | 4608        |
| pixart.attention.head_count          | u32  | 16          |
| pixart.attention.key_length          | u32  | 72          |
| pixart.norm_epsilon                  | f32  | 1e-6        |
| pixart.in_channels                   | u32  | 4           |
| pixart.out_channels                  | u32  | 8           |
| pixart.patch_size                    | u32  | 2           |
| pixart.sample_size                   | u32  | 64          |
| pixart.caption_channels              | u32  | 4096        |
| pixart.caption_length                | u32  | 120         |
| pixart.pos_embed.base_size           | u32  | 32          |
| pixart.pos_embed.interpolation_scale | f32  | 1.0         |
| pixart.timestep.channels             | u32  | 256         |
| pixart.timestep.flip_sin_to_cos      | bool | true        |
| pixart.timestep.freq_shift           | f32  | 0           |
| pixart.guidance_scale                | f32  | 4.5         |
| pixart.sampler.num_train_timesteps   | u32  | 1000        |
| pixart.sampler.beta_start            | f32  | 0.0001      |
| pixart.sampler.beta_end              | f32  | 0.02        |
| pixart.sampler.beta_schedule         | str  | linear      |
| pixart.sampler.algorithm             | str  | dpmsolver++ |
| pixart.sampler.solver_order          | u32  | 2           |
| pixart.sampler.solver_type           | str  | midpoint    |
| pixart.sampler.timestep_spacing      | str  | linspace    |
| pixart.sampler.lower_order_final     | bool | true        |
| pixart.sampler.final_sigmas_type     | str  | zero        |

The loader rejects `sample_size == 128` (the 1024 model, which needs the
size embedders) and a missing `scale_shift_table`. Tensors: every name of
section 4.2 except `caption_projection.y_embedding`, which diffusers does
not use.

### 7.4 VAE decoder

`general.architecture = "autoencoder_kl"`.

| key                                | type   | value                |
|------------------------------------|--------|----------------------|
| autoencoder_kl.latent_channels     | u32    | 4                    |
| autoencoder_kl.out_channels        | u32    | 3                    |
| autoencoder_kl.block_out_channels  | u32[4] | [128, 256, 512, 512] |
| autoencoder_kl.layers_per_block    | u32    | 2                    |
| autoencoder_kl.norm_groups         | u32    | 32                   |
| autoencoder_kl.norm_epsilon        | f32    | 1e-6                 |
| autoencoder_kl.scaling_factor      | f32    | 0.18215              |
| autoencoder_kl.mid_block_attention | bool   | true                 |

Tensors: `post_quant_conv.*` and `decoder.*` (section 6.2). The encoder
and `quant_conv` are left out.

---------------------------------------------------------------------------

## 8. Goldens

`scripts/make-pixart-goldens.py` runs the diffusers pipeline on the CPU:
T5 in bf16, the DiT and the VAE in F32. Prompt `A small red apple on a
wooden table, soft morning light.`, seed 146, guidance 4.5, 8 steps
(the default of its STEPS argument), 512 x 512. It writes raw
little-endian F32 (I64 for ids) plus `index.json`.

The script has run once with these settings (8 steps took about 7
minutes on a 4-core x86 CPU) and its image shows the prompt. The goldens
are not in the repository yet: each implementation PR (T5, DiT,
sampler, VAE) adds the ones its test reads, under `tests/golden/pixart/`.

Each test feeds the golden inputs of its stage, so an error in one stage
does not leak into the next. Only `latent_final` runs several stages (16
DiT rows and the sampler) end to end.

| golden       | shape          | type | gates                                      | proposed locus test                                   |
|--------------|----------------|------|--------------------------------------------|-------------------------------------------------------|
| t5_ids       | [120]          | I64  | cleaning, tokenizer, EOS and padding       | t5 tokenizer reproduces the PixArt prompt ids         |
| t5_mask      | [120]          | I64  | the mask from the tokenizer                | (same test)                                           |
| t5_out       | [120, 4096]    | F32  | T5 forward on t5_ids (rows with mask 1)    | t5 encoder matches the reference on the PixArt prompt |
| t5_neg_out   | [120, 4096]    | F32  | T5 on `""`: one valid token, the mask      | t5 encoder matches the reference on the empty prompt  |
| dit_in       | [2, 4, 64, 64] | F32  | DiT input of step 0; also the start latent | pixart dit matches the reference on the first step    |
| dit_t        | [2]            | F32  | the timestep of step 0 (999 both rows)     | (same test)                                           |
| dit_out      | [2, 8, 64, 64] | F32  | one DiT call, all 8 channels               | (same test)                                           |
| latent_final | [1, 4, 64, 64] | F32  | 8 steps: DiT, CFG, split, sampler          | pixart sampler reproduces the reference latent        |
| image        | [512, 512, 3]  | F32  | VAE decode of latent_final, 0..255         | pixart vae decode matches the reference image         |
| image.png    | 512 x 512 RGB  | PNG  | a visual check only                        | none                                                  |

- The DiT test feeds `t5_neg_out` as row 0 and `t5_out` as row 1, with
  their masks.
- The sampler test starts from `dit_in` row 0 and uses the golden T5
  outputs, not locus's.
- The script does not save the ids or the mask of `""`; the tests assume
  `[1, 0, ...]` for both (open question 6).
- The test names have no commas (a comma in a Catch2 name makes it
  unrunnable by name). The tests need the GGUFs and SKIP without them.
- The goldens total about 7.5 MB (`t5_out` and `t5_neg_out` are 1.97 MB
  each, `image` is 3.1 MB).

Unit tests that need no golden: the 8-step timestep list (5.3), the bucket
table (3.4), the position table at a few tokens (4.3), the timestep
features `tf` (4.4), and the unpatchify index map (4.7).

---------------------------------------------------------------------------

## 9. Open questions

1. Settled: section 5 now follows the checkpoint's own
   `scheduler_config.json`, and the 8-step timesteps and sigmas were
   checked against the scheduler.
2. Settled: with `text_encoder` set, `encode_prompt` casts given
   embeddings to its dtype (bf16), which the F32 DiT refuses (the first
   reference run failed there). The script now sets
   `pipe.text_encoder = None` after encoding.
3. Prompt cleaning: the reference venv has `bs4` and `ftfy`, so the
   goldens use diffusers' full `_clean_caption`. locus has neither
   library. Which part should locus implement? The golden prompt gives
   the same string either way.
4. The T5 tokenizer: locus's SPM tokenizer follows llama.cpp's score-based
   merges; T5's `spiece.model` is a unigram model (assumption), which
   needs a Viterbi search. Also unverified: the `legacy` flag of
   `T5Tokenizer`, the 100 `<extra_id_N>` tokens, and the gap between the
   32128 embedding rows and the tokenizer's vocabulary size.
5. Tolerances. `t5_out` carries bf16 rounding (fp16 weights cast to bf16;
   `wo` possibly in F32, section 3.6). The DiT and VAE references are F32
   but get bf16-valued text states. The thresholds for each golden should
   be set after the first run, from the measured error.
6. The script saves neither the token ids nor the mask of `""`. Saving
   `t5_neg_ids` and `t5_neg_mask` would pin the assumption `[1, 0, ...]`.
7. `caption_projection.y_embedding` is in the checkpoint but diffusers
   never reads it. The original PixArt code may use it as the
   unconditional caption (assumption). locus follows diffusers (T5 of
   `""`) and leaves it out of the GGUF.
8. Noise. locus cannot reproduce `torch.randn`, so a given seed will not
   give diffusers's image. locus needs its own seeded normal generator;
   the tests avoid the issue by starting from `dit_in`.
9. Sizes other than 512 x 512: resolution binning picks a size from
   `ASPECT_RATIO_512_BIN`, `PatchEmbed.forward` then recomputes the
   position table for the new grid (still `base_size = 32`), and
   `resize_and_crop_tensor` (not in the inputs) fits the image back.
   Is this in scope for the first version?
10. The default step count: `__call__` defaults to 20 steps and the golden
    uses 8. Which default should locus expose?
11. GGUF names: unchanged checkpoint names (this proposal) or llama.cpp's
    `t5encoder` names (`token_embd`, `enc.blk.N.attn_q`, ...; assumption),
    which would let locus load T5 GGUFs that already exist. The arch
    names `pixart` and `autoencoder_kl` are proposals.
