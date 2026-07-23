# Test-model architectures and information flow

Information-flow diagrams for every model under `tests/models/`.
The six model files map onto three architecture families the
runtime implements (`src/locus/model/arch.cpp`): `llama`,
`deepseek2`, and `glm-dsa`. Diagrams are per family; the
inventory below maps each file to its family and the spec that
drives its forward pass.

## 1. Inventory

| Model file            | Arch      | Quant  | Layers    | Attn        | FFN            |
|-----------------------|-----------|--------|-----------|-------------|----------------|
| stories260K           | llama     | F32    | 5         | MHA 8       | dense          |
| llama-3.2-1b-q8_0     | llama     | Q8_0   | 16        | GQA 32/8    | dense          |
| llama-3.2-1b-q4_k_m   | llama     | Q4_K_M | 16        | GQA 32/8    | dense          |
| deepseek-v2-lite      | deepseek2 | Q4_K_M | 27        | MLA 16      | 1d, MoE 64/6+2s |
| moonlight-16b-a3b     | deepseek2 | Q4_K_M | 27        | MLA 16/1    | 1d, MoE 64/6+2s |
| glm-5.2-iq1s          | glm-dsa   | IQ1_S  | 79 (78+1) | MLA+DSA 64/1 | 3d, MoE 256/8+1s |

Attn column: MHA/GQA/MLA with `heads` or `heads/kv_heads`. FFN
column: `Nd` = N leading dense layers, `MoE E/U+Ss` = E experts,
U routed per token, S always-on shared experts.

Key differences that change the diagrams:

- llama uses materialized-KV attention (MHA or GQA); deepseek2
  and glm-dsa use MLA latent attention (one compressed row per
  token in the cache).
- deepseek-v2-lite gates experts with softmax; moonlight and
  GLM-5.2 gate with sigmoid + group-limited routing.
- Only glm-dsa runs the DSA lightning indexer; the trailing MTP
  block (layer 78) is loaded but excluded from the causal stack,
  matching llama.cpp.

## 2. Shared forward skeleton

Every model is the same outer loop; the family only changes what
happens inside the Attention and FFN blocks.

```mermaid
flowchart TD
    tok["token id"] --> emb["embedding lookup (dequant row)"]
    emb --> x["hidden state x"]
    x --> L{"for each layer l"}
    L --> an["rmsnorm attn_norm"]
    an --> attn["ATTENTION (family-specific)"]
    attn --> wo["output proj wo"]
    wo --> r1["residual add: x += attn_out"]
    r1 --> fn["rmsnorm ffn_norm"]
    fn --> ffn["FFN: dense OR MoE (family-specific)"]
    ffn --> r2["residual add: x += ffn_out"]
    r2 --> L
    L -->|done| on["rmsnorm output_norm"]
    on --> outw["output proj (tied or output.weight)"]
    outw --> logits["logits over vocab -> argmax"]
```

## 3. llama family

Models: stories260K, llama-3.2-1b (Q8_0 and Q4_K_M).
Dense FFN; attention is multi-head, with grouped KV (GQA) when
`head_count_kv < head_count` (llama-3.2 is 32/8, stories260K is
8/8 = plain MHA). RoPE is the interleaved-pair variant; llama-3.2
carries a large rope base (500000) and optional llama3 scaling
divisors.

```mermaid
flowchart TD
    xb["normed x"] --> q["Q = wq . x"]
    xb --> k["K = wk . x"]
    xb --> v["V = wv . x"]
    q --> rq["RoPE(Q)"]
    k --> rk["RoPE(K)"]
    rk --> kv["append K,V to paged cache"]
    v --> kv
    rq --> sc["per head: scores = Q . K_t * scale"]
    kv --> sc
    sc --> sm["softmax over positions"]
    sm --> av["out_h = sum_t a_t . V_t"]
    av --> ao["attention output"]

    ao -.wo + residual.-> ffnorm["normed x (ffn)"]
    ffnorm --> g["gate = w_gate . x"]
    ffnorm --> u["up = w_up . x"]
    g --> si["silu(gate) * up"]
    u --> si
    si --> d["w_down"]
    d --> out["dense FFN output"]
```

## 4. deepseek2 family (MLA + DeepSeek MoE)

Models: deepseek-v2-lite, moonlight-16b-a3b. MLA weight-absorbed
attention caches one latent row (rms-normed compressed KV plus a
shared roped key) per token instead of full K/V heads. FFN is
dense for the leading layer(s) then DeepSeek MoE (routed experts
+ always-on shared experts). deepseek-v2-lite uses softmax
gating; moonlight uses sigmoid gating.

```mermaid
flowchart TD
    xb["normed x"] --> qy{"q_lora_rank > 0 ?"}
    qy -->|no: direct| q["Q = wq . x"]
    qy -->|yes| qa["wq_a -> rmsnorm q_a_norm -> wq_b"]
    qa --> q
    xb --> kva["wkv_a -> [c_kv | k_pe]"]
    kva --> ckv["rmsnorm(c_kv) = latent"]
    kva --> kpe["RoPE(k_pe) shared key"]
    ckv --> cache["cache latent row (kv_lora + rope)"]
    kpe --> cache

    q --> abs["per head: absorb via wk_b / wkv_b"]
    abs --> sc["scores = q_abs . latents + q_pe . k_pe"]
    cache --> sc
    sc --> sm["softmax"]
    sm --> wl["weighted latent = sum a_t . c_kv(t)"]
    cache --> wl
    wl --> uv["W_uv (wv_b / wkv_b) -> out_h"]

    uv -.wo + residual.-> fn{"layer < n_dense_lead ?"}
    fn -->|yes: dense| dense["dense FFN (gate/up/silu/down)"]
    fn -->|no: MoE| route["gate_inp -> moe_select"]
    route --> exps["top-k routed experts: gate/up/silu/down"]
    route --> shexp["shared experts (always on)"]
    exps --> acc["weighted sum -> FFN output"]
    shexp --> acc
```

MoE routing (`moe_select`, shared by CPU and GPU):

```mermaid
flowchart TD
    rl["router logits = gate_inp . x"] --> gf{"gating func"}
    gf -->|softmax| sfx["softmax"]
    gf -->|sigmoid| sig["sigmoid"]
    sfx --> bias["+ selection bias (exp_probs_b)"]
    sig --> bias
    bias --> grp["group-limited routing (top groups by top-2 sum)"]
    grp --> topk["pick top-k experts"]
    topk --> nrm["optional renorm * expert_weights_scale"]
    nrm --> out["(expert, weight) pairs"]
```

## 5. glm-dsa (MLA + DeepSeek MoE + DSA indexer)

Model: GLM-5.2 (744B, UD-IQ1_S). Computationally the deepseek2
stack (MLA + q_lora + split k_b/v_b, sigmoid gating, 256 experts
/ 8 used / 1 shared, 3 dense-lead layers, rope base 8e6) with one
addition: the DSA lightning indexer selects which cached
positions MLA attends. Below `indexer.top_k` (2048) positions the
selection keeps everything and the layer is identical to dense
MLA (this is what the llama.cpp golden anchors); beyond it, only
the top-k scored positions are attended. The trailing MTP block
(layer 78) is excluded from the causal stack.

```mermaid
flowchart TD
    xb["normed x"] --> idxk["idx_k = LayerNorm+bias(idx_k . x)"]
    idxk --> rhk["half-split RoPE (non-interleaved)"]
    rhk --> icache["cache indexer key (idx_dim)"]

    xb --> cap{"positions > top_k ?"}
    cap -->|no| mlaAll["MLA over ALL cached positions"]
    cap -->|yes| idxq["idx queries from q latent (idx_q_b)"]
    idxq --> rhq["half-split RoPE per head"]
    rhq --> score["score(t) = sum_h w_h . ReLU(q_h . k_idx(t))"]
    icache --> score
    xb --> hw["head weights w = idx_proj . x"]
    hw --> score
    score --> sel["top-k positions (nth_element + sort)"]
    sel --> mlaSel["MLA over SELECTED positions only"]

    mlaAll --> ffn["FFN: dense x3 then sigmoid-gated MoE"]
    mlaSel --> ffn
```

## 6. Attention families at a glance

```
llama (MHA/GQA)      deepseek2 / glm-dsa (MLA)
-----------------    -------------------------------
cache: K,V per head  cache: 1 latent row per token
  (kv_heads * head_    (kv_lora + rope; glm-dsa adds
   dim, 2 tensors)      idx_dim for the indexer key)
scores: Q . K        scores: absorbed q . latents
                       + q_pe . k_pe
value: sum a . V     value: W_uv (sum a . c_kv)
select: all past     select: all past (deepseek2) or
                       DSA top-k (glm-dsa > top_k)
```

## 7. Where the diagrams live in code

| Stage                | Source                                        |
|----------------------|-----------------------------------------------|
| outer forward loop   | `LlamaModel::forward` (llama.cpp)             |
| llama attention      | `llama_attention` (arch.cpp)                  |
| MLA attention core   | `mla_attention` (arch.cpp)                    |
| deepseek2 attention  | `deepseek2_attention` -> mla_attention        |
| glm-dsa attention    | `glm_dsa_attention` (indexer + mla_attention) |
| dense / MoE FFN      | `LlamaModel::forward` / `moe_ffn` (llama.cpp) |
| expert routing       | `moe_select` (llama.cpp)                       |
