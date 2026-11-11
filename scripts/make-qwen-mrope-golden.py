#!/usr/bin/env python3
"""Makes tests/golden/qwen_mrope.bin, the M-RoPE fixture that locus's
qwen2vl arch is compared against (i#545). No model needed.

A tiny Qwen2.5-VL language model (width 64, 4 heads of 16, 2 KV heads,
2 layers, vocab 64, mrope_section [2, 3, 3]) filled from the test's LCG
in SORTED GGUF tensor-name order. tests/test_qwen_mrope.cpp builds the
same qwen2vl GGUF in memory. The prompt is 5 text tokens (the last one
vision_start), a 2 x 3 merged image span of LCG input embeddings, then
4 text tokens. transformers' get_rope_index gives the positions, and
the language model and lm_head give every position's logits.

Writes, little-endian: the positions (3 x n int32, the t row, then h,
then w), then n x 64 float32 logits.

    run-offline.sh scripts/make-qwen-mrope-golden.py

Tested with Python 3.12, torch 2.5.1, transformers 4.56.1 and numpy
2.1.3. With those, on an M2 Pro (CPU), its sha256 is
7535415883a11651d0cc649423b6f38e9614cee7b4238f91943fdb98642dfb79.
"""
import pathlib

import numpy as np
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

GOLDEN = pathlib.Path(__file__).resolve().parent.parent / "tests" / "golden"

# Keep in sync with tests/test_qwen_mrope.cpp.
E, HEADS, KV, FF, LAYERS, VOCAB = 64, 4, 2, 96, 2, 64
SECTIONS = [2, 3, 3]
IMAGE_ID, START_ID = 60, 61
TEXT_BEFORE = [3, 17, 42, 9, START_ID]
GRID = (2, 3)  # merged rows and columns
TEXT_AFTER = [5, 33, 21, 58]


def lcg(state, n):
    """n values of the test's LCG. @returns (values, state)."""
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        vals[i] = (np.float32(state >> 8) / np.float32(16777216.0) -
                   np.float32(0.5)) * np.float32(0.5)
    return vals, state


def gguf_shapes():
    """@returns {GGUF tensor name: torch-order shape}."""
    hd = E // HEADS
    s = {"token_embd.weight": (VOCAB, E), "output_norm.weight": (E,),
         "output.weight": (VOCAB, E)}
    for i in range(LAYERS):
        b = f"blk.{i}."
        s[b + "attn_norm.weight"] = (E,)
        s[b + "ffn_norm.weight"] = (E,)
        s[b + "attn_q.weight"] = (E, E)
        s[b + "attn_q.bias"] = (E,)
        for t in ("attn_k", "attn_v"):
            s[b + t + ".weight"] = (KV * hd, E)
            s[b + t + ".bias"] = (KV * hd,)
        s[b + "attn_output.weight"] = (E, E)
        s[b + "ffn_gate.weight"] = (FF, E)
        s[b + "ffn_up.weight"] = (FF, E)
        s[b + "ffn_down.weight"] = (E, FF)
    return s


def to_hf(t):
    """GGUF names to transformers' Qwen2_5_VLForConditionalGeneration."""
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    p = "model.language_model."
    sd = {p + "embed_tokens.weight": T(t["token_embd.weight"]),
          p + "norm.weight": T(t["output_norm.weight"]),
          "lm_head.weight": T(t["output.weight"])}
    for i in range(LAYERS):
        b, h = f"blk.{i}.", f"{p}layers.{i}."
        sd[h + "input_layernorm.weight"] = T(t[b + "attn_norm.weight"])
        sd[h + "post_attention_layernorm.weight"] = T(
            t[b + "ffn_norm.weight"])
        for g, l in (("q_proj", "attn_q"), ("k_proj", "attn_k"),
                     ("v_proj", "attn_v")):
            sd[h + f"self_attn.{g}.weight"] = T(t[b + l + ".weight"])
            sd[h + f"self_attn.{g}.bias"] = T(t[b + l + ".bias"])
        sd[h + "self_attn.o_proj.weight"] = T(t[b + "attn_output.weight"])
        for g, l in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"),
                     ("down_proj", "ffn_down")):
            sd[h + f"mlp.{g}.weight"] = T(t[b + l + ".weight"])
    return sd


def main():
    shapes = gguf_shapes()
    state, t = 12345, {}
    for name in sorted(shapes):
        vals, state = lcg(state, int(np.prod(shapes[name])))
        t[name] = vals.reshape(shapes[name])
    gh, gw = GRID
    n_img = gh * gw
    img, _ = lcg(777, n_img * E)
    img = torch.from_numpy(img * np.float32(4.0)).reshape(n_img, E)

    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=VOCAB, hidden_size=E,
                         intermediate_size=FF, num_hidden_layers=LAYERS,
                         num_attention_heads=HEADS, num_key_value_heads=KV,
                         rms_norm_eps=1e-6, rope_theta=10000.0,
                         max_position_embeddings=128, hidden_act="silu",
                         tie_word_embeddings=False,
                         rope_scaling={"type": "mrope",
                                       "mrope_section": SECTIONS}),
        vision_config=dict(depth=1, hidden_size=16, num_heads=2,
                           intermediate_size=16, out_hidden_size=E,
                           fullatt_block_indexes=[0]),
        image_token_id=IMAGE_ID, vision_start_token_id=START_ID,
        video_token_id=62, vocab_size=VOCAB, tie_word_embeddings=False)
    cfg.text_config._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    m = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    missing, unexpected = m.load_state_dict(to_hf(t), strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("model.visual.") or "inv_freq" in k
               for k in missing), missing

    ids = TEXT_BEFORE + [IMAGE_ID] * n_img + TEXT_AFTER
    input_ids = torch.tensor([ids])
    # The image grid in patches (2x2 merged), as the processor gives it.
    grid = torch.tensor([[1, 2 * gh, 2 * gw]])
    pos, _ = m.model.get_rope_index(input_ids, image_grid_thw=grid)
    assert pos.shape == (3, 1, len(ids)), pos.shape
    lm = m.model.language_model
    with torch.no_grad():
        emb = lm.embed_tokens(input_ids)
        b = len(TEXT_BEFORE)
        emb[0, b:b + n_img] = img
        h = lm(inputs_embeds=emb, position_ids=pos).last_hidden_state
        logits = m.lm_head(h)[0]
    blob = (pos[:, 0, :].numpy().astype("<i4").tobytes() +
            logits.numpy().astype("<f4").tobytes())
    path = GOLDEN / "qwen_mrope.bin"
    path.write_bytes(blob)
    print(path, pos[:, 0, :].tolist(), logits.shape)


if __name__ == "__main__":
    main()
