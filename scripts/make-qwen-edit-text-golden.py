#!/usr/bin/env python3
"""Makes tests/golden/qwen_edit_text_synth.f32, the fixture that locus's
image::qwen_edit_text_embeds is compared against (i#574). No model
needed.

A tiny Qwen2.5-VL: the language model of make-qwen-mrope-golden.py and
the vision tower of make-qwen-vision-goldens.py synth with its merger
projecting to the model's width (64), each filled from the tests' LCG
in sorted GGUF tensor-name order (tests/qwen_tiny.hpp builds the same
two GGUFs). The prompt is 70 text tokens and <|vision_start|>, the
image span (a 4 x 6 patch grid of LCG patches: 2 x 3 merged tokens),
then 5 text tokens. transformers runs the whole model, as diffusers'
QwenImageEditPipeline._get_qwen_prompt_embeds does (input ids, pixel
values, the grid, output_hidden_states), and the golden is
hidden_states[-1] with the first 64 positions dropped.

Writes, little-endian: 18 x 64 float32.

`ids` writes tests/golden/qwen_edit_prompt_ids.i32: the reference
processor's input ids (int32) for diffusers' edit template with
INSTRUCTION and a 56 x 84 image (a 4 x 6 patch grid, 6 image tokens),
from the processor/ folder of Qwen/Qwen-Image-Edit at ac7f931 (only
its tokenizer and preprocessor files; no model code).

    run-offline.sh scripts/make-qwen-edit-text-golden.py
    run-offline.sh scripts/make-qwen-edit-text-golden.py ids --processor D

Tested with Python 3.12, torch 2.5.1, transformers 4.56.1, numpy 2.1.3,
diffusers 0.35.1 and pillow 11.3.0.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE.parent / "tests" / "golden"


def load(name):
    """@returns The sibling script `name` as a module (no .pyc left in
    scripts/)."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
                                                  HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mrope = load("make-qwen-mrope-golden")
vis = load("make-qwen-vision-goldens")

# Keep in sync with tests/test_qwen_edit_text.cpp.
E = mrope.E
VISION = dict(vis.SYNTH, out_hidden_size=E)
BEFORE = [(7 * i + 3) % 60 for i in range(70)] + [mrope.START_ID]
GRID = (4, 6)  # patch rows and columns
AFTER = [5, 33, 21, 58, 12]
DROP = 64


def filled(shapes):
    """@returns {name: array} from the LCG, seed 12345, sorted names."""
    state, t = 12345, {}
    for name in sorted(shapes):
        vals, state = mrope.lcg(state, int(np.prod(shapes[name])))
        t[name] = vals.reshape(shapes[name])
    return t


INSTRUCTION = "Make the sky purple, and add a small red kite."


def prompt_ids(processor):
    """The reference processor's ids for the edit template."""
    from diffusers import QwenImageEditPipeline
    from PIL import Image
    from transformers import Qwen2VLProcessor

    # The template as the pipeline sets it in __init__, without loading
    # a pipeline.
    template = QwenImageEditPipeline.__init__.__code__.co_consts
    template = next(c for c in template
                    if isinstance(c, str) and "<|image_pad|>" in c)
    proc = Qwen2VLProcessor.from_pretrained(processor, use_fast=False)
    img = Image.new("RGB", (84, 56), (200, 120, 40))
    out = proc(text=[template.format(INSTRUCTION)], images=[img],
               return_tensors="np")
    assert out["image_grid_thw"].tolist() == [[1, 4, 6]], (
        out["image_grid_thw"])
    got = out["input_ids"][0].astype("<i4")
    path = GOLDEN / "qwen_edit_prompt_ids.i32"
    path.write_bytes(got.tobytes())
    print(path, len(got), template.count("{}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="synth",
                    choices=["synth", "ids"])
    ap.add_argument("--processor", type=pathlib.Path)
    a = ap.parse_args()
    if a.mode == "ids":
        prompt_ids(a.processor)
        return
    lm = filled(mrope.gguf_shapes())
    tower = filled(vis.gguf_shapes(VISION))
    sd = mrope.to_hf(lm)
    for k, v in vis.to_hf(tower, VISION["depth"]).items():
        sd["model.visual." + k] = v

    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=mrope.VOCAB, hidden_size=E,
                         intermediate_size=mrope.FF,
                         num_hidden_layers=mrope.LAYERS,
                         num_attention_heads=mrope.HEADS,
                         num_key_value_heads=mrope.KV, rms_norm_eps=1e-6,
                         rope_theta=10000.0, max_position_embeddings=128,
                         hidden_act="silu", tie_word_embeddings=False,
                         rope_scaling={"type": "mrope",
                                       "mrope_section": mrope.SECTIONS}),
        vision_config=VISION,
        image_token_id=mrope.IMAGE_ID, vision_start_token_id=mrope.START_ID,
        video_token_id=62, vocab_size=mrope.VOCAB, tie_word_embeddings=False)
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    m = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all("inv_freq" in k for k in missing), missing

    gh, gw = GRID
    n_img = gh * gw // 4
    ids = BEFORE + [mrope.IMAGE_ID] * n_img + AFTER
    pd = 3 * 2 * 14 * 14
    px, _ = mrope.lcg(777, gh * gw * pd)
    px = torch.from_numpy(px * np.float32(4.0)).reshape(gh * gw, pd)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                pixel_values=px,
                image_grid_thw=torch.tensor([[1, gh, gw]]),
                output_hidden_states=True)
    h = out.hidden_states[-1][0, DROP:]
    assert h.shape == (len(ids) - DROP, E), h.shape
    path = GOLDEN / "qwen_edit_text_synth.f32"
    path.write_bytes(h.numpy().astype("<f4").tobytes())
    print(path, tuple(h.shape), float(h.pow(2).mean().sqrt()))


if __name__ == "__main__":
    main()
