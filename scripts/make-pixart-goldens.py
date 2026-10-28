#!/usr/bin/env python3
"""Writes PixArt-alpha 512 reference tensors (i#146, i#307).

Runs diffusers' PixArtAlphaPipeline on the CPU from a local copy of
PixArt-alpha/PixArt-XL-2-512x512 (commit 50f7021): T5 in bfloat16
(its weights ship as fp16), the DiT and the VAE in float32. At a fixed
prompt, seed and step count it saves, as raw little-endian float32
(int64 for token ids) plus a JSON index:

  t5_ids, t5_mask        the prompt's 120 token ids and attention mask
  t5_out                 the encoder output for the prompt [120, 4096]
  t5_neg_out             the same for the empty negative prompt
  dit_in, dit_t, dit_out the first denoiser call: the CFG-doubled latent
                         [2, 4, 64, 64], the timestep, and its output
                         [2, 8, 64, 64] (epsilon and learned variance)
  latent_final           the latent after the last step [1, 4, 64, 64]
  image.png, image.f32   the VAE decode of it, [512, 512, 3] in 0..255

Needs torch 2.5.1 (CPU), diffusers 0.31.0, transformers 4.46.3,
sentencepiece, ftfy and beautifulsoup4 (the prompt cleaning), and
about 15 GB of RAM. On a 4-core x86 box the T5 pass takes about 15
minutes and the 8 steps about 7.

    python -I scripts/make-pixart-goldens.py MODEL_DIR OUT_DIR [STEPS]
"""

import json
import os
import sys

import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from transformers import T5EncoderModel

PROMPT = "A small red apple on a wooden table, soft morning light."
SEED = 146
GUIDANCE = 4.5


def save(out, index, name, t):
    a = t.detach().to(torch.float32 if t.is_floating_point() else torch.int64)
    a = a.cpu().numpy()
    a.tofile(os.path.join(out, name + (".f32" if a.dtype == np.float32
                                        else ".i64")))
    index[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    # The T5 shards are the fp16 variant; it runs in bfloat16.
    t5 = T5EncoderModel.from_pretrained(os.path.join(model_dir,
                                                     "text_encoder"),
                                        variant="fp16",
                                        torch_dtype=torch.bfloat16)
    pipe = PixArtAlphaPipeline.from_pretrained(model_dir, text_encoder=t5,
                                               torch_dtype=torch.float32)
    index = {"prompt": PROMPT, "seed": SEED, "steps": steps,
             "guidance": GUIDANCE, "revision": "50f7021"}

    # The text encoder, as the pipeline calls it.
    with torch.no_grad():
        ids = pipe.tokenizer(pipe._text_preprocessing(PROMPT, True),
                             padding="max_length", max_length=120,
                             truncation=True, add_special_tokens=True,
                             return_tensors="pt")
        save(out, index, "t5_ids", ids.input_ids[0])
        save(out, index, "t5_mask", ids.attention_mask[0])
        emb, mask, neg, neg_mask = pipe.encode_prompt(
            PROMPT, do_classifier_free_guidance=True, negative_prompt="",
            device="cpu", clean_caption=True, max_sequence_length=120)
        save(out, index, "t5_out", emb[0])
        save(out, index, "t5_neg_out", neg[0])

    # The first denoiser call, captured with a forward hook.
    first = {}

    def hook(mod, args, kwargs, output):
        if first:
            return
        first["in"] = kwargs.get("hidden_states", args[0] if args else None)
        first["t"] = kwargs["timestep"]
        first["out"] = output[0] if isinstance(output, tuple) else output.sample

    # With an encoder attached, encode_prompt casts even given
    # embeddings to its dtype (bf16); without one it uses the DiT's.
    pipe.text_encoder = None
    del t5
    h = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
    gen = torch.Generator("cpu").manual_seed(SEED)
    # The bf16 encoder states go in as float32 (the DiT's dtype), so the
    # pipeline does not run the encoder again or mix dtypes.
    with torch.no_grad():
        latents = pipe(prompt=None, negative_prompt=None,
                       prompt_embeds=emb.float(),
                       prompt_attention_mask=mask,
                       negative_prompt_embeds=neg.float(),
                       negative_prompt_attention_mask=neg_mask,
                       num_inference_steps=steps, guidance_scale=GUIDANCE,
                       height=512, width=512, generator=gen,
                       output_type="latent").images
    h.remove()
    save(out, index, "dit_in", first["in"])
    save(out, index, "dit_t", first["t"].to(torch.float32))
    save(out, index, "dit_out", first["out"])
    save(out, index, "latent_final", latents)

    with torch.no_grad():
        dec = pipe.vae.decode(latents / pipe.vae.config.scaling_factor,
                              return_dict=False)[0]
    img = (dec / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255.0
    save(out, index, "image", img)
    pipe.image_processor.postprocess(dec, output_type="pil")[0].save(
        os.path.join(out, "image.png"))
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
