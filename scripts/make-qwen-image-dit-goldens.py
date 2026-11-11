#!/usr/bin/env python3
"""Makes the Qwen-Image MMDiT fixtures under tests/golden that locus's
image::QwenImageDit is compared against (i#532). No model needed.

A tiny diffusers QwenImageTransformer2DModel (patch 2, 16 input and 4
output channels, 2 blocks, 2 heads of 32, RoPE axes 8 / 12 / 12, joint
width 64), its parameters filled from the test's LCG in SORTED name
order and saved in F32, then converted by scripts/convert-qwen-image.py
(so the fixture is the converter's own output). Writes:

  qwen_image_dit_synth.gguf  the model ("qwen_image", all F32);
  qwen_image_dit_synth.f32   for an LCG input of two image segments
                             (the noise latent, 4 x 6 patches, frame 0;
                             a reference image, 2 x 3 patches, frame 1)
                             and 5 LCG text rows at sigma 0.7: the image
                             and text rows after block 0, then the
                             velocity of every image token.

    run-offline.sh scripts/make-qwen-image-dit-goldens.py

With --model (the pipeline directory, Qwen/Qwen-Image-Edit at ac7f931;
only the transformer shards holding block 0 and the layers outside the
blocks are read, 00001 and 00009), it writes instead:

  qwen_image_dit_real.f32    a one-block model from the real weights
                             (block 0 and the layers outside the blocks,
                             float32 from BF16) on two LCG image
                             segments (2 x 4 and 2 x 2 patches) and 4
                             LCG text rows at sigma 0.5: the image and
                             text rows after block 0, then the velocity.
                             The matching GGUF is convert-qwen-image.py
                             --blocks 1 (BF16) on the same shards.

Tested with Python 3.12, torch 2.5.1, diffusers 0.35.1, gguf 0.17.1 and
numpy 2.1.3. With those, the outputs' sha256 (on an M4 Max, CPU):

  qwen_image_dit_synth.gguf
    22fc5ef8bb279786c6a760f00579ebd1bc50f1cd7c4df67a5ced8471738fafd4
  qwen_image_dit_synth.f32
    fb06bd7b33cd80f50c41b2f4c4d172ca0bf380b34b4a6bd519b269c2ebd7eb94
  qwen_image_dit_real.f32 (--model)
    72704fe3783f77bb9bb0441fc8b7cff0c1cc1203f600dbf009e34f1c3661fe88
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from diffusers import QwenImageTransformer2DModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"

# Keep in sync with tests/test_qwen_image_dit.cpp.
CFG = dict(patch_size=2, in_channels=16, out_channels=4, num_layers=2,
           attention_head_dim=32, num_attention_heads=2,
           joint_attention_dim=64, guidance_embeds=False,
           axes_dims_rope=(8, 12, 12))
SEGS = [(1, 4, 6), (1, 2, 3)]  # (frames, rows, columns) in patches
N_TXT = 5
SIGMA = 0.7
SCHED = dict(_class_name="FlowMatchEulerDiscreteScheduler",
             base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192,
             max_shift=0.9, num_train_timesteps=1000, shift=1.0,
             shift_terminal=0.02, time_shift_type="exponential",
             use_dynamic_shifting=True)


def lcg(state, n):
    """n values of the test's LCG. @returns (values, state)."""
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        vals[i] = (np.float32(state >> 8) / np.float32(16777216.0) -
                   np.float32(0.5)) * np.float32(0.5)
    return vals, state


REAL_SEGS = [(1, 2, 4), (1, 2, 2)]
REAL_TXT = 4
REAL_SIGMA = 0.5


def real(mdir):
    """The real model's block 0 on a small input (see the docstring)."""
    from safetensors import safe_open
    cfg = json.loads((mdir / "transformer" / "config.json").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["num_layers"] = 1
    m = QwenImageTransformer2DModel(**cfg).float().eval()
    idx = json.loads((mdir / "transformer" /
                      "diffusion_pytorch_model.safetensors.index.json")
                     .read_text())["weight_map"]
    sd = m.state_dict()
    for name in sorted(sd):
        with safe_open(str(mdir / "transformer" / idx[name]), "pt") as f:
            sd[name].copy_(f.get_tensor(name).float())
    n_img = sum(f * h * w for f, h, w in REAL_SEGS)
    x, _ = lcg(777, n_img * cfg["in_channels"])
    txt, _ = lcg(999, REAL_TXT * cfg["joint_attention_dim"])
    x = torch.from_numpy(x * np.float32(4.0)).view(1, n_img, -1)
    txt = torch.from_numpy(txt * np.float32(4.0)).view(1, REAL_TXT, -1)
    trace = {}

    def hook(_mod, _inp, outp):
        trace["b0"] = (outp[0].detach().clone(), outp[1].detach().clone())
    m.transformer_blocks[0].register_forward_hook(hook)
    with torch.no_grad():
        v = m(hidden_states=x, encoder_hidden_states=txt,
          encoder_hidden_states_mask=torch.ones(1, REAL_TXT),
          timestep=torch.tensor([REAL_SIGMA]), img_shapes=[REAL_SEGS],
          txt_seq_lens=[REAL_TXT], return_dict=False)[0]
    txt0, img0 = trace["b0"]
    blob = np.concatenate([img0.numpy().ravel(), txt0.numpy().ravel(),
                           v.numpy().ravel()])
    data = blob.astype("<f4").tobytes()
    (GOLDEN / "qwen_image_dit_real.f32").write_bytes(data)
    print("qwen_image_dit_real.f32", hashlib.sha256(data).hexdigest(),
          "rms img0 %g txt0 %g" % (float(img0.pow(2).mean().sqrt()),
                                   float(txt0.pow(2).mean().sqrt())))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--model":
        real(pathlib.Path(sys.argv[2]))
        return
    torch.manual_seed(0)
    m = QwenImageTransformer2DModel(**CFG).float().eval()
    sd = m.state_dict()
    state = 4242
    for name in sorted(sd):
        vals, state = lcg(state, sd[name].numel())
        sd[name].copy_(torch.from_numpy(vals).view(sd[name].shape))

    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "model_index.json").write_text(
            json.dumps({"_class_name": "QwenImageEditPipeline"}))
        (d / "scheduler").mkdir()
        (d / "scheduler" / "scheduler_config.json").write_text(
            json.dumps(SCHED))
        m.save_pretrained(d / "transformer")
        out = d / "qwen_image_dit_synth.gguf"
        subprocess.run([sys.executable, "-I",
                        str(ROOT / "scripts" / "convert-qwen-image.py"),
                        "--model", str(d), "--dit-out", str(out)],
                       check=True)
        shutil.copy(out, GOLDEN / "qwen_image_dit_synth.gguf")

    n_img = sum(f * h * w for f, h, w in SEGS)
    x, _ = lcg(777, n_img * CFG["in_channels"])
    txt, _ = lcg(999, N_TXT * CFG["joint_attention_dim"])
    x = torch.from_numpy(x * np.float32(4.0)).view(1, n_img, -1)
    txt = torch.from_numpy(txt * np.float32(4.0)).view(1, N_TXT, -1)
    trace = {}

    def hook(_mod, _inp, outp):
        trace.setdefault("b0", (outp[0].detach().clone(),
                                outp[1].detach().clone()))
    m.transformer_blocks[0].register_forward_hook(hook)
    with torch.no_grad():
        v = m(hidden_states=x, encoder_hidden_states=txt,
              encoder_hidden_states_mask=torch.ones(1, N_TXT),
              timestep=torch.tensor([SIGMA]), img_shapes=[SEGS],
              txt_seq_lens=[N_TXT], return_dict=False)[0]
    txt0, img0 = trace["b0"]
    blob = np.concatenate([img0.numpy().ravel(), txt0.numpy().ravel(),
                           v.numpy().ravel()])
    data = blob.astype("<f4").tobytes()
    (GOLDEN / "qwen_image_dit_synth.f32").write_bytes(data)
    for name in ("qwen_image_dit_synth.gguf", "qwen_image_dit_synth.f32"):
        print(name, hashlib.sha256((GOLDEN / name).read_bytes()).hexdigest())
    print("rms img0 %g txt0 %g v %g" % (
        float(img0.pow(2).mean().sqrt()), float(txt0.pow(2).mean().sqrt()),
        float(v.pow(2).mean().sqrt())))


if __name__ == "__main__":
    main()
