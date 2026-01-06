# src/ipol_online_demo.py
"""
IPOL online demo (pair-based): embeddings + cosine similarity + threshold decision.

Inputs (from IPOL):
  input_0.png  first aligned face crop
  input_1.png  second aligned face crop

Outputs (fixed filenames, for DDL "results"):
  inputs.png       side-by-side resized preview (112x112 + 112x112)
  similarity.txt   cosine similarity
  decision.txt     "same" or "different"
  result.json      machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import LightFaceSAMV2, infer_num_classes_from_ckpt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("img1", type=str, help="first input image (e.g., input_0.png)")
    ap.add_argument("img2", type=str, help="second input image (e.g., input_1.png)")
    ap.add_argument("--th", type=float, default=0.30, help="cosine similarity threshold")
    ap.add_argument("--ckpt", type=str, default="models/best_lightface_sam_v2.pth", help="checkpoint path")
    return ap.parse_args()


def make_transform():
    # EXACTLY your demo.py normalization and size
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    a = a.resize((112, 112))
    b = b.resize((112, 112))
    out = Image.new("RGB", (224, 112))
    out.paste(a, (0, 0))
    out.paste(b, (112, 0))
    return out


def main():
    args = parse_args()
    t0 = time.time()

    device = torch.device("cpu")  # online demo: CPU only

    # Load model
    state = torch.load(args.ckpt, map_location="cpu")
    num_classes = infer_num_classes_from_ckpt(state)

    model = LightFaceSAMV2(
        num_classes=num_classes,
        bottleneck=128,
        emb_dim=256,
        s=30.0,
        m=0.2,
        imagenet_pretrained=False
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    tfm = make_transform()

    im1 = load_rgb(args.img1)
    im2 = load_rgb(args.img2)

    x1 = tfm(im1).unsqueeze(0)
    x2 = tfm(im2).unsqueeze(0)
    x = torch.cat([x1, x2], dim=0).to(device)  # [2,3,112,112]

    with torch.no_grad():
        emb = model(x, labels=None).cpu().numpy()  # [2,256], L2-normalized
    score = float(np.dot(emb[0], emb[1]))          # cosine due to normalization
    decision = "same" if score > args.th else "different"

    # Write outputs with fixed names (DDL will display these)
    side_by_side(im1, im2).save("inputs.png")

    Path("similarity.txt").write_text(f"{score:.6f}\n", encoding="utf-8")
    Path("decision.txt").write_text(f"{decision}\n", encoding="utf-8")

    summary = {
        "similarity": score,
        "threshold": float(args.th),
        "decision": decision,
        "runtime_seconds": float(time.time() - t0),
        "device": "cpu",
        "ckpt": args.ckpt,
        "note": "Inference only (no training)."
    }
    Path("result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # stdout is fine too
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
