"""IPOL demo runner: embeddings + cosine similarity + threshold.

Input:
  --pairs  text file, each line: <label> <img1> <img2>
           label: 1=same, 0=different
           img paths are relative to --img_dir
  --img_dir directory containing the images referenced by pairs file
  --ckpt  path to model checkpoint (.pth)

Output:
  Prints best accuracy and best threshold (grid search).
  Saves ROC curve, score histogram, and a JSON summary in --out_dir.

Runtime note:
  Keep the number of pairs small (e.g., <= 1000) to stay under ~20s on CPU.
"""

from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import LightFaceSAMV2, infer_num_classes_from_ckpt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="Pairs annotation file")
    ap.add_argument("--img_dir", type=str, required=True, help="Image directory")
    ap.add_argument("--ckpt", type=str, default="models/best_lightface_sam_v2.pth", help="Checkpoint path")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    ap.add_argument("--th_min", type=float, default=-1.0, help="threshold sweep min")
    ap.add_argument("--th_max", type=float, default=1.0, help="threshold sweep max")
    ap.add_argument("--th_steps", type=int, default=400, help="threshold sweep steps")
    ap.add_argument("--out_dir", type=str, default="results", help="output directory")
    ap.add_argument("--max_pairs", type=int, default=1000, help="cap number of pairs for runtime")
    return ap.parse_args()


def load_pairs(pairs_path: str, img_dir: str, max_pairs: int):
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            y, p1, p2 = parts
            try:
                y = int(y)
            except ValueError:
                continue
            img1 = os.path.join(img_dir, p1)
            img2 = os.path.join(img_dir, p2)
            if os.path.isfile(img1) and os.path.isfile(img2):
                pairs.append((y, img1, img2))
            if len(pairs) >= max_pairs:
                break
    if len(pairs) == 0:
        raise RuntimeError("No valid pairs found. Check --pairs and --img_dir.")
    return pairs


def make_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def load_image(path: str, tfm, device):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return x


@torch.no_grad()
def embed(model, img_path: str, tfm, device, cache: dict):
    if img_path in cache:
        return cache[img_path]
    x = load_image(img_path, tfm, device)
    e = model(x, labels=None).cpu().numpy()[0]
    cache[img_path] = e
    return e


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def sweep_threshold(scores: np.ndarray, labels: np.ndarray, th_min: float, th_max: float, th_steps: int):
    best_acc, best_th = -1.0, 0.0
    ths = np.linspace(th_min, th_max, th_steps)
    accs = []
    for th in ths:
        preds = (scores > th).astype(np.int32)
        acc = (preds == labels).mean()
        accs.append(acc)
        if acc > best_acc:
            best_acc, best_th = float(acc), float(th)
    return best_acc, best_th, ths, np.array(accs, dtype=np.float32)


def roc_curve(scores: np.ndarray, labels: np.ndarray, ths: np.ndarray):
    # labels: 1=same (positive)
    tprs, fprs = [], []
    P = max(labels.sum(), 1)
    N = max((1 - labels).sum(), 1)
    for th in ths:
        preds = (scores > th).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tprs.append(tp / P)
        fprs.append(fp / N)
    return np.array(fprs, dtype=np.float32), np.array(tprs, dtype=np.float32)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load checkpoint and infer num_classes
    state = torch.load(args.ckpt, map_location=device)
    num_classes = infer_num_classes_from_ckpt(state)

    model = LightFaceSAMV2(num_classes=num_classes, bottleneck=128, emb_dim=256,
                           s=30.0, m=0.2, imagenet_pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    pairs = load_pairs(args.pairs, args.img_dir, args.max_pairs)
    tfm = make_transform()
    cache = {}

    labels = []
    scores = []

    t0 = time.time()
    for y, p1, p2 in pairs:
        e1 = embed(model, p1, tfm, device, cache)
        e2 = embed(model, p2, tfm, device, cache)
        scores.append(cosine(e1, e2))
        labels.append(y)
    elapsed = time.time() - t0

    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    best_acc, best_th, ths, accs = sweep_threshold(scores, labels, args.th_min, args.th_max, args.th_steps)
    fprs, tprs = roc_curve(scores, labels, ths)

    # Save plots
    import matplotlib.pyplot as plt

    # ROC
    plt.figure(figsize=(4.2, 3.6), dpi=300)
    plt.plot(fprs, tprs)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Demo Pairs)")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(args.out_dir, "demo_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # Score histogram
    plt.figure(figsize=(4.2, 3.6), dpi=300)
    plt.hist(scores[labels == 1], bins=40, alpha=0.7, label="same (1)")
    plt.hist(scores[labels == 0], bins=40, alpha=0.7, label="different (0)")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Score distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    hist_path = os.path.join(args.out_dir, "demo_score_hist.png")
    plt.tight_layout()
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()

    # Threshold sweep
    plt.figure(figsize=(4.2, 3.6), dpi=300)
    plt.plot(ths, accs)
    plt.axvline(best_th, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs threshold")
    plt.grid(True, alpha=0.3)
    sweep_path = os.path.join(args.out_dir, "demo_threshold_sweep.png")
    plt.tight_layout()
    plt.savefig(sweep_path, bbox_inches="tight")
    plt.close()

    summary = {
        "pairs_used": int(len(pairs)),
        "unique_images": int(len(cache)),
        "best_accuracy": float(best_acc * 100.0),
        "best_threshold": float(best_th),
        "runtime_seconds": float(elapsed),
        "device": str(device),
        "outputs": {
            "roc": os.path.basename(roc_path),
            "hist": os.path.basename(hist_path),
            "threshold_sweep": os.path.basename(sweep_path),
        }
    }
    with open(os.path.join(args.out_dir, "demo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Demo summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
