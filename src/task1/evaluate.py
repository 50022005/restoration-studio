"""
src/task1/evaluate.py
───────────────────────────────────────────────────────────────────────────────
Evaluation script for Task 1: Universal Denoising Autoencoder.

Computes:
  • SSIM and L1 metrics per corruption type and severity level
  • 12-example reconstruction visual grid (Clean, Corrupted, Recon, Error Map)
  • 4 failure cases grid (lowest SSIM examples)

Usage:
    python src/task1/evaluate.py --config configs/task1.yaml
    python src/task1/evaluate.py --smoke
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task1.model import UniversalAE
from src.utils.corruption import (
    CORRUPTION_TYPES,
    corrupt_from_record,
)
from src.utils.datasets import PetDataset
from src.utils.losses import ssim
from src.utils.training import load_checkpoint, set_seeds


def resolve_img_path(raw_path: str) -> Path:
    p = Path(raw_path.replace(".jpg", ".npy").replace(".png", ".npy"))
    if p.exists():
        return p
    fname = p.name
    for c_dir in [
        Path("/content/drive/MyDrive/genai_assignment/data/pets_cache/test"),
        Path("/content/drive/MyDrive/genai_assignment/data/pets_cache/val"),
        Path("/content/drive/MyDrive/genai_assignment/data/pets_cache/train"),
        Path("data/pets_cache/test"),
        Path("data/pets_cache/val"),
        Path("data/pets_cache/train"),
    ]:
        if (c_dir / fname).exists():
            return c_dir / fname
    return p


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return (t.squeeze().permute(1, 2, 0).cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)


def plot_visual_grid(model, manifest, n=12, title="Reconstruction Grid", device="cpu", save_path="grid.png"):
    model.eval()
    idxs = list(range(0, len(manifest), max(1, len(manifest) // n)))[:n]

    fig, axes = plt.subplots(len(idxs), 4, figsize=(12, max(2.5 * len(idxs), 3)))
    if len(idxs) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, weight="bold")
    cols = ["Clean Target", "Corrupted Input", "Reconstructed", "|Error| x 3"]

    for col, ax in enumerate(axes[0]):
        ax.set_title(cols[col], fontsize=9)

    for row, idx in enumerate(idxs):
        rec = manifest[idx]
        img_p = resolve_img_path(rec["image_path"])
        if not img_p.exists():
            clean_np = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            clean_np = np.load(str(img_p))
        corr_np = corrupt_from_record(clean_np, rec)

        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        corr_t = torch.from_numpy(corr_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            recon_t = model(corr_t.to(device)).cpu()

        error_t = (clean_t - recon_t).abs() * 3
        imgs = [clean_t, corr_t, recon_t, error_t.clamp(0, 1)]
        label = f"{rec['corruption_type']} / {rec.get('severity_level', '')}"

        for col, img_t in enumerate(imgs):
            axes[row][col].imshow(tensor_to_np(img_t))
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(label, fontsize=7, rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved visual grid -> {save_path}")


def load_uae_from_checkpoint(ckpt_path: str, device: torch.device) -> UniversalAE:
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if "model" in state else state
    base_channels = state_dict["enc.0.block.0.weight"].shape[0]
    bottleneck_dim = state_dict["fc_enc.1.weight"].shape[0]
    model = UniversalAE(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 1 Universal Autoencoder")
    parser.add_argument("--config", default="configs/task1.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu" if args.smoke or not torch.cuda.is_available() else "cuda")
    set_seeds(42)

    manifest_dir = REPO_ROOT / "data" / "manifests"
    test_manifest_path = manifest_dir / "test_manifest.json"

    if args.smoke and not test_manifest_path.exists():
        from src.task1.train import ensure_dummy_data_for_smoke
        ensure_dummy_data_for_smoke(REPO_ROOT)

    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Test manifest missing at {test_manifest_path}")

    with open(test_manifest_path) as f:
        test_manifest = json.load(f)

    if args.smoke:
        test_manifest = test_manifest[:10]

    # Find checkpoint
    ckpt_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints/task1")
    if args.smoke:
        ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task1")
    ckpt_path = args.checkpoint or str(Path(ckpt_dir) / "best.pt")
    if not Path(ckpt_path).exists():
        ckpt_path = str(Path(ckpt_dir) / "checkpoint.pt")

    if Path(ckpt_path).exists():
        model = load_uae_from_checkpoint(ckpt_path, device)
        print(f"[>] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[WARN] No checkpoint found at {ckpt_path}, evaluating un-trained model.")
        base_channels = 16 if args.smoke else 32
        bottleneck_dim = 64 if args.smoke else 256
        model = UniversalAE(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout=0.0).to(device)

    model.eval()
    results = defaultdict(lambda: defaultdict(lambda: {"ssim": [], "l1": []}))

    print(f"[>] Evaluating {len(test_manifest)} records on {device}...")
    with torch.no_grad():
        for rec in test_manifest:
            img_p = resolve_img_path(rec["image_path"])
            if not img_p.exists():
                clean_np = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                clean_np = np.load(str(img_p))
            corr_np = corrupt_from_record(clean_np, rec)

            clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            corr_t = torch.from_numpy(corr_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            out_t = model(corr_t)
            s_val = ssim(out_t, clean_t).item()
            l1_val = F.l1_loss(out_t, clean_t).item()

            ctype = rec["corruption_type"]
            slevel = rec.get("severity_level", "none")
            results[ctype][slevel]["ssim"].append(s_val)
            results[ctype][slevel]["l1"].append(l1_val)

    rows = []
    for ctype in CORRUPTION_TYPES:
        for slevel in (["none"] if ctype == "clean" else ["low", "medium", "high"]):
            data = results.get(ctype, {}).get(slevel, {})
            if data:
                rows.append({
                    "Corruption": ctype,
                    "Severity": slevel,
                    "SSIM": f"{np.mean(data['ssim']):.4f}",
                    "L1": f"{np.mean(data['l1']):.4f}",
                    "N": len(data["ssim"]),
                })

    df = pd.DataFrame(rows)
    print("\n--- Task 1 Evaluation Results ---")
    print(df.to_string(index=False))

    # Visual grids
    out_dir = Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = str(out_dir / "task1_reconstruction_grid.png")
    plot_visual_grid(model, test_manifest, n=min(12, len(test_manifest)), title="Task 1 - Universal AE Reconstructions", device=device, save_path=grid_path)

    # Failure cases
    failure_records = []
    for rec in test_manifest:
        if rec["corruption_type"] != "clean":
            img_p = resolve_img_path(rec["image_path"])
            clean_np = np.zeros((128, 128, 3), dtype=np.uint8) if not img_p.exists() else np.load(str(img_p))
            corr_np = corrupt_from_record(clean_np, rec)
            clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            corr_t = torch.from_numpy(corr_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                out_t = model(corr_t)
            s_val = ssim(out_t, clean_t).item()
            failure_records.append((s_val, rec))

    failure_records.sort(key=lambda x: x[0])
    worst_manifest = [r for _, r in failure_records[:4]]
    if worst_manifest:
        fail_path = str(out_dir / "task1_failure_cases.png")
        plot_visual_grid(model, worst_manifest, n=len(worst_manifest), title="Task 1 - Failure Cases (Lowest SSIM)", device=device, save_path=fail_path)

    print("[OK] Task 1 Evaluation Complete.")


if __name__ == "__main__":
    main()
