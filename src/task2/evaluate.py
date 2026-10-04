"""
src/task2/evaluate.py
───────────────────────────────────────────────────────────────────────────────
Evaluation script for Task 2: Hard-Routed Restoration.

Compares:
  • Oracle routing (uses ground-truth manifest label to pick specialist)
  • Predicted routing (uses CorruptionClassifier argmax output; clean -> identity bypass)
  • Reports SSIM metrics per corruption type and lists top routing errors

Usage:
    python src/task2/evaluate.py --config configs/task2.yaml
    python src/task2/evaluate.py --smoke
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task1.model import UniversalAE
from src.task2.classifier import CorruptionClassifier
from src.utils.corruption import (
    CORRUPTION_TYPES,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    corrupt_from_record,
)
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


def hard_route_infer(corrupted_t, model_clf, model_salt, model_blur, model_occ, routing_label=None):
    with torch.no_grad():
        probs = model_clf.predict_proba(corrupted_t)

    if routing_label is not None:
        routed = routing_label
    else:
        routed = int(probs.argmax(dim=1).item())

    ctype = IDX_TO_LABEL[routed]
    with torch.no_grad():
        if ctype == "clean":
            x_hat = corrupted_t
        elif ctype == "salt_pepper":
            x_hat = model_salt(corrupted_t)
        elif ctype == "blur":
            x_hat = model_blur(corrupted_t)
        elif ctype == "occlusion":
            x_hat = model_occ(corrupted_t)
        else:
            x_hat = corrupted_t

    return x_hat, probs, routed


def load_uae_from_ckpt(ckpt_dir, device, default_bc=32, default_bn=256):
    p_best = Path(ckpt_dir) / "best.pt"
    p_ckpt = Path(ckpt_dir) / "checkpoint.pt"
    p_target = p_best if p_best.exists() else p_ckpt
    if p_target.exists():
        state = torch.load(p_target, map_location=device)
        state_dict = state["model"] if "model" in state else state
        bc = state_dict["enc.0.block.0.weight"].shape[0]
        bn = state_dict["fc_enc.1.weight"].shape[0]
        model = UniversalAE(base_channels=bc, bottleneck_dim=bn, dropout=0.0).to(device)
        model.load_state_dict(state_dict)
        print(f"[>] Loaded UAE from {p_target}")
    else:
        print(f"[WARN] No checkpoint at {p_target}; using untrained weights.")
        model = UniversalAE(base_channels=default_bc, bottleneck_dim=default_bn, dropout=0.0).to(device)
    model.eval()
    return model


def load_clf_from_ckpt(ckpt_dir, device, default_bc=32):
    p_best = Path(ckpt_dir) / "best.pt"
    p_ckpt = Path(ckpt_dir) / "checkpoint.pt"
    p_target = p_best if p_best.exists() else p_ckpt
    if p_target.exists():
        state = torch.load(p_target, map_location=device)
        state_dict = state["model"] if "model" in state else state
        bc = state_dict["features.0.0.weight"].shape[0]
        model = CorruptionClassifier(base_channels=bc, dropout=0.0).to(device)
        model.load_state_dict(state_dict)
        print(f"[>] Loaded Classifier from {p_target}")
    else:
        print(f"[WARN] No checkpoint at {p_target}; using untrained weights.")
        model = CorruptionClassifier(base_channels=default_bc, dropout=0.0).to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 2 Hard Routing Restoration")
    parser.add_argument("--config", default="configs/task2.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
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

    # Load Classifier
    clf_ckpt_dir = config.get("classifier", {}).get("training", {}).get("checkpoint_dir", "checkpoints/task2_clf")
    if args.smoke:
        clf_ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task2_clf")

    model_clf = load_clf_from_ckpt(clf_ckpt_dir, device, default_bc=16 if args.smoke else 32)

    # Load Specialists
    spec_base_dir = config.get("specialists", {}).get("training", {}).get("checkpoint_dir", "checkpoints/task2_specialists")
    if args.smoke:
        spec_base_dir = str(REPO_ROOT / "checkpoints_smoke" / "task2_specialists")

    default_bc = 16 if args.smoke else 32
    default_bn = 64 if args.smoke else 256

    model_salt = load_uae_from_ckpt(str(Path(spec_base_dir) / "salt_pepper"), device, default_bc, default_bn)
    model_blur = load_uae_from_ckpt(str(Path(spec_base_dir) / "blur"), device, default_bc, default_bn)
    model_occ = load_uae_from_ckpt(str(Path(spec_base_dir) / "occlusion"), device, default_bc, default_bn)

    oracle_ssim = defaultdict(list)
    predicted_ssim = defaultdict(list)
    routing_errors = []

    eval_manifest = test_manifest[: min(500, len(test_manifest))]
    print(f"[>] Evaluating Hard Routing on {len(eval_manifest)} test records...")

    for rec in eval_manifest:
        img_p = resolve_img_path(rec["image_path"])
        if not img_p.exists():
            clean_np = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            clean_np = np.load(str(img_p))
        corr_np = corrupt_from_record(clean_np, rec)

        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        corr_t = torch.from_numpy(corr_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        true_label = LABEL_TO_IDX[rec["corruption_type"]]

        # Oracle routing
        out_oracle, _, _ = hard_route_infer(corr_t, model_clf, model_salt, model_blur, model_occ, routing_label=true_label)
        s_oracle = ssim(out_oracle, clean_t).item()
        oracle_ssim[rec["corruption_type"]].append(s_oracle)

        # Predicted routing
        out_pred, probs, routed = hard_route_infer(corr_t, model_clf, model_salt, model_blur, model_occ)
        s_pred = ssim(out_pred, clean_t).item()
        predicted_ssim[rec["corruption_type"]].append(s_pred)

        if routed != true_label and rec["corruption_type"] != "clean":
            routing_errors.append({
                "true": CORRUPTION_TYPES[true_label],
                "predicted": CORRUPTION_TYPES[routed],
                "ssim_oracle": s_oracle,
                "ssim_predicted": s_pred,
                "image_path": rec["image_path"],
            })

    print("\n--- Routing Evaluation Results (SSIM) ---")
    print(f"{'Type':<15} {'Oracle':>10} {'Predicted':>12} {'Delta':>9}")
    for ctype in CORRUPTION_TYPES:
        o = np.mean(oracle_ssim[ctype]) if oracle_ssim[ctype] else 0.0
        p = np.mean(predicted_ssim[ctype]) if predicted_ssim[ctype] else 0.0
        print(f"  {ctype:<13} {o:>10.4f} {p:>12.4f} {p-o:>+9.4f}")

    total_non_clean = len([r for r in eval_manifest if r["corruption_type"] != "clean"])
    print(f"\nRouting Errors: {len(routing_errors)} / {total_non_clean}")

    if routing_errors:
        print("\nTop 5 Worst Routing Failures (Oracle SSIM - Predicted SSIM):")
        routing_errors.sort(key=lambda e: e["ssim_oracle"] - e["ssim_predicted"], reverse=True)
        for e in routing_errors[:5]:
            print(
                f"  {e['true']} -> {e['predicted']} | "
                f"oracle_ssim={e['ssim_oracle']:.4f} | pred_ssim={e['ssim_predicted']:.4f}"
            )

    print("[OK] Task 2 Routing Evaluation Complete.")


if __name__ == "__main__":
    main()
