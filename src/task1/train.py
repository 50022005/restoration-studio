"""
src/task1/train.py
───────────────────────────────────────────────────────────────────────────────
Training script for Task 1: Universal Denoising Autoencoder.

Supports:
  • Config-driven hyperparameter specification (--config configs/task1.yaml)
  • Optuna hyperparameter optimization with SQLite storage & MedianPruner
  • Smoke test mode (--smoke): CPU, tiny subset, 1 epoch, 2 Optuna trials
  • Resumable training checkpoints (checkpoint.pt / best.pt)
  • W&B experiment tracking

Usage:
    python src/task1/train.py --config configs/task1.yaml
    python src/task1/train.py --smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task1.model import UniversalAE
from src.utils.corruption import (
    generate_test_manifest,
    generate_val_manifest,
)
from src.utils.datasets import PetDataset
from src.utils.losses import ReconstructionLoss, ssim
from src.utils.tracking import finish_run, init_run, log_metrics, log_optuna_trial
from src.utils.training import load_checkpoint, save_checkpoint, set_seeds


def ensure_dummy_data_for_smoke(base_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    """Creates a tiny dummy dataset and manifests for smoke testing if real data is missing."""
    cache_dir = base_dir / "data" / "pets_cache_smoke"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = REPO_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_paths = []
    val_paths = []
    rng = np.random.default_rng(42)

    for i in range(8):
        p = cache_dir / f"train_{i}.npy"
        if not p.exists():
            np.save(str(p), rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
        train_paths.append(str(p))

    for i in range(4):
        p = cache_dir / f"val_{i}.npy"
        if not p.exists():
            np.save(str(p), rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
        val_paths.append(str(p))

    val_manifest = generate_val_manifest(val_paths, str(manifest_dir / "val_manifest.json"), seed=100)
    generate_test_manifest(val_paths, str(manifest_dir / "test_manifest.json"), seed=200)

    split_data = {"seed": 42, "train": train_paths, "val": val_paths, "test": val_paths}
    with open(manifest_dir / "split.json", "w") as f:
        json.dump(split_data, f, indent=2)

    return train_paths, val_paths, val_manifest


def get_data_splits(config: dict, smoke: bool = False) -> Tuple[List[str], List[str], List[dict]]:
    """Load cached dataset split paths and validation manifest."""
    manifest_dir = REPO_ROOT / "data" / "manifests"
    split_path = manifest_dir / "split.json"
    val_path = manifest_dir / "val_manifest.json"

    if smoke and (not split_path.exists() or not val_path.exists()):
        return ensure_dummy_data_for_smoke(REPO_ROOT)

    if not split_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset manifests missing at {manifest_dir}. "
            "Please run 'python data/prepare_pets.py' first."
        )

    with open(split_path) as f:
        split = json.load(f)
    with open(val_path) as f:
        val_manifest = json.load(f)

    train_npy = split["train"]
    val_npy = split["val"]

    if smoke:
        train_npy = train_npy[:8]
        val_npy = val_npy[:4]
        val_manifest = val_manifest[:4]

    return train_npy, val_npy, val_manifest


def train_one_epoch(model, loader, criterion, optimiser, scaler, device, use_amp=True):
    model.train()
    total_loss = 0.0
    for corrupted, clean, _ in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        optimiser.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with autocast():
                out = model(corrupted)
                loss = criterion(out, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            out = model(corrupted)
            loss = criterion(out, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def val_one_epoch(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for corrupted, clean, _ in loader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            if use_amp and device.type == "cuda":
                with autocast():
                    out = model(corrupted)
                    loss = criterion(out, clean)
            else:
                out = model(corrupted)
                loss = criterion(out, clean)
            total_loss += loss.item()
            total_ssim += ssim(out, clean).item()
    n = max(len(loader), 1)
    return total_loss / n, total_ssim / n


def main():
    parser = argparse.ArgumentParser(description="Train Task 1 Universal Autoencoder")
    parser.add_argument("--config", default="configs/task1.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna study and use defaults")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seeds(config.get("optuna", {}).get("sampler", {}).get("seed", 42))
    device = torch.device("cpu" if args.smoke or not torch.cuda.is_available() else "cuda")
    print(f"[>] Task 1 Training | Device: {device} | Smoke: {args.smoke}")

    train_npy, val_npy, val_manifest = get_data_splits(config, smoke=args.smoke)

    # ── Optuna Hyperparameter Optimization ──────────────────────────────────
    optuna_cfg = config.get("optuna", {})
    storage_uri = optuna_cfg.get("storage", "sqlite:///optuna/task1.db")
    if args.smoke:
        storage_uri = f"sqlite:///{REPO_ROOT}/optuna_smoke_t1.db"
    
    # Ensure directory exists for sqlite database
    if storage_uri.startswith("sqlite:///"):
        db_path = Path(storage_uri.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = 2 if args.smoke else optuna_cfg.get("n_trials", 20)
    trial_epochs = 1 if args.smoke else optuna_cfg.get("trial_epochs", 5)

    best_params = {
        "lr": 1e-3,
        "batch_size": 4 if args.smoke else 32,
        "bottleneck_dim": 128,
        "base_channels": 16 if args.smoke else 32,
        "dropout": 0.0,
        "alpha": 0.8,
    }

    if not args.no_optuna:
        def objective(trial: optuna.Trial):
            if args.smoke:
                lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
                batch_size = 4
                bottleneck_dim = 64
                base_channels = 16
                dropout = 0.0
                alpha = 0.8
            else:
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
                bottleneck_dim = trial.suggest_int("bottleneck_dim", 64, 512, step=64)
                base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
                dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)
                alpha = trial.suggest_float("alpha", 0.50, 0.95, step=0.05)

            model = UniversalAE(base_channels, bottleneck_dim, dropout).to(device)
            criterion = ReconstructionLoss(alpha)
            optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = GradScaler() if config.get("training", {}).get("amp", True) and device.type == "cuda" else None

            train_ds = PetDataset(train_npy)
            val_ds = PetDataset(val_npy, manifest=val_manifest)
            train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            best_val_ssim = 0.0
            for epoch in range(trial_epochs):
                train_one_epoch(model, train_ld, criterion, optimiser, scaler, device, use_amp=scaler is not None)
                _, val_ssim_score = val_one_epoch(model, val_ld, criterion, device, use_amp=scaler is not None)
                trial.report(val_ssim_score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                best_val_ssim = max(best_val_ssim, val_ssim_score)
            return best_val_ssim

        sampler = TPESampler(seed=optuna_cfg.get("sampler", {}).get("seed", 42))
        pruner = MedianPruner(n_startup_trials=2 if args.smoke else 5, n_warmup_steps=1 if args.smoke else 2)
        study = optuna.create_study(
            study_name=optuna_cfg.get("study_name", "task1_universal_ae"),
            direction="maximize",
            storage=storage_uri,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

        print(f"[>] Optuna Study: {study.study_name} | Running {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=not args.smoke)
        best_params = study.best_params
        print(f"[OK] Best Trial Value (SSIM): {study.best_value:.4f}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    # ── Final Training with Best Config ──────────────────────────────────────
    max_epochs = 1 if args.smoke else config.get("training", {}).get("max_epochs", 30)
    batch_size = best_params.get("batch_size", 32)
    base_channels = best_params.get("base_channels", 32)
    bottleneck_dim = best_params.get("bottleneck_dim", 256)
    dropout = best_params.get("dropout", 0.0)
    alpha = best_params.get("alpha", 0.8)
    lr = best_params.get("lr", 1e-3)

    model = UniversalAE(base_channels, bottleneck_dim, dropout).to(device)
    criterion = ReconstructionLoss(alpha)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.get("training", {}).get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs)
    scaler = GradScaler() if config.get("training", {}).get("amp", True) and device.type == "cuda" else None

    ckpt_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints/task1")
    if args.smoke:
        ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task1")

    model, optimiser, start_epoch, prev_metrics = load_checkpoint(
        model, ckpt_dir, optimiser=optimiser, device=device
    )
    best_ssim_score = prev_metrics.get("val_ssim", 0.0)

    train_ld = DataLoader(
        PetDataset(train_npy),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_ld = DataLoader(
        PetDataset(val_npy, manifest=val_manifest),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"[>] Starting Final Training from epoch {start_epoch} to {max_epochs}")
    for epoch in range(start_epoch, max_epochs):
        train_loss = train_one_epoch(model, train_ld, criterion, optimiser, scaler, device, use_amp=scaler is not None)
        val_loss, val_ssim_score = val_one_epoch(model, val_ld, criterion, device, use_amp=scaler is not None)
        scheduler.step()

        is_best = val_ssim_score > best_ssim_score
        if is_best:
            best_ssim_score = val_ssim_score

        save_checkpoint(
            model=model,
            optimiser=optimiser,
            epoch=epoch,
            metrics={"val_loss": val_loss, "val_ssim": val_ssim_score},
            checkpoint_dir=ckpt_dir,
            is_best=is_best,
        )

        print(
            f"  Epoch {epoch+1:2d}/{max_epochs:2d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ssim={val_ssim_score:.4f} {'[* BEST]' if is_best else ''}"
        )

    print(f"[OK] Task 1 Training Completed. Best Val SSIM: {best_ssim_score:.4f}")


if __name__ == "__main__":
    main()
