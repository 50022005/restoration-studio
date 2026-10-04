"""
src/task2/train_specialists.py
───────────────────────────────────────────────────────────────────────────────
Training script for Task 2 Specialist Autoencoders.

Workflow:
  1. Performs one shared Optuna architecture search on mixed corruptions
  2. Trains 3 specialist autoencoders independently (salt_pepper, blur, occlusion)
     on their respective corruption types
  3. Saves checkpoints to checkpoints/task2_specialists/<corruption_type>/

Usage:
    python src/task2/train_specialists.py --config configs/task2.yaml
    python src/task2/train_specialists.py --smoke
"""

import argparse
import json
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
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task1.model import UniversalAE
from src.utils.corruption import (
    LABEL_TO_IDX,
    _severity_for,
    corrupt_from_severity,
)
from src.utils.datasets import PetDataset
from src.utils.losses import ReconstructionLoss, ssim
from src.utils.training import load_checkpoint, save_checkpoint, set_seeds


def get_data_splits(config: dict, smoke: bool = False) -> Tuple[List[str], List[str], List[dict]]:
    manifest_dir = REPO_ROOT / "data" / "manifests"
    split_path = manifest_dir / "split.json"
    val_path = manifest_dir / "val_manifest.json"

    if smoke and (not split_path.exists() or not val_path.exists()):
        from src.task1.train import ensure_dummy_data_for_smoke
        return ensure_dummy_data_for_smoke(REPO_ROOT)

    if not split_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Manifests missing at {manifest_dir}")

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


class SingleCorrDataset(Dataset):
    """Dataset producing only one specific corruption type."""

    def __init__(self, npy_paths: List[str], ctype: str):
        self.paths = npy_paths
        self.ctype = ctype

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img = np.load(self.paths[idx])
        rng = np.random.default_rng()
        sev = _severity_for(self.ctype, rng)
        cor = corrupt_from_severity(img, self.ctype, sev, rng)
        c_t = torch.from_numpy(cor).permute(2, 0, 1).float() / 255.0
        x_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return c_t, x_t, LABEL_TO_IDX[self.ctype]


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
    parser = argparse.ArgumentParser(description="Train Task 2 Specialist Autoencoders")
    parser.add_argument("--config", default="configs/task2.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna study")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu" if args.smoke or not torch.cuda.is_available() else "cuda")
    set_seeds(42)

    train_npy, val_npy, val_manifest = get_data_splits(config, smoke=args.smoke)
    spec_cfg = config.get("specialists", {})

    optuna_cfg = spec_cfg.get("shared_optuna", {})
    storage_uri = optuna_cfg.get("storage", "sqlite:///optuna/task2_specialist.db")
    if args.smoke:
        storage_uri = f"sqlite:///{REPO_ROOT}/optuna_smoke_t2_spec.db"

    if storage_uri.startswith("sqlite:///"):
        db_path = Path(storage_uri.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = 2 if args.smoke else optuna_cfg.get("n_trials", 15)
    trial_epochs = 1 if args.smoke else optuna_cfg.get("trial_epochs", 5)

    best_spec = {
        "lr": 1e-3,
        "batch_size": 4 if args.smoke else 32,
        "bottleneck_dim": 64 if args.smoke else 256,
        "base_channels": 16 if args.smoke else 32,
        "alpha": 0.8,
    }

    if not args.no_optuna:
        class SpecSearchDataset(PetDataset):
            def __getitem__(self, idx):
                c, x, l = super().__getitem__(idx)
                if l == 0:
                    rng = np.random.default_rng(idx + 10000)
                    img = np.load(self.cache_paths[idx])
                    nc = str(rng.choice(["salt_pepper", "blur", "occlusion"]))
                    sev = _severity_for(nc, rng)
                    cim = corrupt_from_severity(img, nc, sev, rng)
                    c = torch.from_numpy(cim).permute(2, 0, 1).float() / 255.0
                    l = LABEL_TO_IDX[nc]
                return c, x, l

        def objective(trial: optuna.Trial):
            if args.smoke:
                lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
                batch_size = 4
                bottleneck_dim = 64
                base_channels = 16
                alpha = 0.8
            else:
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
                bottleneck_dim = trial.suggest_int("bottleneck_dim", 64, 512, step=64)
                base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
                alpha = trial.suggest_float("alpha", 0.50, 0.95, step=0.05)

            model = UniversalAE(base_channels, bottleneck_dim, 0.0).to(device)
            criterion = ReconstructionLoss(alpha)
            optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = GradScaler() if spec_cfg.get("training", {}).get("amp", True) and device.type == "cuda" else None

            ds = SpecSearchDataset(train_npy)
            ld = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            val_ld = DataLoader(PetDataset(val_npy, manifest=val_manifest), batch_size=batch_size, shuffle=False)

            best_ssim = 0.0
            for epoch in range(trial_epochs):
                train_one_epoch(model, ld, criterion, optimiser, scaler, device, use_amp=scaler is not None)
                _, val_ssim_score = val_one_epoch(model, val_ld, criterion, device, use_amp=scaler is not None)
                trial.report(val_ssim_score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                best_ssim = max(best_ssim, val_ssim_score)
            return best_ssim

        study = optuna.create_study(
            study_name=optuna_cfg.get("study_name", "task2_specialist_arch"),
            direction="maximize",
            storage=storage_uri,
            load_if_exists=True,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=2 if args.smoke else 4, n_warmup_steps=1 if args.smoke else 2),
        )

        print(f"[>] Shared Specialist Optuna Study: {study.study_name} | {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=not args.smoke)
        best_spec = study.best_params
        print(f"[OK] Best Specialist Architecture SSIM: {study.best_value:.4f}")

    # ── Train 3 Specialists Independently ───────────────────────────────────
    max_epochs = 1 if args.smoke else spec_cfg.get("training", {}).get("max_epochs", 20)
    corruption_types = spec_cfg.get("corruption_types", ["salt_pepper", "blur", "occlusion"])

    base_ckpt_dir = spec_cfg.get("training", {}).get("checkpoint_dir", "checkpoints/task2_specialists")
    if args.smoke:
        base_ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task2_specialists")

    for ctype in corruption_types:
        print(f"\n[>] Training Specialist: {ctype}")
        ckpt_dir = str(Path(base_ckpt_dir) / ctype)
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

        model = UniversalAE(
            base_channels=best_spec.get("base_channels", 32),
            bottleneck_dim=best_spec.get("bottleneck_dim", 256),
            dropout=0.0,
        ).to(device)

        criterion = ReconstructionLoss(best_spec.get("alpha", 0.8))
        optimiser = torch.optim.AdamW(model.parameters(), lr=best_spec.get("lr", 1e-3), weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs)
        scaler = GradScaler() if spec_cfg.get("training", {}).get("amp", True) and device.type == "cuda" else None

        model, optimiser, start_epoch, prev_metrics = load_checkpoint(model, ckpt_dir, optimiser=optimiser, device=device)
        best_ssim = prev_metrics.get("val_ssim", 0.0)

        train_ld = DataLoader(
            SingleCorrDataset(train_npy, ctype),
            batch_size=best_spec.get("batch_size", 32),
            shuffle=True,
            drop_last=True,
        )
        val_ld = DataLoader(
            PetDataset(val_npy, manifest=val_manifest),
            batch_size=best_spec.get("batch_size", 32),
            shuffle=False,
        )

        for epoch in range(start_epoch, max_epochs):
            tr_loss = train_one_epoch(model, train_ld, criterion, optimiser, scaler, device, use_amp=scaler is not None)
            val_loss, val_ssim_score = val_one_epoch(model, val_ld, criterion, device, use_amp=scaler is not None)
            scheduler.step()

            is_best = val_ssim_score > best_ssim
            if is_best:
                best_ssim = val_ssim_score

            save_checkpoint(
                model=model,
                optimiser=optimiser,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_ssim": val_ssim_score},
                checkpoint_dir=ckpt_dir,
                is_best=is_best,
            )

            print(
                f"  [{ctype}] Epoch {epoch+1:2d}/{max_epochs:2d} | "
                f"tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_ssim={val_ssim_score:.4f} {'[* BEST]' if is_best else ''}"
            )

        print(f"[OK] {ctype} Specialist complete. Best SSIM: {best_ssim:.4f}")

    print("\n[OK] All 3 Specialist Autoencoders Trained.")


if __name__ == "__main__":
    main()
