"""
src/task2/train_classifier.py
───────────────────────────────────────────────────────────────────────────────
Training script for Task 2: 4-class Corruption Classifier.

Features:
  • Enforces class-balanced sampling via WeightedRandomSampler
  • Optuna hyperparameter tuning (learning rate, batch size, channels, dropout, weight decay)
  • Resumable training checkpoints
  • Evaluation: classification report + normalized confusion matrix plot

Usage:
    python src/task2/train_classifier.py --config configs/task2.yaml
    python src/task2/train_classifier.py --smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task2.classifier import CorruptionClassifier
from src.utils.corruption import CORRUPTION_TYPES, LABEL_TO_IDX
from src.utils.datasets import PetDataset, make_balanced_sampler
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


def train_clf_epoch(model, loader, optimiser, scaler, device, use_amp=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for corrupted, _, labels in loader:
        corrupted, labels = corrupted.to(device), labels.to(device)
        optimiser.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast("cuda"):
                logits = model(corrupted)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            logits = model(corrupted)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    n_batches = max(len(loader), 1)
    acc = correct / max(total, 1)
    return total_loss / n_batches, acc


def val_clf_epoch(model, loader, device, use_amp=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for corrupted, _, labels in loader:
            corrupted, labels = corrupted.to(device), labels.to(device)
            if use_amp and device.type == "cuda":
                with autocast("cuda"):
                    logits = model(corrupted)
            else:
                logits = model(corrupted)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Task 2 Corruption Classifier")
    parser.add_argument("--config", default="configs/task2.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
    parser.add_argument("--no-optuna", action="store_true", help="Skip Optuna study")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu" if args.smoke or not torch.cuda.is_available() else "cuda")
    set_seeds(42)
    print(f"[>] Task 2 Classifier Training | Device: {device} | Smoke: {args.smoke}")

    train_npy, val_npy, val_manifest = get_data_splits(config, smoke=args.smoke)
    clf_cfg = config.get("classifier", {})

    optuna_cfg = clf_cfg.get("optuna", {})
    storage_uri = optuna_cfg.get("storage", "sqlite:///optuna/task2_clf.db")
    if args.smoke:
        storage_uri = f"sqlite:///{REPO_ROOT}/optuna_smoke_t2_clf.db"

    if storage_uri.startswith("sqlite:///"):
        db_path = Path(storage_uri.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = 2 if args.smoke else optuna_cfg.get("n_trials", 15)
    trial_epochs = 1 if args.smoke else optuna_cfg.get("trial_epochs", 5)

    best_params = {
        "lr": 1e-3,
        "batch_size": 4 if args.smoke else 32,
        "base_channels": 16 if args.smoke else 32,
        "dropout": 0.2,
        "weight_decay": 1e-4,
    }

    if not args.no_optuna:
        def objective(trial: optuna.Trial):
            if args.smoke:
                lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
                batch_size = 4
                base_channels = 16
                dropout = 0.1
                weight_decay = 1e-4
            else:
                lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
                batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
                base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
                dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

            model = CorruptionClassifier(base_channels, dropout).to(device)
            optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scaler = GradScaler("cuda") if clf_cfg.get("training", {}).get("amp", True) and device.type == "cuda" else None

            train_ds = PetDataset(train_npy, deterministic=True, seed=0)
            sample_labels = [train_ds[i][2] for i in range(len(train_ds))]
            sampler = make_balanced_sampler(sample_labels)

            train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
            val_ld = DataLoader(PetDataset(val_npy, manifest=val_manifest), batch_size=batch_size, shuffle=False)

            best_acc = 0.0
            for epoch in range(trial_epochs):
                train_clf_epoch(model, train_ld, optimiser, scaler, device, use_amp=scaler is not None)
                val_acc = val_clf_epoch(model, val_ld, device, use_amp=scaler is not None)
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                best_acc = max(best_acc, val_acc)
            return best_acc

        study = optuna.create_study(
            study_name=optuna_cfg.get("study_name", "task2_classifier"),
            direction="maximize",
            storage=storage_uri,
            load_if_exists=True,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=2 if args.smoke else 4, n_warmup_steps=1 if args.smoke else 2),
        )
        print(f"[>] Classifier Optuna Study: {study.study_name} | {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=not args.smoke)
        best_params = study.best_params
        print(f"[OK] Best Optuna Classifier Acc: {study.best_value:.4f}")

    # ── Final Classifier Training ────────────────────────────────────────────
    max_epochs = 1 if args.smoke else clf_cfg.get("training", {}).get("max_epochs", 20)
    batch_size = best_params.get("batch_size", 32)
    base_channels = best_params.get("base_channels", 32)
    dropout = best_params.get("dropout", 0.3)
    lr = best_params.get("lr", 1e-3)
    weight_decay = best_params.get("weight_decay", 1e-4)

    model = CorruptionClassifier(base_channels, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs)
    scaler = GradScaler("cuda") if clf_cfg.get("training", {}).get("amp", True) and device.type == "cuda" else None

    ckpt_dir = clf_cfg.get("training", {}).get("checkpoint_dir", "checkpoints/task2_clf")
    if args.smoke:
        ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task2_clf")

    model, optimiser, start_epoch, prev_metrics = load_checkpoint(model, ckpt_dir, optimiser=optimiser, device=device)
    best_acc = prev_metrics.get("val_acc", 0.0)

    train_ds = PetDataset(train_npy, deterministic=True, seed=0)
    sample_labels = [train_ds[i][2] for i in range(len(train_ds))]
    sampler = make_balanced_sampler(sample_labels)

    train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_ld = DataLoader(PetDataset(val_npy, manifest=val_manifest), batch_size=batch_size, shuffle=False)

    print(f"[>] Final Classifier Training epochs {start_epoch} to {max_epochs}")
    for epoch in range(start_epoch, max_epochs):
        tr_loss, tr_acc = train_clf_epoch(model, train_ld, optimiser, scaler, device, use_amp=scaler is not None)
        val_acc = val_clf_epoch(model, val_ld, device, use_amp=scaler is not None)
        scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint(
            model=model,
            optimiser=optimiser,
            epoch=epoch,
            metrics={"val_loss": tr_loss, "val_acc": val_acc},
            checkpoint_dir=ckpt_dir,
            is_best=is_best,
        )

        print(
            f"  Epoch {epoch+1:2d}/{max_epochs:2d} | "
            f"tr_loss={tr_loss:.4f} | tr_acc={tr_acc:.3f} | "
            f"val_acc={val_acc:.3f} {'[* BEST]' if is_best else ''}"
        )

    # Confusion matrix evaluation
    model.eval()
    test_manifest_path = REPO_ROOT / "data" / "manifests" / "test_manifest.json"
    if test_manifest_path.exists():
        with open(test_manifest_path) as f:
            test_manifest = json.load(f)
        if args.smoke:
            test_manifest = test_manifest[:10]
        test_ld = DataLoader(PetDataset(val_npy, manifest=test_manifest), batch_size=batch_size, shuffle=False)
        all_preds, all_targets = [], []
        with torch.no_grad():
            for corrupted, _, labels in test_ld:
                logits = model(corrupted.to(device))
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(labels.numpy())

        print("\n--- Classifier Test Metrics ---")
        print(classification_report(all_targets, all_preds, target_names=CORRUPTION_TYPES, digits=4, zero_division=0))

        cm = confusion_matrix(all_targets, all_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CORRUPTION_TYPES, yticklabels=CORRUPTION_TYPES, ax=ax)
        ax.set_title("Task 2 - Normalised Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        cm_path = str(Path(ckpt_dir) / "task2_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"[OK] Saved confusion matrix -> {cm_path}")

    print("[OK] Classifier Training Complete.")


if __name__ == "__main__":
    main()
