"""
src/utils/training.py
───────────────────────────────────────────────────────────────────────────────
Shared training-loop utilities used across all four tasks.

Provides:
  • run_epoch()          – one train or val epoch with AMP support
  • save_checkpoint()    – save model + optimiser + epoch to Drive
  • load_checkpoint()    – resume from last saved checkpoint
  • set_seeds()          – fix all RNG seeds for reproducibility
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """Fix Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    optimiser: Optional[Optimizer],
    device: torch.device,
    scaler: Optional[GradScaler],
    clip_grad_norm: float = 1.0,
    training: bool = True,
    extra_metrics: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Run one full pass over *loader*.

    Args:
        model:          PyTorch Module.
        loader:         DataLoader.
        criterion:      callable(output, target) → scalar loss.
        optimiser:      Optimizer; None for evaluation mode.
        device:         torch.device.
        scaler:         GradScaler for AMP; None to disable AMP.
        clip_grad_norm: gradient clipping max norm (0 = disabled).
        training:       if False, model.eval() and no grad computation.
        extra_metrics:  optional callable(outputs, targets) → dict[str,float].

    Returns:
        dict with at least {"loss": mean_loss} and any extra metric keys.
    """
    if training:
        model.train()
    else:
        model.eval()

    total_loss  = 0.0
    n_batches   = 0
    agg_metrics: Dict[str, float] = {}

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            # Unpack – corrupted, clean, label (or just corrupted, clean)
            if len(batch) == 3:
                corrupted, clean, labels = batch
                corrupted = corrupted.to(device)
                clean     = clean.to(device)
                labels    = labels.to(device)
                inputs, targets = corrupted, clean
            else:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                labels = None

            if training:
                optimiser.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets)
                if training:
                    scaler.scale(loss).backward()
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimiser)
                    scaler.update()
            else:
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                if training:
                    loss.backward()
                    if clip_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

            if extra_metrics is not None:
                m = extra_metrics(outputs.detach(), targets.detach()
                                  if labels is None else labels.detach())
                for k, v in m.items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v

    mean_loss = total_loss / max(n_batches, 1)
    result    = {"loss": mean_loss}
    for k, v in agg_metrics.items():
        result[k] = v / max(n_batches, 1)
    return result


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimiser: Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    """
    Save model + optimiser state to *checkpoint_dir*.

    Saves:
      • checkpoint_dir/checkpoint.pt  – always overwritten (resume point)
      • checkpoint_dir/best.pt        – only when is_best=True

    Returns path to the saved file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":       epoch,
        "model":       model.state_dict(),
        "optimiser":   optimiser.state_dict(),
        "metrics":     metrics,
    }
    path = str(ckpt_dir / filename)
    torch.save(state, path)
    if is_best:
        best_path = str(ckpt_dir / "best.pt")
        torch.save(state, best_path)
        print(f"[checkpoint] Best model saved → {best_path}")
    return path


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    optimiser: Optional[Optimizer] = None,
    filename: str = "checkpoint.pt",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Optional[Optimizer], int, dict]:
    """
    Resume from *checkpoint_dir/filename*.

    Returns:
        (model, optimiser_or_None, start_epoch, metrics_dict)
    """
    path = Path(checkpoint_dir) / filename
    if not path.exists():
        print(f"[checkpoint] No checkpoint at {path}; starting fresh.")
        return model, optimiser, 0, {}

    map_location = device if device is not None else "cpu"
    state        = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state["model"])
    if optimiser is not None and "optimiser" in state:
        optimiser.load_state_dict(state["optimiser"])
    start_epoch = state.get("epoch", 0) + 1
    metrics     = state.get("metrics", {})
    print(f"[checkpoint] Resumed from epoch {state['epoch']}  ({path})")
    return model, optimiser, start_epoch, metrics
