"""
src/utils/tracking.py
───────────────────────────────────────────────────────────────────────────────
Thin W&B wrapper used across all tasks.

Choice rationale: W&B was chosen over MLflow because:
  1. Free hosted storage – no local DB to manage.
  2. Better Colab integration (one wandb.login() call).
  3. Superior image logging (wandb.Image) for visual grids.
  4. Optuna integration via wandb.log per trial.

Reference: Weights & Biases documentation
  https://docs.wandb.ai/ref/python
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb


def init_run(
    project: str,
    name: str,
    config: dict,
    tags: Optional[List[str]] = None,
    resume: str = "allow",
    id: Optional[str] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Initialise (or resume) a W&B run.

    Args:
        project: W&B project name.
        name:    human-readable run name.
        config:  hyperparameter dict logged to W&B.
        tags:    optional list of string tags.
        resume:  "allow" lets W&B resume if a run with the same id exists.
        id:      run id for resuming (e.g., from a checkpoint).

    Returns:
        wandb.Run object.
    """
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags or [],
        resume=resume,
        id=id,
    )
    return run


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log a dict of scalar metrics.  step=None → auto-increment."""
    wandb.log(metrics, step=step)


def log_image_grid(
    key: str,
    images: List[np.ndarray],
    captions: Optional[List[str]] = None,
    step: Optional[int] = None,
) -> None:
    """
    Log a list of uint8 HWC images as a W&B image panel.

    Args:
        key:      metric key (e.g., "val/reconstructions").
        images:   list of uint8 ndarrays (H, W, 3) or (H, W).
        captions: optional list of strings, same length as images.
        step:     global step for alignment with loss curves.
    """
    wb_images = [
        wandb.Image(img, caption=cap if captions else None)
        for img, cap in zip(images, captions or [""] * len(images))
    ]
    wandb.log({key: wb_images}, step=step)


def log_optuna_trial(
    trial_number: int,
    params: dict,
    value: float,
    step: Optional[int] = None,
) -> None:
    """Log one Optuna trial result as a W&B row (useful for HP search viz)."""
    wandb.log(
        {"optuna/trial": trial_number, "optuna/value": value, **params},
        step=step,
    )


def finish_run() -> None:
    """Finalise the current W&B run (call at end of each notebook section)."""
    if wandb.run is not None:
        wandb.finish()
