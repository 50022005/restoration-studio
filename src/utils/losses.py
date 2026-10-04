"""
src/utils/losses.py
───────────────────────────────────────────────────────────────────────────────
Loss functions used across Tasks 1-3.

Combined L1 + SSIM loss
  Reference: Zhao et al. (2017) "Loss Functions for Image Restoration
  with Neural Networks", IEEE Trans. Comp. Imaging.

SSIM implementation follows Wang et al. (2004) TMMI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SSIM helper ───────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """1-D Gaussian kernel, normalised."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute the pixel-wise SSIM map between *x* and *y*.

    Both tensors should be in [0, 1] with shape (B, C, H, W).
    """
    B, C, H, W = x.shape
    k1d   = _gaussian_kernel_1d(window_size, sigma).to(x.device)
    k2d   = torch.outer(k1d, k1d)                                 # (w, w)
    kernel = k2d.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)    # (C,1,w,w)
    pad   = window_size // 2

    mu_x  = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y  = F.conv2d(y, kernel, padding=pad, groups=C)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_xx = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_xx
    sig_yy = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_yy
    sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
    return num / (denom + 1e-8)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Mean SSIM over a batch of images.

    Args:
        x, y: float tensors in [0, 1], shape (B, C, H, W).

    Returns:
        Scalar tensor in [-1, 1]  (typically in [0, 1] for natural images).
    """
    return _ssim_map(x, y, window_size=window_size).mean()


# ── Combined reconstruction loss ──────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """
    L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))

    *alpha* is a learnable-schedule hyperparameter tuned by Optuna (Task 1/2).
    Default alpha=0.8 gives ~4× weight to pixel accuracy vs. structure.

    Both inputs must be in [0, 1].
    """

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1]; got {alpha}")
        self.alpha = alpha

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        l1   = F.l1_loss(x_hat, x)
        ssim_val = ssim(x_hat, x)
        return self.alpha * l1 + (1.0 - self.alpha) * (1.0 - ssim_val)


# ── Balance loss (Task 3) ─────────────────────────────────────────────────────

def balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Routing-balance regulariser for the soft MoE gate (Task 3).

    L_bal = sum_k (mean_k(w_k) - 1/K)^2

    Encourages each branch to receive roughly equal average weight across
    a balanced batch.

    Reference: Shazeer et al. (2017) "Outrageously Large Neural Networks:
    The Sparsely-Gated Mixture-of-Experts Layer."  arXiv:1701.06538 §4.

    Args:
        weights: (B, K) softmax routing weights, K = number of branches.

    Returns:
        Scalar loss.
    """
    K           = weights.shape[1]
    mean_per_k  = weights.mean(dim=0)          # (K,)
    target      = torch.full_like(mean_per_k, 1.0 / K)
    return ((mean_per_k - target) ** 2).sum()
