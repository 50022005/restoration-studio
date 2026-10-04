"""
src/utils/corruption.py
───────────────────────────────────────────────────────────────────────────────
Canonical corruption functions shared across Tasks 1-3, the backend
corrupt.py router, and the test suite.

All ops are NumPy / PyTorch-based.  No OpenCV dependency (avoids Colab
version conflicts).  Gaussian blur uses a hand-built kernel via F.conv2d
which is ONNX-exportable.

Reference architecture:
  - Noise types follow  Hendrycks & Dietterich (2019)
    "Benchmarking Neural Network Robustness to Common Corruptions and
    Perturbations"  (arXiv:1903.12261).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────

CORRUPTION_TYPES: List[str] = ["clean", "salt_pepper", "blur", "occlusion"]
LABEL_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CORRUPTION_TYPES)}
IDX_TO_LABEL: Dict[int, str] = {i: c for c, i in LABEL_TO_IDX.items()}


# ── Low-level pixel operations ────────────────────────────────────────────────

def apply_salt_pepper(
    img: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Salt-and-pepper noise on a uint8 HWC RGB image.

    Pixels are independently selected with probability *p*; each selected
    pixel is set to 255 (salt) or 0 (pepper) with equal probability.

    Args:
        img: uint8 ndarray of shape (H, W, 3).
        p:   corruption probability in [0, 1].
        rng: seeded numpy Generator for reproducibility.

    Returns:
        Corrupted copy (same dtype and shape).
    """
    assert 0.0 <= p <= 1.0, f"p must be in [0,1], got {p}"
    img = img.copy()
    noise_mask = rng.random(img.shape[:2]) < p          # which pixels to corrupt
    salt_mask  = rng.random(img.shape[:2]) >= 0.5       # salt or pepper
    img[noise_mask & salt_mask]  = 255
    img[noise_mask & ~salt_mask] = 0
    return img


def apply_gaussian_blur(
    img: np.ndarray,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian blur via a hand-built separable kernel (F.conv2d).

    Using PyTorch rather than OpenCV avoids version conflicts in Colab
    and keeps the operation compatible with torch.onnx.export (opset 17).

    Reference: Gaussian kernel derivation in Bishop (2006) PRML §2.3.

    Args:
        img:         uint8 ndarray (H, W, 3).
        kernel_size: odd integer in {3, 5, 7}.
        sigma:       Gaussian standard deviation > 0.

    Returns:
        Blurred copy, uint8.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert sigma > 0, "sigma must be positive"

    # Build 1-D Gaussian, outer-product to 2-D
    half   = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    g1d    = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g1d    = g1d / g1d.sum()
    g2d    = torch.outer(g1d, g1d)                     # (k, k)
    kernel = g2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # (3,1,k,k)

    # (H,W,3) → (1,3,H,W) → conv → (H,W,3) uint8
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    pad = kernel_size // 2
    with torch.no_grad():
        blurred = F.conv2d(t, kernel, padding=pad, groups=3)
    out = (blurred.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
           ).astype(np.uint8)
    return out


def apply_occlusion(
    img: np.ndarray,
    rects: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Black rectangular occlusion masks.

    Args:
        img:   uint8 ndarray (H, W, 3).
        rects: list of (x, y, w, h) in pixel coordinates.

    Returns:
        Copy with black rectangles applied.
    """
    img = img.copy()
    for x, y, w, h in rects:
        y_end = min(y + h, img.shape[0])
        x_end = min(x + w, img.shape[1])
        img[y:y_end, x:x_end] = 0
    return img


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sample_rects(
    rng: np.random.Generator,
    H: int,
    W: int,
    num_rects: int,
    target_frac: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Sample *num_rects* non-overlapping-ish rectangles covering ~target_frac
    of the image area jointly.
    """
    per_frac = target_frac / num_rects
    rects: List[Tuple[int, int, int, int]] = []
    for _ in range(num_rects):
        area    = max(1, int(per_frac * H * W))
        rh      = max(1, int(rng.integers(max(1, area // W), max(2, H // 2) + 1)))
        rh      = min(rh, H)
        rw      = max(1, min(area // max(rh, 1), W))
        x_start = int(rng.integers(0, max(1, W - rw + 1)))
        y_start = int(rng.integers(0, max(1, H - rh + 1)))
        rects.append((x_start, y_start, rw, rh))
    return rects


def _severity_for(
    corruption: str,
    rng: np.random.Generator,
    H: int = 128,
    W: int = 128,
) -> dict:
    """Sample a random severity dict for the given corruption type."""
    if corruption == "clean":
        return {}
    if corruption == "salt_pepper":
        return {
            "p": float(rng.uniform(0.02, 0.15)),
            "noise_seed": int(rng.integers(0, 2 ** 31)),
        }
    if corruption == "blur":
        ks    = int(rng.choice([3, 5, 7]))
        sigma = float(rng.uniform(0.5, 2.5))
        return {"kernel_size": ks, "sigma": sigma}
    if corruption == "occlusion":
        num_rects   = int(rng.integers(1, 4))
        target_frac = float(rng.uniform(0.10, 0.35))
        rects       = _sample_rects(rng, H, W, num_rects, target_frac)
        return {"num_rects": num_rects, "target_frac": target_frac, "rects": rects}
    raise ValueError(f"Unknown corruption: {corruption!r}")


# ── Public training API ───────────────────────────────────────────────────────

def corrupt_random(
    img: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, str, dict]:
    """
    Apply one of the four corruption types with equal probability.

    Used *inside* the training data-loader.  A fresh corruption type and
    severity are sampled each time an image is loaded—corrupted copies are
    never saved to disk.

    Args:
        img: uint8 ndarray (H, W, 3).
        rng: optional seeded Generator; a fresh unseeded one is used if None.

    Returns:
        (corrupted_img, corruption_type_str, severity_dict)
    """
    if rng is None:
        rng = np.random.default_rng()
    corruption = str(rng.choice(CORRUPTION_TYPES))
    severity   = _severity_for(corruption, rng, img.shape[0], img.shape[1])
    corrupted  = corrupt_from_severity(img, corruption, severity, rng)
    return corrupted, corruption, severity


def corrupt_from_severity(
    img: np.ndarray,
    corruption: str,
    severity: dict,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply a specific corruption with a pre-defined severity dict."""
    if corruption == "clean":
        return img.copy()
    if corruption == "salt_pepper":
        if "noise_seed" in severity:
            sp_rng = np.random.default_rng(severity["noise_seed"])
        elif rng is not None:
            sp_rng = rng
        else:
            sp_rng = np.random.default_rng()
        return apply_salt_pepper(img, severity["p"], sp_rng)
    if corruption == "blur":
        return apply_gaussian_blur(img, severity["kernel_size"], severity["sigma"])
    if corruption == "occlusion":
        return apply_occlusion(img, [tuple(r) for r in severity["rects"]])
    raise ValueError(f"Unknown corruption: {corruption!r}")


def corrupt_from_record(img: np.ndarray, record: dict) -> np.ndarray:
    """
    Re-apply a corruption exactly as stored in a manifest record.

    The seed stored in the record is used so that stochastic operations
    (salt-and-pepper pixel selection) are fully deterministic.
    """
    ctype    = record["corruption_type"]
    severity = record["severity"]
    seed     = record["seed"]
    rng      = np.random.default_rng(seed)
    if ctype == "salt_pepper" and "noise_seed" not in severity:
        severity = dict(severity)
        severity["noise_seed"] = seed
    return corrupt_from_severity(img, ctype, severity, rng)


# ── Manifest generation ───────────────────────────────────────────────────────

def generate_val_manifest(
    image_paths: List[str],
    output_path: str,
    seed: int = 100,
    H: int = 128,
    W: int = 128,
) -> List[dict]:
    """
    Generate the validation corruption manifest **once** and save as JSON.

    Each image is assigned exactly one corruption type (equal probability)
    with a random severity.  The seed makes the whole manifest deterministic.

    Args:
        image_paths: ordered list of image file paths (relative or absolute).
        output_path: where to write the JSON manifest.
        seed:        master seed for the manifest.
        H, W:        image dimensions (for rect sampling).

    Returns:
        List of record dicts (same as the written JSON).
    """
    master_rng = np.random.default_rng(seed)
    records: List[dict] = []

    for idx, img_path in enumerate(image_paths):
        # Give each image its own independent sub-seed
        img_seed   = int(master_rng.integers(0, 2 ** 31))
        img_rng    = np.random.default_rng(img_seed)
        corruption = str(img_rng.choice(CORRUPTION_TYPES))
        severity   = _severity_for(corruption, img_rng, H, W)
        records.append({
            "index":          idx,
            "image_path":     str(img_path),
            "corruption_type": corruption,
            "severity":       severity,
            "seed":           img_seed,
        })

    _write_manifest(records, output_path)
    return records


def generate_test_manifest(
    image_paths: List[str],
    output_path: str,
    seed: int = 200,
    H: int = 128,
    W: int = 128,
) -> List[dict]:
    """
    Generate the test corruption manifest with **3 fixed severity levels**
    per non-clean corruption type (spec §"Final test severities").

    For each image this produces 10 records:
      • 1 clean
      • 3 salt-and-pepper (low/medium/high)
      • 3 blur (low/medium/high)
      • 3 occlusion (low/medium/high)

    Args:
        image_paths: ordered list of image file paths.
        output_path: where to write the JSON manifest.
        seed:        master seed; rect coordinates are derived from this.
        H, W:        image dimensions.

    Returns:
        List of record dicts.
    """
    master_rng = np.random.default_rng(seed)

    SALT_SEVERITIES = [
        ("low",    {"p": 0.03}),
        ("medium", {"p": 0.08}),
        ("high",   {"p": 0.15}),
    ]
    BLUR_SEVERITIES = [
        ("low",    {"kernel_size": 3, "sigma": 0.7}),
        ("medium", {"kernel_size": 5, "sigma": 1.5}),
        ("high",   {"kernel_size": 7, "sigma": 2.5}),
    ]
    OCC_CONFIGS = [
        ("low",    1, 0.10),
        ("medium", 2, 0.20),
        ("high",   3, 0.35),
    ]

    records: List[dict] = []

    def _rec(idx, path, ctype, level, sev, seed_val):
        return {
            "image_index":    idx,
            "image_path":     str(path),
            "corruption_type": ctype,
            "severity_level": level,
            "severity":       sev,
            "seed":           seed_val,
        }

    for img_idx, img_path in enumerate(image_paths):
        # Clean
        records.append(_rec(img_idx, img_path, "clean", "none", {}, 0))

        # Salt-and-pepper (severity dict is fully deterministic – no rng needed)
        for level, sev in SALT_SEVERITIES:
            img_seed = int(master_rng.integers(0, 2 ** 31))
            records.append(_rec(img_idx, img_path, "salt_pepper", level, sev, img_seed))

        # Blur (same – deterministic params)
        for level, sev in BLUR_SEVERITIES:
            img_seed = int(master_rng.integers(0, 2 ** 31))
            records.append(_rec(img_idx, img_path, "blur", level, sev, img_seed))

        # Occlusion – rect positions are random, so we store them
        for level, num_rects, target_frac in OCC_CONFIGS:
            img_seed = int(master_rng.integers(0, 2 ** 31))
            img_rng  = np.random.default_rng(img_seed)
            rects    = _sample_rects(img_rng, H, W, num_rects, target_frac)
            sev      = {"num_rects": num_rects, "target_frac": target_frac, "rects": rects}
            records.append(_rec(img_idx, img_path, "occlusion", level, sev, img_seed))

    _write_manifest(records, output_path)
    return records


def _write_manifest(records: List[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"[corruption] Manifest written → {output_path}  ({len(records)} records)")
