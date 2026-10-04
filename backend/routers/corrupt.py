"""
backend/routers/corrupt.py
───────────────────────────────────────────────────────────────────────────────
FastAPI router: /corrupt endpoint.

Lets the frontend user pick a corruption type and severity level and
receive the corrupted image as a base64-encoded JPEG.

This uses the SAME corruption functions as the training pipeline
(src/utils/corruption) – no code duplication.

Endpoints
─────────
POST /corrupt/apply
    Body: multipart/form-data
      • file:            image upload (JPEG / PNG / WEBP)
      • corruption_type: "clean" | "salt_pepper" | "blur" | "occlusion"
      • severity_level:  "low" | "medium" | "high" | "custom"
      • custom_params:   JSON string (only used when severity_level="custom")

GET /corrupt/types
    Returns the list of available corruption types and their severity presets.
"""

import base64
import io
import json
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Import canonical corruption functions
from src.utils.corruption import (
    CORRUPTION_TYPES,
    apply_gaussian_blur,
    apply_occlusion,
    apply_salt_pepper,
    corrupt_from_severity,
)

router = APIRouter(prefix="/corrupt", tags=["corruption"])

# ── Severity presets (spec §"Final test severities") ─────────────────────────

PRESETS = {
    "salt_pepper": {
        "low":    {"p": 0.03},
        "medium": {"p": 0.08},
        "high":   {"p": 0.15},
    },
    "blur": {
        "low":    {"kernel_size": 3, "sigma": 0.7},
        "medium": {"kernel_size": 5, "sigma": 1.5},
        "high":   {"kernel_size": 7, "sigma": 2.5},
    },
    "occlusion": {
        "low": {
            "num_rects": 1, "target_frac": 0.10,
            "rects": [[8, 8, 40, 40]],
        },
        "medium": {
            "num_rects": 2, "target_frac": 0.20,
            "rects": [[8, 8, 40, 40], [70, 70, 35, 35]],
        },
        "high": {
            "num_rects": 3, "target_frac": 0.35,
            "rects": [[5, 5, 38, 38], [50, 50, 35, 35], [85, 10, 30, 45]],
        },
    },
    "clean": {"none": {}},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image_as_array(upload: UploadFile) -> np.ndarray:
    """Decode uploaded file → uint8 (128,128,3) ndarray."""
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if upload.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {upload.content_type}. "
                   f"Accepted: {allowed}",
        )
    data = upload.file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    img  = img.resize((128, 128), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _array_to_b64_jpeg(arr: np.ndarray) -> str:
    """uint8 HWC → base64-encoded JPEG string."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/types")
def list_corruption_types():
    """Return available corruption types and their severity presets."""
    return {
        "corruption_types": CORRUPTION_TYPES,
        "presets": {k: list(v.keys()) for k, v in PRESETS.items()},
    }


@router.post("/apply")
async def apply_corruption(
    file: UploadFile = File(...),
    corruption_type: str = Form(...),
    severity_level: str = Form("medium"),
    custom_params: Optional[str] = Form(None),
    seed: int = Form(42),
):
    """
    Apply a selected corruption to an uploaded image.

    Returns JSON with:
      • corrupted_b64:    base64 JPEG of the corrupted image
      • original_b64:     base64 JPEG of the 128×128 resized original
      • corruption_type:  echoed back
      • severity_level:   echoed back
      • severity_params:  dict of the actual severity parameters used
      • inference_ms:     time taken for the corruption operation
    """
    if corruption_type not in CORRUPTION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown corruption_type {corruption_type!r}. "
                   f"Must be one of {CORRUPTION_TYPES}.",
        )

    # Resolve severity parameters
    if custom_params:
        try:
            severity = json.loads(custom_params)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid custom_params JSON: {e}"
            )
    else:
        type_presets = PRESETS.get(corruption_type, {})
        if severity_level not in type_presets:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown severity_level {severity_level!r} for "
                       f"{corruption_type!r}. Available: {list(type_presets.keys())}",
            )
        severity = type_presets[severity_level]

    # Load and corrupt image
    img = _load_image_as_array(file)
    rng = np.random.default_rng(seed)

    t0       = time.perf_counter()
    corrupted = corrupt_from_severity(img, corruption_type, severity, rng)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return JSONResponse({
        "corrupted_b64":   _array_to_b64_jpeg(corrupted),
        "original_b64":    _array_to_b64_jpeg(img),
        "corruption_type": corruption_type,
        "severity_level":  severity_level,
        "severity_params": severity,
        "inference_ms":    round(elapsed_ms, 2),
    })
