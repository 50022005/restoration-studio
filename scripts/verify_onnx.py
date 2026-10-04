"""
scripts/verify_onnx.py
───────────────────────────────────────────────────────────────────────────────
Verify that an ONNX model's outputs match the corresponding PyTorch model
within a numerical tolerance.

Spec requirement: "a script verifying ONNX output matches PyTorch within
tolerance."

Usage:
    python scripts/verify_onnx.py \
        --onnx  /content/drive/.../models/task1_universal.onnx \
        --model task1 \
        --ckpt  /content/drive/.../checkpoints/task1/best.pt \
        --n_samples 16 \
        --atol 1e-4

Exit code 0 = all checks passed.  Non-zero = failure.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_pytorch_model(model_name: str, ckpt_path: str) -> torch.nn.Module:
    """Load a PyTorch model from checkpoint."""
    if model_name == "task1":
        from src.task1.model import UniversalAE
        state  = torch.load(ckpt_path, map_location="cpu")
        config = state.get("config", {})
        model  = UniversalAE(
            base_channels=config.get("base_channels", 32),
            bottleneck_dim=config.get("bottleneck_dim", 256),
            dropout=0.0,            # eval mode – no dropout
        )
        model.load_state_dict(state["model"])

    elif model_name in ("task2_classifier",):
        from src.task2.classifier import CorruptionClassifier
        state  = torch.load(ckpt_path, map_location="cpu")
        config = state.get("config", {})
        model  = CorruptionClassifier(
            base_channels=config.get("base_channels", 32),
            dropout=0.0,
        )
        model.load_state_dict(state["model"])

    elif model_name in ("task2_salt", "task2_blur", "task2_occlusion"):
        from src.task2.specialist import SpecialistAE
        state  = torch.load(ckpt_path, map_location="cpu")
        config = state.get("config", {})
        model  = SpecialistAE(
            base_channels=config.get("base_channels", 32),
            bottleneck_dim=config.get("bottleneck_dim", 256),
            dropout=0.0,
        )
        model.load_state_dict(state["model"])

    else:
        raise ValueError(
            f"Unknown model name: {model_name!r}. "
            f"Choices: task1, task2_classifier, task2_salt, task2_blur, task2_occlusion"
        )

    model.eval()
    return model


# ── Verification ──────────────────────────────────────────────────────────────

def verify(onnx_path: str, model_name: str, ckpt_path: str,
           n_samples: int = 16, atol: float = 1e-4, rtol: float = 1e-3) -> bool:
    """
    Run *n_samples* random inputs through both PyTorch and ONNXRuntime,
    then compare outputs.

    Returns True if all checks pass; raises AssertionError otherwise.
    """
    print(f"[verify] Loading PyTorch model ({model_name}) from {ckpt_path}")
    pt_model = _load_pytorch_model(model_name, ckpt_path)

    print(f"[verify] Loading ONNX model from {onnx_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3       # suppress INFO spam
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    rng    = np.random.default_rng(0)
    passed = 0
    failed = 0

    for i in range(n_samples):
        x_np  = rng.uniform(0.0, 1.0, (1, 3, 128, 128)).astype(np.float32)
        x_pt  = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_out  = pt_model(x_pt).numpy()

        ort_out = sess.run([output_name], {input_name: x_np})[0]

        max_diff = np.abs(pt_out - ort_out).max()
        ok       = np.allclose(pt_out, ort_out, atol=atol, rtol=rtol)
        status   = "✓" if ok else "✗"
        print(f"  [{status}] sample {i:2d}  max_diff={max_diff:.2e}  "
              f"atol={atol:.0e}  rtol={rtol:.0e}")

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n[verify] {passed}/{n_samples} passed  (atol={atol}, rtol={rtol})")
    if failed > 0:
        raise AssertionError(
            f"{failed} sample(s) exceeded tolerance. "
            f"Check ONNX export opset or model changes."
        )
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify ONNX ↔ PyTorch output parity")
    parser.add_argument("--onnx",      required=True, help="Path to .onnx file")
    parser.add_argument("--model",     required=True,
                        choices=["task1", "task2_classifier",
                                 "task2_salt", "task2_blur", "task2_occlusion"],
                        help="Which model architecture to load")
    parser.add_argument("--ckpt",      required=True, help="Path to PyTorch checkpoint .pt")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--atol",      type=float, default=1e-4)
    parser.add_argument("--rtol",      type=float, default=1e-3)
    args = parser.parse_args()

    try:
        verify(args.onnx, args.model, args.ckpt,
               n_samples=args.n_samples, atol=args.atol, rtol=args.rtol)
        print("\n✓  ONNX verification PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗  ONNX verification FAILED: {e}")
        sys.exit(1)
