"""
src/task1/export_onnx.py
───────────────────────────────────────────────────────────────────────────────
ONNX export and verification script for Task 1: Universal Autoencoder.

Exports PyTorch model to ONNX (opset 17) with dynamic batch dimension,
and verifies numerical parity against PyTorch output (atol <= 1e-4).

Usage:
    python src/task1/export_onnx.py --config configs/task1.yaml
    python src/task1/export_onnx.py --smoke
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task1.model import UniversalAE
from src.utils.training import load_checkpoint, set_seeds


def load_uae_from_checkpoint(ckpt_path: str, device: torch.device) -> UniversalAE:
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if "model" in state else state
    base_channels = state_dict["enc.0.block.0.weight"].shape[0]
    bottleneck_dim = state_dict["fc_enc.1.weight"].shape[0]
    model = UniversalAE(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description="Export Task 1 Universal Autoencoder to ONNX")
    parser.add_argument("--config", default="configs/task1.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint .pt")
    parser.add_argument("--output", default=None, help="Path to output ONNX file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seeds(42)
    device = torch.device("cpu")

    ckpt_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints/task1")
    if args.smoke:
        ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task1")

    ckpt_path = args.checkpoint or str(Path(ckpt_dir) / "best.pt")
    if not Path(ckpt_path).exists():
        ckpt_path = str(Path(ckpt_dir) / "checkpoint.pt")

    onnx_out_path = args.output or config.get("training", {}).get("onnx_path", "models/task1_universal.onnx")
    if args.smoke:
        onnx_out_path = str(REPO_ROOT / "models_smoke" / "task1_universal.onnx")

    Path(onnx_out_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(ckpt_path).exists():
        model = load_uae_from_checkpoint(ckpt_path, device)
        print(f"[>] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_path}, exporting initialized model weights.")
        base_channels = 16 if args.smoke else 32
        bottleneck_dim = 64 if args.smoke else 256
        model = UniversalAE(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout=0.0).to(device)

    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    print(f"[>] Exporting ONNX model to {onnx_out_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out_path,
        input_names=["corrupted_image"],
        output_names=["restored_image"],
        opset_version=17,
        dynamic_axes={"corrupted_image": {0: "batch"}, "restored_image": {0: "batch"}},
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_out_path)
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX export successful & valid schema: {onnx_out_path}")

    # Parity verification
    print("[>] Running numerical parity check (PyTorch vs ONNX Runtime)...")
    sess = ort.InferenceSession(onnx_out_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    rng = np.random.default_rng(42)
    all_pass = True
    num_samples = 2 if args.smoke else 16

    for i in range(num_samples):
        x_np = rng.random((1, 3, 128, 128)).astype(np.float32)
        pt_out = model(torch.from_numpy(x_np).to(device)).cpu().detach().numpy()
        ort_out = sess.run([oname], {iname: x_np})[0]
        max_diff = np.abs(pt_out - ort_out).max()
        is_ok = np.allclose(pt_out, ort_out, atol=1e-4, rtol=1e-3)
        if not is_ok:
            print(f"  [FAIL] Sample {i}: max_diff={max_diff:.2e}")
            all_pass = False

    if all_pass:
        print("[OK] ONNX parity check PASSED (all samples within atol=1e-4)")
    else:
        raise ValueError("ONNX parity check failed!")


if __name__ == "__main__":
    main()
