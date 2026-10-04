"""
src/task2/export_onnx.py
───────────────────────────────────────────────────────────────────────────────
ONNX export and verification script for Task 2 models:
  • CorruptionClassifier
  • Salt Specialist AE
  • Blur Specialist AE
  • Occlusion Specialist AE

Exports models with dynamic batching (opset 17) and verifies parity.

Usage:
    python src/task2/export_onnx.py --config configs/task2.yaml
    python src/task2/export_onnx.py --smoke
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
from src.task2.classifier import CorruptionClassifier
from src.utils.training import load_checkpoint, set_seeds


def load_uae_from_checkpoint(ckpt_path: str, device: torch.device) -> UniversalAE:
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if "model" in state else state
    base_channels = state_dict["enc.0.block.0.weight"].shape[0]
    bottleneck_dim = state_dict["fc_enc.1.weight"].shape[0]
    model = UniversalAE(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    return model


def load_classifier_from_checkpoint(ckpt_path: str, device: torch.device) -> CorruptionClassifier:
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if "model" in state else state
    base_channels = state_dict["features.0.0.weight"].shape[0]
    model = CorruptionClassifier(base_channels=base_channels, dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    return model


def export_model_onnx(model, onnx_path, name, device, num_samples=4):
    model.eval()
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    print(f"[>] Exporting {name} -> {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[OK] {name} ONNX schema checked.")

    # Parity check
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    rng = np.random.default_rng(99)

    for i in range(num_samples):
        x_np = rng.random((1, 3, 128, 128)).astype(np.float32)
        pt_out = model(torch.from_numpy(x_np).to(device)).cpu().detach().numpy()
        ort_out = sess.run([oname], {iname: x_np})[0]
        max_diff = np.abs(pt_out - ort_out).max()
        assert np.allclose(pt_out, ort_out, atol=1e-4, rtol=1e-3), (
            f"Parity check failed for {name}, sample {i}, max_diff={max_diff:.2e}"
        )
    print(f"  [OK] {name} Parity Check PASSED (atol=1e-4)")


def main():
    parser = argparse.ArgumentParser(description="Export Task 2 Classifier and Specialist models to ONNX")
    parser.add_argument("--config", default="configs/task2.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on CPU")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seeds(42)
    device = torch.device("cpu")

    base_channels = 16 if args.smoke else 32
    bottleneck_dim = 64 if args.smoke else 256

    models_out_dir = Path("models_smoke" if args.smoke else config.get("specialists", {}).get("training", {}).get("onnx_dir", "models"))

    # 1. Classifier
    clf_ckpt_dir = config.get("classifier", {}).get("training", {}).get("checkpoint_dir", "checkpoints/task2_clf")
    if args.smoke:
        clf_ckpt_dir = str(REPO_ROOT / "checkpoints_smoke" / "task2_clf")
    p_clf = Path(clf_ckpt_dir) / "best.pt" if (Path(clf_ckpt_dir) / "best.pt").exists() else Path(clf_ckpt_dir) / "checkpoint.pt"
    if p_clf.exists():
        model_clf = load_classifier_from_checkpoint(str(p_clf), device)
    else:
        model_clf = CorruptionClassifier(base_channels=base_channels, dropout=0.0).to(device)
    export_model_onnx(model_clf, str(models_out_dir / "task2_classifier.onnx"), "Classifier", device)

    # 2. Specialists
    spec_base_dir = config.get("specialists", {}).get("training", {}).get("checkpoint_dir", "checkpoints/task2_specialists")
    if args.smoke:
        spec_base_dir = str(REPO_ROOT / "checkpoints_smoke" / "task2_specialists")

    for ctype in ["salt_pepper", "blur", "occlusion"]:
        name = f"{ctype.title()} Specialist"
        spec_ckpt_dir = Path(spec_base_dir) / ctype
        p_spec = spec_ckpt_dir / "best.pt" if (spec_ckpt_dir / "best.pt").exists() else spec_ckpt_dir / "checkpoint.pt"
        if p_spec.exists():
            model_spec = load_uae_from_checkpoint(str(p_spec), device)
        else:
            model_spec = UniversalAE(base_channels=base_channels, bottleneck_dim=bottleneck_dim, dropout=0.0).to(device)
        
        onnx_name = f"task2_{'salt' if ctype == 'salt_pepper' else ctype}.onnx"
        export_model_onnx(model_spec, str(models_out_dir / onnx_name), name, device)

    print("\n[OK] All Task 2 ONNX Exports & Parity Checks Completed Successfully.")


if __name__ == "__main__":
    main()
