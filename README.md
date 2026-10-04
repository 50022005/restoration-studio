# Generative AI Image Restoration & Synthesis System (Milestone 1)

This repository contains the complete implementation for **Milestone 1: Foundations, Task 1 (Universal Restoration), and Task 2 (Hard-Routed Restoration)** for the Oxford-IIIT Pet dataset.

---

## 📁 Repository Structure

```
.
├── configs/
│   ├── data.yaml            # Data configuration & corruption specs
│   ├── task1.yaml           # Task 1 Universal Autoencoder config
│   ├── task2.yaml           # Task 2 Classifier & Specialist config
│   ├── task3.yaml           # Task 3 Soft MoE config
│   └── task4.yaml           # Task 4 Face-to-Sketch cGAN config
├── data/
│   ├── prepare_pets.py      # Download, 80/20 split (seed 42), cache 128x128 .npy
│   └── manifests/           # Deterministic val and test corruption manifests
│       ├── split.json
│       ├── val_manifest.json
│       └── test_manifest.json
├── src/
│   ├── task1/
│   │   ├── model.py         # UniversalAE (no skip connections)
│   │   ├── train.py         # Optuna + final training loop
│   │   ├── evaluate.py      # Per-corruption & per-severity evaluation + grids
│   │   └── export_onnx.py   # ONNX export & parity check
│   ├── task2/
│   │   ├── classifier.py    # CorruptionClassifier (4 classes)
│   │   ├── specialist.py    # Specialist autoencoder definition
│   │   ├── train_classifier.py   # Balanced-batch classifier train
│   │   ├── train_specialists.py  # Shared Optuna + 3 independent specialist trains
│   │   ├── evaluate.py      # Oracle vs predicted hard routing evaluation
│   │   └── export_onnx.py   # Export 4 ONNX models & parity check
│   └── utils/
│       ├── corruption.py    # Canonical runtime corruption functions
│       ├── datasets.py      # PetDataset & WeightedRandomSampler
│       ├── losses.py        # ReconstructionLoss (L1 + SSIM)
│       ├── tracking.py      # W&B experiment tracking wrapper
│       └── training.py      # Seeds, checkpointing, training utilities
├── scripts/
│   └── verify_onnx.py       # ONNX vs PyTorch output parity verification
├── tests/
│   ├── test_corruption.py   # Unit tests for corruption functions
│   └── test_manifests.py    # Unit tests for manifest determinism
├── notebooks/
│   └── M1_foundations_task1_task2.ipynb # Colab runner notebook
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart (Local Verification Mode)

To run all unit tests and verify scripts locally on CPU in `--smoke` mode:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pytest suite (25 unit tests)
python -m pytest -v

# 3. Task 1 Pipeline Smoke Test
python src/task1/train.py --smoke
python src/task1/evaluate.py --smoke
python src/task1/export_onnx.py --smoke

# 4. Task 2 Pipeline Smoke Test
python src/task2/train_classifier.py --smoke
python src/task2/train_specialists.py --smoke
python src/task2/evaluate.py --smoke
python src/task2/export_onnx.py --smoke
```

---

## 🚀 Google Colab Training Instructions

All full-scale training runs on a Google Colab free T4 GPU.

### 1. Private Repository Authentication (Colab Secrets)
1. In Colab, open the key icon on the left sidebar (**Secrets**).
2. Add a new secret named `GH_TOKEN` with your Personal Access Token (`repo` scope).
3. Enable "Notebook access" for `GH_TOKEN`.

### 2. Execution Order
Open `notebooks/M1_foundations_task1_task2.ipynb` in Colab and execute cells in order:

1. **Cell 1**: Install dependencies (`pip install -r requirements.txt`).
2. **Cell 2**: Mount Google Drive (`/content/drive`).
3. **Cell 3**: Prepare Oxford-IIIT Pet dataset & manifests (`python data/prepare_pets.py`).
4. **Cell 4**: Train Task 1 Universal Autoencoder (`python src/task1/train.py --config configs/task1.yaml`).
5. **Cell 5**: Evaluate Task 1 model (`python src/task1/evaluate.py --config configs/task1.yaml`).
6. **Cell 6**: Export Task 1 ONNX model (`python src/task1/export_onnx.py --config configs/task1.yaml`).
7. **Cell 7**: Train Task 2 Corruption Classifier (`python src/task2/train_classifier.py --config configs/task2.yaml`).
8. **Cell 8**: Train Task 2 Specialist Autoencoders (`python src/task2/train_specialists.py --config configs/task2.yaml`).
9. **Cell 9**: Evaluate Task 2 Hard Routing (`python src/task2/evaluate.py --config configs/task2.yaml`).
10. **Cell 10**: Export Task 2 ONNX models (`python src/task2/export_onnx.py --config configs/task2.yaml`).

### 3. Expected Runtimes & Artifact Locations
- **Data prep**: ~8 min (run once; cached on Drive).
- **Task 1 Optuna + Train**: ~3.5 h (Optuna) + ~25 min (30 epochs).
- **Task 2 Classifier**: ~1.5 h (Optuna) + ~20 min (20 epochs).
- **Task 2 Specialists**: ~1.5 h (Optuna) + ~3 h (3 x 20 epochs).
- **Artifacts on Drive** (`/content/drive/MyDrive/genai_assignment/`):
  - Checkpoints: `checkpoints/task1/`, `checkpoints/task2_clf/`, `checkpoints/task2_specialists/`
  - Optuna DBs: `optuna/task1.db`, `optuna/task2_clf.db`, `optuna/task2_specialist.db`
  - ONNX Models: `models/task1_universal.onnx`, `models/task2_classifier.onnx`, `models/task2_*.onnx`

### 4. Handling Disconnections
- All Optuna studies persist to SQLite on Google Drive with `load_if_exists=True`. Resuming automatically continues unfinished studies.
- All models save `checkpoint.pt` at every epoch. Running the script again automatically resumes from the last completed epoch.
