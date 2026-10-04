# Generative AI Image Restoration & Synthesis System 

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
