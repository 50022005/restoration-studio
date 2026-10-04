"""
src/task2/classifier.py
───────────────────────────────────────────────────────────────────────────────
CNN Corruption Classifier (Task 2).

Predicts one of 4 classes: clean, salt_pepper, blur, occlusion.
Architecture: VGG-style conv stack + GlobalAvgPool + FC.

Design justification:
  • GlobalAveragePooling avoids large FC layers and gives spatial invariance
    – useful because corruption location can vary (random occlusion).
    Reference: Lin et al. (2013) "Network in Network." arXiv:1312.4400.
  • WeightedRandomSampler (in datasets.py) handles class balance;
    BatchNorm reduces covariate shift between corruption types.
  • 4-stage design keeps the model small (~1 M params at base_channels=32)
    for fast Colab iteration.
"""

from typing import List

import torch
import torch.nn as nn


class CorruptionClassifier(nn.Module):
    """
    4-class corruption classifier.

    Input:  (B, 3, 128, 128)  float32 in [0,1]
    Output: (B, 4)            raw logits (use CrossEntropyLoss, not softmax)

    Args:
        base_channels: number of channels in stage 1 (doubles each stage).
        dropout:       dropout rate before the final linear layer.
    """

    def __init__(self, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        C = base_channels

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),                        # halve spatial dims
            )

        self.features = nn.Sequential(
            conv_block(3,    C),     # 128 → 64
            conv_block(C,   C*2),   # 64  → 32
            conv_block(C*2, C*4),   # 32  → 16
            conv_block(C*4, C*8),   # 16  →  8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, 8C, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(C * 8, 4),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (B, 4)."""
        return torch.softmax(self.forward(x), dim=1)
