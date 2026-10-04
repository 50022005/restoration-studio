"""
src/task1/model.py
───────────────────────────────────────────────────────────────────────────────
Universal Denoising Autoencoder – Task 1.

Architecture: Conv encoder → FC bottleneck → FC → Conv decoder.
NO skip connections (disabled per spec).

Design justification:
  • 4 conv stages (stride-2) give 8× spatial downsampling (128 → 8 px).
  • The fully-connected bottleneck creates a genuine compressed latent;
    bottleneck_dim << spatial feature size forces the network to discard
    noise and learn a compact clean-image manifold.
    Reference: Vincent et al. (2010) "Stacked Denoising Autoencoders."
               JMLR 11.
  • BatchNorm + LeakyReLU encoder, BatchNorm + ReLU decoder (common in
    image-to-image networks – Isola et al., 2017 pix2pix paper).
  • ConvTranspose2d decoder: learnable upsampling preferred over bilinear
    for reconstruction tasks (Zeiler & Fergus 2014).
  • ONNX opset 17 compatible: no dynamic shapes, no Python control flow.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


# ── Encoder / Decoder blocks ──────────────────────────────────────────────────

class EncBlock(nn.Module):
    """Conv2d (stride 2) → BN → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock(nn.Module):
    """ConvTranspose2d (stride 2) → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, activation: str = "relu"):
        super().__init__()
        act: nn.Module = nn.ReLU(inplace=True) if activation == "relu" else nn.Tanh()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch) if activation == "relu" else nn.Identity(),
            act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Universal Autoencoder ─────────────────────────────────────────────────────

class UniversalAE(nn.Module):
    """
    Universal denoising autoencoder for 128×128 RGB images.

    Input shape:  (B, 3, 128, 128)   float32 in [0, 1]
    Output shape: (B, 3, 128, 128)   float32 in [0, 1]

    Spatial flow:
      Encoder   128 → 64 → 32 → 16 → 8
      Bottleneck 8*8*8C → bottleneck_dim → 8*8*8C
      Decoder   8 → 16 → 32 → 64 → 128

    Args:
        base_channels:   number of channels in the first encoder stage (C).
                         Subsequent stages use C, 2C, 4C, 8C.
        bottleneck_dim:  dimension of the compressed latent vector.
                         Must be < 8*8*8C = 512C to be a real bottleneck.
        dropout:         spatial dropout rate applied inside encoder blocks.
    """

    def __init__(
        self,
        base_channels: int = 32,
        bottleneck_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        C  = base_channels
        # Encoder: 4 stages, each halves spatial dims
        self.enc = nn.Sequential(
            EncBlock(3,    C,    dropout),   # (B, C,   64, 64)
            EncBlock(C,   C*2,  dropout),   # (B, 2C,  32, 32)
            EncBlock(C*2, C*4,  dropout),   # (B, 4C,  16, 16)
            EncBlock(C*4, C*8,  dropout),   # (B, 8C,   8,  8)
        )
        flat_dim = C * 8 * 8 * 8           # 512*C after 4 stages on 128px input

        # Bottleneck fully-connected layers
        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(bottleneck_dim, flat_dim),
            nn.ReLU(inplace=True),
        )

        self._C        = C
        self._flat_dim = flat_dim

        # Decoder: 4 stages, each doubles spatial dims
        self.dec = nn.Sequential(
            DecBlock(C*8, C*4),             # (B, 4C, 16, 16)
            DecBlock(C*4, C*2),             # (B, 2C, 32, 32)
            DecBlock(C*2, C),               # (B,  C, 64, 64)
            DecBlock(C,   3, activation="tanh"),  # (B,  3,128,128)
        )

        # tanh outputs in [-1,1]; we map to [0,1] in forward
        self._apply_weights_init()

    def _apply_weights_init(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the compressed latent vector z."""
        return self.fc_enc(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z → (B, 3, 128, 128) in [0,1]."""
        C = self._C
        x = self.fc_dec(z).view(-1, C * 8, 8, 8)
        x = self.dec(x)
        return (x + 1.0) * 0.5        # tanh [-1,1] → [0,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def build_universal_ae(
    base_channels: int = 32,
    bottleneck_dim: int = 256,
    dropout: float = 0.0,
) -> UniversalAE:
    """Factory used by Optuna and the export script."""
    return UniversalAE(base_channels=base_channels,
                       bottleneck_dim=bottleneck_dim,
                       dropout=dropout)
