"""
src/task2/specialist.py
───────────────────────────────────────────────────────────────────────────────
Specialist denoising autoencoders (Task 2).

Each specialist uses the *same architecture* as UniversalAE but has
independently trained parameters and is trained ONLY on its own corruption
type (salt, blur, or occlusion).

Design justification: specialist models can overfit to the statistics of
a single corruption type, achieving lower reconstruction error on that type
than a universal model – the trade-off between specialisation and
generalisation is investigated in the oracle vs. predicted routing
evaluation (spec §Task 2).

This module simply re-exports UniversalAE under the SpecialistAE name and
provides a factory that accepts the architecture config from Optuna.
"""

from src.task1.model import UniversalAE


# Re-use the identical architecture – independently initialised per specialist
SpecialistAE = UniversalAE


def build_specialist(
    corruption_type: str,
    base_channels: int = 32,
    bottleneck_dim: int = 256,
    dropout: float = 0.0,
) -> SpecialistAE:
    """
    Factory for one specialist AE.

    Args:
        corruption_type: one of "salt_pepper", "blur", "occlusion".
        base_channels:   first-stage channel count (from shared Optuna search).
        bottleneck_dim:  bottleneck size (from shared Optuna search).
        dropout:         spatial dropout rate.

    Returns:
        Freshly initialised SpecialistAE (same arch as UniversalAE).
    """
    if corruption_type not in ("salt_pepper", "blur", "occlusion"):
        raise ValueError(
            f"Specialist corruption_type must be one of "
            f"'salt_pepper', 'blur', 'occlusion'; got {corruption_type!r}"
        )
    return SpecialistAE(
        base_channels=base_channels,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
    )
