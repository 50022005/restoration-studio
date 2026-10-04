"""
tests/test_corruption.py
───────────────────────────────────────────────────────────────────────────────
Unit tests for src/utils/corruption.py.

Run locally:   pytest tests/test_corruption.py -v
Run on Colab:  !pytest tests/test_corruption.py -v   (after pip install pytest)

These tests prove:
  1. All corruption functions preserve shape and dtype.
  2. Salt-and-pepper is truly stochastic (different seeds → different pixels).
  3. Gaussian blur is deterministic (same params → identical output).
  4. Occlusion covers correct area fraction.
  5. corrupt_random samples all 4 types over many draws.
  6. corrupt_from_record exactly reproduces corrupt_random results.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.corruption import (
    CORRUPTION_TYPES,
    apply_gaussian_blur,
    apply_occlusion,
    apply_salt_pepper,
    corrupt_from_record,
    corrupt_random,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_image() -> np.ndarray:
    """128×128 RGB image filled with a mid-grey gradient."""
    rng = np.random.default_rng(0)
    return (rng.uniform(50, 200, (128, 128, 3))).astype(np.uint8)


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ── Shape / dtype preservation ────────────────────────────────────────────────

class TestShapePreservation:
    def test_salt_pepper_shape(self, clean_image, seeded_rng):
        out = apply_salt_pepper(clean_image, p=0.1, rng=seeded_rng)
        assert out.shape == clean_image.shape
        assert out.dtype == np.uint8

    def test_blur_shape(self, clean_image):
        for ks in [3, 5, 7]:
            out = apply_gaussian_blur(clean_image, ks, 1.5)
            assert out.shape == clean_image.shape, f"Failed at kernel={ks}"
            assert out.dtype == np.uint8

    def test_occlusion_shape(self, clean_image):
        rects = [(10, 10, 30, 30), (60, 60, 20, 20)]
        out   = apply_occlusion(clean_image, rects)
        assert out.shape == clean_image.shape
        assert out.dtype == np.uint8


# ── Correctness ───────────────────────────────────────────────────────────────

class TestCorrectnessProperties:
    def test_salt_pepper_only_extreme_values(self, clean_image, seeded_rng):
        """Corrupted pixels must be exactly 0 or 255."""
        out = apply_salt_pepper(clean_image, p=0.3, rng=seeded_rng)
        # Find pixels that changed
        changed = (out != clean_image).any(axis=2)
        assert changed.sum() > 0, "No pixels were corrupted"
        for pix in out[changed]:
            assert all(v in (0, 255) for v in pix), (
                f"Corrupted pixel has non-extreme value: {pix}"
            )

    def test_salt_pepper_different_seeds_differ(self, clean_image):
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        out_a = apply_salt_pepper(clean_image, p=0.1, rng=rng_a)
        out_b = apply_salt_pepper(clean_image, p=0.1, rng=rng_b)
        assert not np.array_equal(out_a, out_b), (
            "Different seeds must produce different noise patterns"
        )

    def test_blur_is_deterministic(self, clean_image):
        """Same params → identical output every time."""
        out_a = apply_gaussian_blur(clean_image, kernel_size=5, sigma=1.5)
        out_b = apply_gaussian_blur(clean_image, kernel_size=5, sigma=1.5)
        assert np.array_equal(out_a, out_b)

    def test_blur_smooths_image(self, clean_image):
        """Blurred image should have lower total variation than original."""
        def total_variation(img):
            return (np.abs(np.diff(img.astype(int), axis=0)).sum()
                    + np.abs(np.diff(img.astype(int), axis=1)).sum())
        out = apply_gaussian_blur(clean_image, 7, 2.5)
        assert total_variation(out) < total_variation(clean_image)

    def test_occlusion_area(self, clean_image):
        """Black pixels in occluded image should approximate target area."""
        H, W   = clean_image.shape[:2]
        rects  = [(0, 0, W // 4, H // 4)]                   # 6.25% target
        out    = apply_occlusion(clean_image, rects)
        zeroed = (out.sum(axis=2) == 0).sum()
        assert zeroed >= (H * W * 0.05), "Less area zeroed than expected"

    def test_clean_returned_unchanged(self, clean_image, seeded_rng):
        out, ctype, severity = corrupt_random(clean_image, rng=seeded_rng)
        # Run many draws to eventually hit 'clean'
        for seed_val in range(200):
            rng = np.random.default_rng(seed_val)
            out, ctype, _ = corrupt_random(clean_image, rng=rng)
            if ctype == "clean":
                assert np.array_equal(out, clean_image), (
                    "clean corruption must return an identical copy"
                )
                break


# ── Stochasticity / coverage ──────────────────────────────────────────────────

class TestCoverageAndStochasticity:
    def test_all_four_types_sampled(self, clean_image):
        """Over 400 random draws, all 4 corruption types must appear."""
        seen = set()
        for seed in range(400):
            rng = np.random.default_rng(seed)
            _, ctype, _ = corrupt_random(clean_image, rng=rng)
            seen.add(ctype)
        assert seen == set(CORRUPTION_TYPES), (
            f"Not all corruption types were sampled; only saw: {seen}"
        )

    def test_roughly_uniform_distribution(self, clean_image):
        """Each type should appear in roughly 25 ± 10 % of 1000 draws."""
        counts = {t: 0 for t in CORRUPTION_TYPES}
        N = 1000
        for seed in range(N):
            rng = np.random.default_rng(seed)
            _, ctype, _ = corrupt_random(clean_image, rng=rng)
            counts[ctype] += 1
        for ctype, count in counts.items():
            frac = count / N
            assert 0.15 < frac < 0.35, (
                f"Type {ctype!r} appeared {frac:.1%} of the time (expected ≈25%)"
            )


# ── Deterministic reproduction ────────────────────────────────────────────────

class TestManifestReproduction:
    def test_corrupt_from_record_matches_original(self, clean_image):
        """
        corrupt_from_record must reproduce the EXACT same pixels as the
        original corrupt_random call that created the record.
        """
        for seed in range(50):
            rng    = np.random.default_rng(seed)
            img_seed = int(rng.integers(0, 2**31))
            img_rng  = np.random.default_rng(img_seed)
            corrupted, ctype, severity = corrupt_random(clean_image, img_rng)

            record = {
                "corruption_type": ctype,
                "severity":        severity,
                "seed":            img_seed,
            }
            reproduced = corrupt_from_record(clean_image, record)
            assert np.array_equal(corrupted, reproduced), (
                f"Seed {img_seed}: corrupt_from_record did not reproduce the "
                f"original {ctype!r} corruption exactly."
            )
