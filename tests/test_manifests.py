"""
tests/test_manifests.py
───────────────────────────────────────────────────────────────────────────────
Unit tests proving manifest determinism.

These tests satisfy spec §"unit tests proving determinism":
  • Re-running generate_val_manifest / generate_test_manifest with the same
    seed and paths produces bit-identical JSON every time.
  • corrupt_from_record on a manifest record reproduces the corruption
    pixel-for-pixel (SHA-256 hash check).
  • Test manifest contains exactly 10 records per image (1 clean +
    3 salt + 3 blur + 3 occlusion).
  • Val manifest contains exactly 1 record per image.

Run:  pytest tests/test_manifests.py -v
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.corruption import (
    corrupt_from_record,
    generate_test_manifest,
    generate_val_manifest,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_fake_paths(n: int) -> list:
    """Return n fake path strings (no real files needed for manifest tests)."""
    return [f"/data/pets/img_{i:04d}.npy" for i in range(n)]


# ── Val manifest ──────────────────────────────────────────────────────────────

class TestValManifest:
    def test_deterministic_across_calls(self):
        """Two calls with the same seed → identical JSON bytes."""
        paths = _make_fake_paths(20)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = f"{tmp}/val1.json"
            p2 = f"{tmp}/val2.json"
            generate_val_manifest(paths, p1, seed=100)
            generate_val_manifest(paths, p2, seed=100)
            h1 = _sha256(open(p1, "rb").read())
            h2 = _sha256(open(p2, "rb").read())
        assert h1 == h2, "Val manifest is not deterministic across two calls"

    def test_different_seeds_differ(self):
        paths = _make_fake_paths(10)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = f"{tmp}/val_a.json"
            p2 = f"{tmp}/val_b.json"
            generate_val_manifest(paths, p1, seed=100)
            generate_val_manifest(paths, p2, seed=999)
            h1 = _sha256(open(p1, "rb").read())
            h2 = _sha256(open(p2, "rb").read())
        assert h1 != h2, "Different seeds must produce different manifests"

    def test_one_record_per_image(self):
        paths = _make_fake_paths(30)
        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/val.json"
            records = generate_val_manifest(paths, p, seed=100)
        assert len(records) == len(paths)

    def test_valid_corruption_types(self):
        paths = _make_fake_paths(40)
        from src.utils.corruption import CORRUPTION_TYPES
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_val_manifest(paths, f"{tmp}/val.json", seed=42)
        for rec in records:
            assert rec["corruption_type"] in CORRUPTION_TYPES

    def test_all_four_types_present(self):
        """With 100 images all four corruption types should appear."""
        paths = _make_fake_paths(100)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_val_manifest(paths, f"{tmp}/val.json", seed=42)
        ctypes = {r["corruption_type"] for r in records}
        from src.utils.corruption import CORRUPTION_TYPES
        assert ctypes == set(CORRUPTION_TYPES)

    def test_required_fields_present(self):
        paths = _make_fake_paths(5)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_val_manifest(paths, f"{tmp}/val.json", seed=0)
        required = {"index", "image_path", "corruption_type", "severity", "seed"}
        for rec in records:
            assert required.issubset(rec.keys())


# ── Test manifest ─────────────────────────────────────────────────────────────

class TestTestManifest:
    def test_deterministic_across_calls(self):
        paths = _make_fake_paths(10)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = f"{tmp}/test1.json"
            p2 = f"{tmp}/test2.json"
            generate_test_manifest(paths, p1, seed=200)
            generate_test_manifest(paths, p2, seed=200)
            h1 = _sha256(open(p1, "rb").read())
            h2 = _sha256(open(p2, "rb").read())
        assert h1 == h2, "Test manifest is not deterministic across two calls"

    def test_ten_records_per_image(self):
        """1 clean + 3 salt + 3 blur + 3 occlusion = 10 per image."""
        N     = 5
        paths = _make_fake_paths(N)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_test_manifest(paths, f"{tmp}/test.json", seed=200)
        assert len(records) == N * 10

    def test_three_severity_levels_per_type(self):
        paths = _make_fake_paths(3)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_test_manifest(paths, f"{tmp}/test.json", seed=200)
        # For img_index 0
        per_type = {}
        for r in records:
            if r["image_index"] == 0:
                per_type.setdefault(r["corruption_type"], []).append(r["severity_level"])
        for ctype in ("salt_pepper", "blur", "occlusion"):
            levels = per_type.get(ctype, [])
            assert sorted(levels) == ["high", "low", "medium"], (
                f"{ctype} severity levels: {levels}"
            )

    def test_salt_pepper_probabilities_match_spec(self):
        paths = _make_fake_paths(2)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_test_manifest(paths, f"{tmp}/test.json", seed=200)
        sp_recs = [r for r in records
                   if r["corruption_type"] == "salt_pepper" and r["image_index"] == 0]
        p_values = sorted([r["severity"]["p"] for r in sp_recs])
        assert p_values == pytest.approx([0.03, 0.08, 0.15], abs=1e-9)

    def test_blur_configs_match_spec(self):
        paths = _make_fake_paths(2)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_test_manifest(paths, f"{tmp}/test.json", seed=200)
        blur_recs = [r for r in records
                     if r["corruption_type"] == "blur" and r["image_index"] == 0]
        configs = sorted([(r["severity"]["kernel_size"], r["severity"]["sigma"])
                          for r in blur_recs])
        assert configs == [(3, 0.7), (5, 1.5), (7, 2.5)]

    def test_occlusion_num_rects_match_spec(self):
        paths = _make_fake_paths(2)
        with tempfile.TemporaryDirectory() as tmp:
            records = generate_test_manifest(paths, f"{tmp}/test.json", seed=200)
        occ_recs = [r for r in records
                    if r["corruption_type"] == "occlusion" and r["image_index"] == 0]
        num_rects = sorted([r["severity"]["num_rects"] for r in occ_recs])
        assert num_rects == [1, 2, 3]


# ── Pixel-level reproduction ──────────────────────────────────────────────────

class TestPixelReproduction:
    def test_all_corruption_types_reproduce_exactly(self):
        """
        For each non-clean corruption in the val manifest, applying
        corrupt_from_record twice must give SHA-256-identical arrays.
        """
        img     = (np.random.default_rng(0).uniform(50, 200, (128, 128, 3))
                   ).astype(np.uint8)
        paths   = _make_fake_paths(20)

        with tempfile.TemporaryDirectory() as tmp:
            records = generate_val_manifest(paths, f"{tmp}/val.json", seed=42)

        for rec in records:
            a = corrupt_from_record(img, rec)
            b = corrupt_from_record(img, rec)
            assert _sha256(a.tobytes()) == _sha256(b.tobytes()), (
                f"Reproduction failed for {rec['corruption_type']!r} "
                f"record index {rec['index']}"
            )
