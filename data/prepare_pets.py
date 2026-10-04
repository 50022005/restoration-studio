"""
data/prepare_pets.py
───────────────────────────────────────────────────────────────────────────────
Download Oxford-IIIT Pet dataset, build the 80/20 train/val split (seed=42),
resize all images to 128×128 RGB, cache as .npy files on Google Drive, and
generate deterministic val + test corruption manifests.

Run this script ONCE per Colab session (it is idempotent – already-cached
files are skipped).

Expected runtime: ~8 min on Colab (download ~800 MB + resize 7,390 images).

Usage (in Colab cell):
    !python data/prepare_pets.py \
        --raw_dir /content/drive/MyDrive/genai_assignment/data/pets_raw \
        --cache_dir /content/drive/MyDrive/genai_assignment/data/pets_cache \
        --manifest_dir data/manifests \
        --seed 42
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_URL      = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
IMAGE_SIZE      = 128


# ── Download helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str) -> Path:
    """Download *url* to *dest* with a progress bar.  Skips if exists."""
    if dest.exists():
        print(f"[skip] {desc} already downloaded → {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {desc}")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
              desc=desc) as pbar:
        def _hook(block, block_size, total_size):
            if total_size > 0:
                pbar.total = total_size
            pbar.update(block * block_size - pbar.n)
        urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
    return dest


def _extract(tarpath: Path, dest: Path) -> None:
    """Extract tar.gz if destination directory does not yet exist."""
    if dest.exists():
        print(f"[skip] Already extracted → {dest}")
        return
    print(f"[extract] {tarpath.name} → {dest}")
    with tarfile.open(str(tarpath)) as tar:
        tar.extractall(str(dest.parent))
    print(f"[extract] Done.")


# ── Image collection ──────────────────────────────────────────────────────────

def _collect_trainval_images(images_dir: Path, annot_dir: Path) -> List[str]:
    """
    Return the list of image paths for the official trainval split.

    The annotations/trainval.txt file contains lines like:
        Abyssinian_1 1 1 1
    where the first token is the image stem (no extension).
    """
    tv_file = annot_dir / "trainval.txt"
    if not tv_file.exists():
        raise FileNotFoundError(
            f"trainval.txt not found at {tv_file}. "
            f"Check that the annotations tar was extracted correctly."
        )
    stems: List[str] = []
    with open(tv_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stem = line.split()[0]
            stems.append(stem)

    paths: List[str] = []
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                paths.append(str(p))
                break
        else:
            print(f"[warn] Image not found for stem: {stem!r}")
    print(f"[collect] Found {len(paths)} trainval images.")
    return paths


def _collect_test_images(images_dir: Path, annot_dir: Path) -> List[str]:
    """Return image paths for the official test split."""
    test_file = annot_dir / "test.txt"
    if not test_file.exists():
        raise FileNotFoundError(f"test.txt not found at {test_file}")
    stems: List[str] = []
    with open(test_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stem = line.split()[0]
            stems.append(stem)
    paths: List[str] = []
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                paths.append(str(p))
                break
    print(f"[collect] Found {len(paths)} test images.")
    return paths


# ── Caching ───────────────────────────────────────────────────────────────────

def cache_images(image_paths: List[str], cache_dir: Path) -> List[str]:
    """Resize to 128×128 RGB, save as .npy.  Returns .npy paths."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_paths: List[str] = []
    for p in tqdm(image_paths, desc="Caching images"):
        stem     = Path(p).stem
        out_path = cache_dir / f"{stem}.npy"
        if not out_path.exists():
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                np.save(str(out_path), np.array(img, dtype=np.uint8))
            except Exception as e:
                print(f"[warn] Skipping {p}: {e}")
                continue
        npy_paths.append(str(out_path))
    print(f"[cache] {len(npy_paths)} images in {cache_dir}")
    return npy_paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    raw_dir      = Path(args.raw_dir)
    cache_dir    = Path(args.cache_dir)
    manifest_dir = Path(args.manifest_dir)
    seed         = args.seed

    # 1. Download
    img_tar  = _download(IMAGES_URL,      raw_dir / "images.tar.gz",      "Images tar")
    ann_tar  = _download(ANNOTATIONS_URL, raw_dir / "annotations.tar.gz", "Annotations tar")

    # 2. Extract
    _extract(img_tar,  raw_dir / "images")
    _extract(ann_tar,  raw_dir / "annotations")

    images_dir = raw_dir / "images"
    annot_dir  = raw_dir / "annotations"

    # 3. Collect trainval / test paths
    tv_paths   = _collect_trainval_images(images_dir, annot_dir)
    test_paths = _collect_test_images(images_dir, annot_dir)

    # 4. 80/20 split (seed=42)
    train_paths, val_paths = train_test_split(
        tv_paths, test_size=0.20, random_state=seed, shuffle=True
    )
    print(f"[split] train={len(train_paths)}  val={len(val_paths)}  "
          f"test={len(test_paths)}  seed={seed}")

    # 5. Cache ALL splits
    train_npy = cache_images(train_paths, cache_dir / "train")
    val_npy   = cache_images(val_paths,   cache_dir / "val")
    test_npy  = cache_images(test_paths,  cache_dir / "test")

    # 6. Save split index (so every task uses identical paths)
    split = {
        "seed":       seed,
        "train":      train_npy,
        "val":        val_npy,
        "test":       test_npy,
    }
    split_path = manifest_dir / "split.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as fh:
        json.dump(split, fh, indent=2)
    print(f"[split] Saved → {split_path}")

    # 7. Generate val and test corruption manifests
    from src.utils.corruption import generate_test_manifest, generate_val_manifest

    # Use .npy paths in manifests (dataset loads from cache, not raw)
    val_manifest  = generate_val_manifest(
        val_npy,  str(manifest_dir / "val_manifest.json"),  seed=100
    )
    test_manifest = generate_test_manifest(
        test_npy, str(manifest_dir / "test_manifest.json"), seed=200
    )

    print("\n✓ Data preparation complete!")
    print(f"  Train:         {len(train_npy)} images")
    print(f"  Val:           {len(val_npy)}   images  →  {len(val_manifest)} manifest records")
    print(f"  Test:          {len(test_npy)}  images  →  {len(test_manifest)} manifest records")
    print(f"  Val manifest:  {manifest_dir}/val_manifest.json")
    print(f"  Test manifest: {manifest_dir}/test_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pet dataset")
    parser.add_argument("--raw_dir",      default="/content/drive/MyDrive/genai_assignment/data/pets_raw")
    parser.add_argument("--cache_dir",    default="/content/drive/MyDrive/genai_assignment/data/pets_cache")
    parser.add_argument("--manifest_dir", default="data/manifests")
    parser.add_argument("--seed",         type=int, default=42)
    main(parser.parse_args())
