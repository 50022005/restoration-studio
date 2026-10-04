# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 1: Foundations + Task 1 (Universal AE) + Task 2 (Hard Routing)
# Deadline: October 4 2026
# Runtime environment: Google Colab (free T4 GPU, 12 h session limit)
#
# HOW TO USE:
#   Copy each cell (separated by  # %% CELL N  markers) into a Colab notebook,
#   OR run this file directly with:  python notebooks/M1_foundations_task1_task2.py
#
# Expected total runtime on T4:  ~8-9 hours
#   Data prep:      ~8 min   (once; subsequent sessions skip download)
#   Task 1 Optuna:  ~3-4 h   (20 trials × 5 epochs)
#   Task 1 train:   ~20 min  (30 epochs)
#   Task 2 clf:     ~1.5 h   (15 optuna trials + 20 final epochs)
#   Task 2 spec:    ~3 h     (15 shared optuna + 3×20 independent epochs)
# ══════════════════════════════════════════════════════════════════════════════

# %% CELL 1 ─────────────────────────── Install dependencies (run once per session)
# Colab already has torch 2.x; we only need to add optuna, wandb, scikit-image
# Expected runtime: ~90 seconds

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("optuna==3.6.1",
    "wandb==0.17.0",
    "scikit-image==0.23.2",
    "onnx==1.16.1",
    "onnxruntime==1.18.0",
    "tqdm",
    "seaborn",
    "pytest")

print("✓ Dependencies installed")

# %% CELL 2 ─────────────────────────── Mount Google Drive + set global paths

from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

# ── Editable paths ──────────────────────────────────────────────────────────
BASE_DIR       = Path("/content/drive/MyDrive/genai_assignment")
DATA_RAW       = BASE_DIR / "data" / "pets_raw"
DATA_CACHE     = BASE_DIR / "data" / "pets_cache"
MANIFEST_DIR   = Path("data/manifests")          # committed to git
CKPT_T1        = BASE_DIR / "checkpoints" / "task1"
CKPT_T2_CLF    = BASE_DIR / "checkpoints" / "task2_clf"
CKPT_T2_SPEC   = BASE_DIR / "checkpoints" / "task2_specialists"
OPTUNA_DIR     = BASE_DIR / "optuna"
MODELS_DIR     = BASE_DIR / "models"
REPO_DIR       = Path("/content/genai_assignment")   # cloned repo root

for d in [BASE_DIR, DATA_RAW, DATA_CACHE, MANIFEST_DIR,
          CKPT_T1, CKPT_T2_CLF, CKPT_T2_SPEC, OPTUNA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✓ Paths ready")
print(f"  BASE_DIR : {BASE_DIR}")

# %% CELL 3 ─────────────────────────── Clone / update repo + add to sys.path

import subprocess, sys

GITHUB_URL = "https://github.com/YOUR_USERNAME/genai-assignment.git"   # ← set this

repo_path  = Path("/content/genai_assignment")
if not repo_path.exists():
    subprocess.run(["git", "clone", GITHUB_URL, str(repo_path)], check=True)
else:
    subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)

if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

print(f"✓ Repo at {repo_path}")

# %% CELL 4 ─────────────────────────── Fix all random seeds (determinism)

import random
import os
import numpy as np
import torch

GLOBAL_SEED = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✓ All seeds fixed to {seed}")

set_all_seeds()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% CELL 5 ─────────────────────────── W&B login

import wandb

# Option A: interactive login (prompts in Colab)
wandb.login()

# Option B: headless (uncomment and set key in Colab secrets)
# import os; os.environ["WANDB_API_KEY"] = "your_key"
# wandb.login(key=os.environ["WANDB_API_KEY"])

WB_PROJECT = "genai-assignment"
print(f"✓ W&B project: {WB_PROJECT}")

# %% CELL 6 ─────────────────────────── Download + cache Oxford-IIIT Pet
# Expected runtime: ~8 min (first run); ~10 sec (subsequent runs, data cached)

import tarfile, urllib.request, json
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

IMAGES_URL      = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

def _download(url, dest):
    if Path(dest).exists():
        return
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} …")
    with tqdm(unit="B", unit_scale=True, miniters=1) as pbar:
        urllib.request.urlretrieve(url, str(dest),
            lambda b, bs, ts: (setattr(pbar, "total", ts) or pbar.update(b*bs - pbar.n)))
    print(f"  ✓ saved to {dest}")

def _extract(tarpath, dest_parent):
    marker = Path(dest_parent) / "_extracted"
    if marker.exists():
        return
    print(f"Extracting {tarpath} …")
    with tarfile.open(str(tarpath)) as t:
        t.extractall(str(dest_parent))
    marker.touch()
    print("  ✓ done")

_download(IMAGES_URL,      DATA_RAW / "images.tar.gz")
_download(ANNOTATIONS_URL, DATA_RAW / "annotations.tar.gz")
_extract(DATA_RAW / "images.tar.gz",      DATA_RAW)
_extract(DATA_RAW / "annotations.tar.gz", DATA_RAW)

IMAGES_DIR = DATA_RAW / "images"
ANNOT_DIR  = DATA_RAW / "annotations"

def _read_file_list(txt_path):
    stems = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                stems.append(line.split()[0])
    return stems

def _stems_to_paths(stems, img_dir):
    paths = []
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                paths.append(str(p))
                break
    return paths

tv_paths   = _stems_to_paths(_read_file_list(ANNOT_DIR / "trainval.txt"), IMAGES_DIR)
test_paths = _stems_to_paths(_read_file_list(ANNOT_DIR / "test.txt"),     IMAGES_DIR)

train_paths, val_paths = train_test_split(
    tv_paths, test_size=0.20, random_state=42, shuffle=True
)

print(f"✓ Split: train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

def cache_images(paths, cache_dir, desc="Caching"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_paths = []
    for p in tqdm(paths, desc=desc):
        stem     = Path(p).stem
        out_path = cache_dir / f"{stem}.npy"
        if not out_path.exists():
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((128, 128), Image.BILINEAR)
                np.save(str(out_path), np.array(img, dtype=np.uint8))
            except Exception as e:
                print(f"  [warn] {p}: {e}")
                continue
        npy_paths.append(str(out_path))
    return npy_paths

train_npy = cache_images(train_paths, DATA_CACHE / "train", "Caching train")
val_npy   = cache_images(val_paths,   DATA_CACHE / "val",   "Caching val")
test_npy  = cache_images(test_paths,  DATA_CACHE / "test",  "Caching test")

split_data = {"seed": 42, "train": train_npy, "val": val_npy, "test": test_npy}
with open(MANIFEST_DIR / "split.json", "w") as f:
    json.dump(split_data, f, indent=2)

print(f"✓ Caching complete.  split.json → {MANIFEST_DIR / 'split.json'}")

# %% CELL 7 ─────────────────────────── Corruption functions (inline copy of src/utils/corruption.py)
# These are IDENTICAL to the module in src/utils/corruption.py.
# Inline here so the notebook is self-contained.

import json, torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path

CORRUPTION_TYPES = ["clean", "salt_pepper", "blur", "occlusion"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CORRUPTION_TYPES)}
IDX_TO_LABEL = {i: c for c, i in LABEL_TO_IDX.items()}

def apply_salt_pepper(img: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    img = img.copy()
    noise_mask = rng.random(img.shape[:2]) < p
    salt_mask  = rng.random(img.shape[:2]) >= 0.5
    img[noise_mask & salt_mask]  = 255
    img[noise_mask & ~salt_mask] = 0
    return img

def apply_gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    assert kernel_size % 2 == 1
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    g1d  = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g1d  = g1d / g1d.sum()
    g2d  = torch.outer(g1d, g1d)
    kern = g2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    pad  = kernel_size // 2
    t    = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        out = F.conv2d(t, kern, padding=pad, groups=3)
    return (out.squeeze(0).permute(1,2,0).clamp(0,1).numpy() * 255).astype(np.uint8)

def apply_occlusion(img: np.ndarray, rects: List[Tuple]) -> np.ndarray:
    img = img.copy()
    for x, y, w, h in rects:
        img[y:y+h, x:x+w] = 0
    return img

def _sample_rects(rng, H, W, num_rects, target_frac):
    per_frac = target_frac / num_rects
    rects = []
    for _ in range(num_rects):
        area = max(1, int(per_frac * H * W))
        rh   = max(1, int(rng.integers(max(1, area // W), max(2, H//2) + 1)))
        rh   = min(rh, H)
        rw   = max(1, min(area // max(rh, 1), W))
        x    = int(rng.integers(0, max(1, W - rw + 1)))
        y    = int(rng.integers(0, max(1, H - rh + 1)))
        rects.append((x, y, rw, rh))
    return rects

def corrupt_random(img, rng=None):
    if rng is None: rng = np.random.default_rng()
    ctype    = str(rng.choice(CORRUPTION_TYPES))
    severity = _sample_severity(ctype, rng, img.shape[0], img.shape[1])
    corrupted = corrupt_from_severity(img, ctype, severity, rng)
    return corrupted, ctype, severity

def _sample_severity(corruption, rng, H=128, W=128):
    if corruption == "clean":       return {}
    if corruption == "salt_pepper": return {"p": float(rng.uniform(0.02, 0.15))}
    if corruption == "blur":
        return {"kernel_size": int(rng.choice([3,5,7])), "sigma": float(rng.uniform(0.5,2.5))}
    if corruption == "occlusion":
        n = int(rng.integers(1, 4)); f = float(rng.uniform(0.10, 0.35))
        return {"num_rects": n, "target_frac": f,
                "rects": _sample_rects(rng, H, W, n, f)}

def corrupt_from_severity(img, corruption, severity, rng=None):
    if corruption == "clean":       return img.copy()
    if rng is None: rng = np.random.default_rng()
    if corruption == "salt_pepper": return apply_salt_pepper(img, severity["p"], rng)
    if corruption == "blur":        return apply_gaussian_blur(img, severity["kernel_size"], severity["sigma"])
    if corruption == "occlusion":   return apply_occlusion(img, [tuple(r) for r in severity["rects"]])
    raise ValueError(f"Unknown corruption: {corruption}")

def corrupt_from_record(img, record):
    rng = np.random.default_rng(record["seed"])
    return corrupt_from_severity(img, record["corruption_type"], record["severity"], rng)

print("✓ Corruption functions loaded")

# %% CELL 8 ─────────────────────────── Unit tests (run in-notebook)
# These must all pass before proceeding. Expected runtime: ~10 seconds.

import hashlib

def _sha(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

def _test_corruption_functions():
    rng   = np.random.default_rng(0)
    img   = (rng.uniform(50, 200, (128, 128, 3))).astype(np.uint8)

    # Shape preservation
    for fn, args in [
        (apply_salt_pepper,   (0.1,  np.random.default_rng(1))),
        (apply_gaussian_blur, (5,    1.5)),
        (apply_occlusion,     ([(10, 10, 30, 30)],)),
    ]:
        out = fn(img, *args)
        assert out.shape == img.shape and out.dtype == np.uint8, f"Shape/dtype fail: {fn.__name__}"

    # Salt-and-pepper only extreme values
    out = apply_salt_pepper(img, 0.3, np.random.default_rng(42))
    changed = (out != img).any(axis=2)
    assert changed.sum() > 0
    for pix in out[changed]:
        assert all(v in (0, 255) for v in pix)

    # Blur determinism
    b1 = apply_gaussian_blur(img, 5, 1.5)
    b2 = apply_gaussian_blur(img, 5, 1.5)
    assert np.array_equal(b1, b2), "Blur not deterministic"

    # All 4 types sampled
    seen = set()
    for s in range(400):
        _, ctype, _ = corrupt_random(img, np.random.default_rng(s))
        seen.add(ctype)
    assert seen == set(CORRUPTION_TYPES), f"Not all types seen: {seen}"

    # corrupt_from_record reproduces original
    for seed in range(30):
        img_seed = int(np.random.default_rng(seed).integers(0, 2**31))
        img_rng  = np.random.default_rng(img_seed)
        corrupted, ctype, severity = corrupt_random(img, img_rng)
        record = {"corruption_type": ctype, "severity": severity, "seed": img_seed}
        reproduced = corrupt_from_record(img, record)
        assert np.array_equal(corrupted, reproduced), (
            f"Seed {img_seed}: record did not reproduce {ctype!r} exactly")

    print("✓ All corruption unit tests passed")

_test_corruption_functions()

# %% CELL 9 ─────────────────────────── Generate val manifest (deterministic)

def generate_val_manifest(image_paths, output_path, seed=100, H=128, W=128):
    master_rng = np.random.default_rng(seed)
    records = []
    for idx, img_path in enumerate(image_paths):
        img_seed   = int(master_rng.integers(0, 2**31))
        img_rng    = np.random.default_rng(img_seed)
        corruption = str(img_rng.choice(CORRUPTION_TYPES))
        severity   = _sample_severity(corruption, img_rng, H, W)
        records.append({
            "index": idx, "image_path": str(img_path),
            "corruption_type": corruption, "severity": severity, "seed": img_seed,
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f: json.dump(records, f, indent=2)
    return records

val_manifest = generate_val_manifest(
    val_npy, str(MANIFEST_DIR / "val_manifest.json"), seed=100
)
print(f"✓ Val manifest: {len(val_manifest)} records")
print(f"  Corruption distribution: { {c: sum(r['corruption_type']==c for r in val_manifest) for c in CORRUPTION_TYPES} }")

# %% CELL 10 ─────────────────────────── Generate test manifest (3 fixed severities)

def generate_test_manifest(image_paths, output_path, seed=200, H=128, W=128):
    master_rng = np.random.default_rng(seed)
    records    = []

    SALT_SEV = [("low",{"p":0.03}), ("medium",{"p":0.08}), ("high",{"p":0.15})]
    BLUR_SEV = [
        ("low",    {"kernel_size":3,"sigma":0.7}),
        ("medium", {"kernel_size":5,"sigma":1.5}),
        ("high",   {"kernel_size":7,"sigma":2.5}),
    ]
    OCC_CFG  = [("low",1,0.10), ("medium",2,0.20), ("high",3,0.35)]

    def rec(idx, path, ctype, level, sev, s):
        return {"image_index":idx,"image_path":str(path),
                "corruption_type":ctype,"severity_level":level,"severity":sev,"seed":s}

    for img_idx, img_path in enumerate(image_paths):
        records.append(rec(img_idx, img_path, "clean", "none", {}, 0))
        for level, sev in SALT_SEV:
            s = int(master_rng.integers(0, 2**31))
            records.append(rec(img_idx, img_path, "salt_pepper", level, sev, s))
        for level, sev in BLUR_SEV:
            s = int(master_rng.integers(0, 2**31))
            records.append(rec(img_idx, img_path, "blur", level, sev, s))
        for level, nr, tf in OCC_CFG:
            s   = int(master_rng.integers(0, 2**31))
            rng = np.random.default_rng(s)
            rects = _sample_rects(rng, H, W, nr, tf)
            sev = {"num_rects": nr, "target_frac": tf, "rects": rects}
            records.append(rec(img_idx, img_path, "occlusion", level, sev, s))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f: json.dump(records, f, indent=2)
    return records

test_manifest = generate_test_manifest(
    test_npy, str(MANIFEST_DIR / "test_manifest.json"), seed=200
)
print(f"✓ Test manifest: {len(test_manifest)} records  ({len(test_npy)} images × 10 conditions)")
assert len(test_manifest) == len(test_npy) * 10

# %% CELL 11 ─────────────────────────── Determinism check for manifests

import hashlib, json, copy

def _manifest_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()

# Re-generate both manifests with same seeds → must be identical
val2  = generate_val_manifest(val_npy,  "/tmp/val2.json",  seed=100)
test2 = generate_test_manifest(test_npy, "/tmp/test2.json", seed=200)

assert _manifest_hash(val_manifest) == _manifest_hash(val2),  "Val manifest not deterministic!"
assert _manifest_hash(test_manifest) == _manifest_hash(test2), "Test manifest not deterministic!"
print("✓ Manifest determinism check PASSED")

# Pixel-level check: corrupt_from_record on the same record is identical
sample_img = np.load(val_npy[0])
for record in val_manifest[:20]:
    a = corrupt_from_record(sample_img, record)
    b = corrupt_from_record(sample_img, record)
    assert np.array_equal(a, b), f"Pixel reproduction failed for record {record['index']}"
print("✓ Pixel-level reproduction check PASSED")

# %% CELL 12 ─────────────────────────── Dataset class

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class PetDataset(Dataset):
    def __init__(self, npy_paths, manifest=None, deterministic=False, seed=0):
        self.npy_paths    = npy_paths
        self.manifest     = manifest
        self.training     = manifest is None
        self.deterministic = deterministic
        self.seed         = seed

    def __len__(self):
        return len(self.npy_paths) if self.training else len(self.manifest)

    def __getitem__(self, idx):
        if self.training:
            img = np.load(self.npy_paths[idx])
            rng = (np.random.default_rng(self.seed + idx)
                   if self.deterministic else np.random.default_rng())
            corrupted, ctype, _ = corrupt_random(img, rng)
            label = LABEL_TO_IDX[ctype]
        else:
            rec       = self.manifest[idx]
            img       = np.load(rec["image_path"])
            corrupted = corrupt_from_record(img, rec)
            label     = LABEL_TO_IDX[rec["corruption_type"]]

        clean_t     = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        corrupted_t = torch.from_numpy(corrupted).permute(2,0,1).float() / 255.0
        return corrupted_t, clean_t, label


def make_balanced_sampler(labels):
    counts  = np.bincount(labels, minlength=4).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    sw      = torch.tensor([weights[l] for l in labels], dtype=torch.float64)
    return WeightedRandomSampler(sw, len(sw), replacement=True)

# Sanity check
_ds  = PetDataset(train_npy[:4], deterministic=True)
_c, _x, _l = _ds[0]
assert _c.shape == (3, 128, 128) and _x.shape == (3, 128, 128)
assert 0 <= _l <= 3
print(f"✓ Dataset OK  shape={tuple(_c.shape)}  label={IDX_TO_LABEL[_l]}")

# %% CELL 13 ─────────────────────────── Task 1: Universal AE model

import torch.nn as nn

class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0: layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.Identity() if last else nn.BatchNorm2d(out_ch),
            nn.Tanh() if last else nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UniversalAE(nn.Module):
    """
    Convolutional autoencoder with FC bottleneck.  No skip connections.
    Input / output: (B, 3, 128, 128) float32 in [0, 1]
    """
    def __init__(self, base_channels=32, bottleneck_dim=256, dropout=0.0):
        super().__init__()
        C         = base_channels
        flat_dim  = C * 8 * 8 * 8          # after 4 stride-2 stages on 128px
        self._C   = C
        self.enc  = nn.Sequential(
            EncBlock(3,   C,   dropout),    # 128→64
            EncBlock(C,   C*2, dropout),    # 64 →32
            EncBlock(C*2, C*4, dropout),    # 32 →16
            EncBlock(C*4, C*8, dropout),    # 16 → 8
        )
        self.fc_enc = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, bottleneck_dim), nn.ReLU(True))
        self.fc_dec = nn.Sequential(nn.Linear(bottleneck_dim, flat_dim), nn.ReLU(True))
        self.dec  = nn.Sequential(
            DecBlock(C*8, C*4),             # 8 →16
            DecBlock(C*4, C*2),             # 16→32
            DecBlock(C*2, C),               # 32→64
            DecBlock(C,   3,  last=True),   # 64→128  (tanh)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)

    def encode(self, x):  return self.fc_enc(self.enc(x))
    def decode(self, z):
        C = self._C
        return ((self.dec(self.fc_dec(z).view(-1, C*8, 8, 8))) + 1.0) * 0.5
    def forward(self, x): return self.decode(self.encode(x))

# Quick sanity check
_m = UniversalAE(32, 256, 0.0).to(DEVICE)
_x = torch.randn(2, 3, 128, 128).to(DEVICE)
_y = _m(_x)
assert _y.shape == (2, 3, 128, 128)
assert _y.min() >= 0 and _y.max() <= 1
params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(f"✓ UniversalAE OK  output={tuple(_y.shape)}  params={params/1e6:.2f}M")
del _m, _x, _y

# %% CELL 14 ─────────────────────────── L1 + SSIM loss

import torch.nn.functional as F

def _gauss_kernel(size=11, sigma=1.5, channels=3):
    half   = size // 2
    coords = torch.arange(-half, half+1, dtype=torch.float32)
    g1d    = torch.exp(-coords**2 / (2*sigma**2)); g1d /= g1d.sum()
    g2d    = torch.outer(g1d, g1d)
    return g2d.unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1)

_SSIM_KERNEL = None   # cached lazily

def ssim_val(x, y, window=11, C1=0.01**2, C2=0.03**2):
    global _SSIM_KERNEL
    B, C, H, W = x.shape
    if _SSIM_KERNEL is None or _SSIM_KERNEL.shape[0] != C:
        _SSIM_KERNEL = _gauss_kernel(window, 1.5, C).to(x.device)
    k   = _SSIM_KERNEL
    pad = window // 2
    mu_x  = F.conv2d(x, k, padding=pad, groups=C)
    mu_y  = F.conv2d(y, k, padding=pad, groups=C)
    mu_xx = mu_x*mu_x;  mu_yy = mu_y*mu_y;  mu_xy = mu_x*mu_y
    sig_xx = F.conv2d(x*x, k, padding=pad, groups=C) - mu_xx
    sig_yy = F.conv2d(y*y, k, padding=pad, groups=C) - mu_yy
    sig_xy = F.conv2d(x*y, k, padding=pad, groups=C) - mu_xy
    num    = (2*mu_xy + C1)*(2*sig_xy + C2)
    denom  = (mu_xx + mu_yy + C1)*(sig_xx + sig_yy + C2)
    return (num / (denom + 1e-8)).mean()

class ReconLoss(nn.Module):
    """alpha * L1 + (1-alpha) * (1-SSIM)"""
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
    def forward(self, x_hat, x):
        return self.alpha * F.l1_loss(x_hat,x) + (1-self.alpha)*(1-ssim_val(x_hat,x))

# Test loss
_l = ReconLoss(0.8)
_x = torch.rand(2, 3, 128, 128)
assert _l(_x, _x).item() < 0.01, "Loss on identical images should be near zero"
print("✓ ReconLoss OK")

# %% CELL 15 ─────────────────────────── Task 1: Optuna study
# Expected runtime: 20 trials × ~10 min/trial = ~3.5 hours on T4
# The study is stored in SQLite on Drive; safe to disconnect.

import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from torch.cuda.amp  import GradScaler, autocast

STORAGE_T1 = f"sqlite:///{OPTUNA_DIR}/task1.db"

def train_one_epoch_t1(model, loader, criterion, optimiser, scaler):
    model.train()
    total = 0
    for corrupted, clean, _ in loader:
        corrupted, clean = corrupted.to(DEVICE), clean.to(DEVICE)
        optimiser.zero_grad(set_to_none=True)
        with autocast():
            out  = model(corrupted)
            loss = criterion(out, clean)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser); scaler.update()
        total += loss.item()
    return total / len(loader)

def val_one_epoch_t1(model, loader, criterion):
    model.eval()
    total_loss = 0; total_ssim = 0
    with torch.no_grad():
        for corrupted, clean, _ in loader:
            corrupted, clean = corrupted.to(DEVICE), clean.to(DEVICE)
            with autocast():
                out  = model(corrupted)
                loss = criterion(out, clean)
            total_loss += loss.item()
            total_ssim += ssim_val(out, clean).item()
    n = len(loader)
    return total_loss/n, total_ssim/n

def t1_objective(trial):
    # ── Hyperparameter suggestions ────────────────────────────────────────
    lr             = trial.suggest_float("lr",             1e-4, 1e-2,  log=True)
    batch_size     = trial.suggest_categorical("batch_size",     [16, 32, 64])
    bottleneck_dim = trial.suggest_int("bottleneck_dim",         64, 512, step=64)
    base_channels  = trial.suggest_categorical("base_channels",  [16, 32, 48])
    dropout        = trial.suggest_float("dropout",        0.0,  0.4,  step=0.05)
    alpha          = trial.suggest_float("alpha",          0.50, 0.95, step=0.05)

    model     = UniversalAE(base_channels, bottleneck_dim, dropout).to(DEVICE)
    criterion = ReconLoss(alpha)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = GradScaler()

    train_ds  = PetDataset(train_npy)
    val_ds    = PetDataset(val_npy, manifest=val_manifest)
    train_ld  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    val_ld    = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)

    TRIAL_EPOCHS = 5
    best_obj     = -1.0
    for epoch in range(TRIAL_EPOCHS):
        train_one_epoch_t1(model, train_ld, criterion, optimiser, scaler)
        val_loss, val_ssim = val_one_epoch_t1(model, val_ld, criterion)
        # Objective: maximise SSIM + (1 - normalised L1)
        obj = val_ssim
        trial.report(obj, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_obj = max(best_obj, obj)

    return best_obj

sampler_t1 = TPESampler(seed=42)
pruner_t1  = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
study_t1   = optuna.create_study(
    study_name="task1_universal_ae",
    direction="maximize",
    storage=STORAGE_T1,
    load_if_exists=True,
    sampler=sampler_t1,
    pruner=pruner_t1,
)

print(f"▶ Starting Task 1 Optuna (already have {len(study_t1.trials)} trials)")
study_t1.optimize(t1_objective, n_trials=20, show_progress_bar=True)

best_t1 = study_t1.best_params
print("\n─── Task 1 Best Trial ───")
print(f"  Value (SSIM): {study_t1.best_value:.4f}")
for k, v in best_t1.items(): print(f"  {k}: {v}")

# W&B: log Optuna results
run_optuna_t1 = wandb.init(project=WB_PROJECT, name="task1-optuna",
                            tags=["task1","optuna"], reinit=True)
for i, trial in enumerate(study_t1.trials):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        wandb.log({"optuna/trial": i, "optuna/value": trial.value, **trial.params})
wandb.finish()
print("✓ Task 1 Optuna complete.  Study persisted to Drive SQLite.")

# %% CELL 16 ─────────────────────────── Task 1: Final training (best config)
# Expected runtime: ~20-25 min on T4

from torch.optim.lr_scheduler import CosineAnnealingLR

T1_EPOCHS   = 30
best_lr     = best_t1["lr"]
best_bs     = best_t1["batch_size"]
best_bn     = best_t1["bottleneck_dim"]
best_bc     = best_t1["base_channels"]
best_dp     = best_t1["dropout"]
best_alpha  = best_t1["alpha"]

model_t1    = UniversalAE(best_bc, best_bn, best_dp).to(DEVICE)
criterion_t1 = ReconLoss(best_alpha)
optimiser_t1 = torch.optim.AdamW(model_t1.parameters(), lr=best_lr, weight_decay=1e-4)
scheduler_t1 = CosineAnnealingLR(optimiser_t1, T_max=T1_EPOCHS)
scaler_t1    = GradScaler()

train_ld_t1 = DataLoader(PetDataset(train_npy), batch_size=best_bs,
                          shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_ld_t1   = DataLoader(PetDataset(val_npy, manifest=val_manifest),
                          batch_size=best_bs, shuffle=False, num_workers=2, pin_memory=True)

# Resume from checkpoint if present
CKPT_T1.mkdir(parents=True, exist_ok=True)
ckpt_file   = CKPT_T1 / "checkpoint.pt"
start_epoch = 0
best_ssim   = 0.0
if ckpt_file.exists():
    state       = torch.load(str(ckpt_file), map_location=DEVICE)
    model_t1.load_state_dict(state["model"])
    optimiser_t1.load_state_dict(state["optimiser"])
    start_epoch = state["epoch"] + 1
    best_ssim   = state.get("metrics", {}).get("val_ssim", 0.0)
    print(f"▶ Resumed from epoch {state['epoch']}  best_ssim={best_ssim:.4f}")

run_t1 = wandb.init(project=WB_PROJECT, name="task1-final-train",
                    config=best_t1, tags=["task1","final"], reinit=True)

for epoch in range(start_epoch, T1_EPOCHS):
    train_loss = train_one_epoch_t1(model_t1, train_ld_t1, criterion_t1, optimiser_t1, scaler_t1)
    val_loss, val_ssim = val_one_epoch_t1(model_t1, val_ld_t1, criterion_t1)
    scheduler_t1.step()
    is_best = val_ssim > best_ssim
    if is_best: best_ssim = val_ssim

    # Log to W&B
    wandb.log({"train/loss": train_loss, "val/loss": val_loss, "val/ssim": val_ssim,
               "epoch": epoch, "lr": scheduler_t1.get_last_lr()[0]})

    # Checkpoint every epoch (safe for disconnect)
    state = {"epoch": epoch, "model": model_t1.state_dict(),
             "optimiser": optimiser_t1.state_dict(),
             "metrics": {"val_ssim": val_ssim, "val_loss": val_loss},
             "config": best_t1}
    torch.save(state, str(ckpt_file))
    if is_best:
        torch.save(state, str(CKPT_T1 / "best.pt"))

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:3d}/{T1_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_ssim={val_ssim:.4f}  {'★ BEST' if is_best else ''}")

wandb.finish()
print(f"\n✓ Task 1 training complete.  Best val_ssim={best_ssim:.4f}")

# %% CELL 17 ─────────────────────────── Task 1: Per-corruption & per-severity evaluation

import pandas as pd
from collections import defaultdict

model_t1.eval()
results_t1 = defaultdict(lambda: defaultdict(list))  # [ctype][severity] → [ssim values]

test_ld_t1 = DataLoader(
    PetDataset(test_npy, manifest=test_manifest),
    batch_size=64, shuffle=False, num_workers=2
)

def compute_metrics_t1(model, loader, device):
    """Returns dict: {corruption_type: {severity_level: {"ssim":..., "l1":...}}}"""
    from collections import defaultdict
    results = defaultdict(lambda: defaultdict(lambda: {"ssim": [], "l1": []}))
    model.eval()
    rec_idx = 0
    with torch.no_grad():
        for corrupted, clean, labels in loader:
            corrupted = corrupted.to(device)
            clean     = clean.to(device)
            with autocast():
                out = model(corrupted)
            B = corrupted.shape[0]
            for i in range(B):
                if rec_idx < len(test_manifest):
                    rec     = test_manifest[rec_idx]
                    ctype   = rec["corruption_type"]
                    slevel  = rec.get("severity_level", "none")
                    s_val   = ssim_val(out[i:i+1], clean[i:i+1]).item()
                    l1_val  = F.l1_loss(out[i:i+1], clean[i:i+1]).item()
                    results[ctype][slevel]["ssim"].append(s_val)
                    results[ctype][slevel]["l1"].append(l1_val)
                    rec_idx += 1
    return results

t1_results = compute_metrics_t1(model_t1, test_ld_t1, DEVICE)

# Print table
print("\n─── Task 1 Evaluation Results ───")
rows = []
for ctype in CORRUPTION_TYPES:
    for slevel in (["none"] if ctype=="clean" else ["low","medium","high"]):
        data = t1_results.get(ctype, {}).get(slevel, {})
        if data:
            rows.append({
                "Corruption": ctype, "Severity": slevel,
                "SSIM":  f"{np.mean(data['ssim']):.4f}",
                "L1":    f"{np.mean(data['l1']):.4f}",
                "N":     len(data['ssim']),
            })
df_t1 = pd.DataFrame(rows)
print(df_t1.to_string(index=False))

# Log table to W&B
run_eval_t1 = wandb.init(project=WB_PROJECT, name="task1-eval",
                          tags=["task1","evaluation"], reinit=True)
wandb.log({"task1/results_table": wandb.Table(dataframe=df_t1)})
wandb.finish()

# %% CELL 18 ─────────────────────────── Task 1: 12-example visual grid + 4 failure cases
# EVIDENCE REQUIRED: Save these plots to Drive (SSIM grid and failure grid)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams["figure.dpi"] = 130

def tensor_to_np(t):
    return (t.squeeze().permute(1,2,0).cpu().float().clamp(0,1).numpy() * 255).astype(np.uint8)

def plot_visual_grid(model, npy_paths, manifest, n=12, title="Reconstruction Grid"):
    """
    Plot grid:  Clean | Corrupted | Reconstructed | |Error|
    """
    model.eval()
    # Pick n evenly-spaced examples (one per corruption type × 3 severity levels)
    idxs = list(range(0, min(n*10, len(manifest)), max(1, len(manifest)//(n*2))))[:n]

    fig, axes = plt.subplots(n, 4, figsize=(12, n*2.5))
    fig.suptitle(title, fontsize=12, weight="bold")
    cols = ["Clean Target", "Corrupted Input", "Reconstructed", "|Error| × 3"]

    for col, ax in enumerate(axes[0]):
        ax.set_title(cols[col], fontsize=9)

    for row, idx in enumerate(idxs):
        rec      = manifest[idx]
        img_path = rec["image_path"]
        clean_np = np.load(img_path)
        corr_np  = corrupt_from_record(clean_np, rec)

        clean_t  = torch.from_numpy(clean_np).permute(2,0,1).float().unsqueeze(0) / 255.0
        corr_t   = torch.from_numpy(corr_np).permute(2,0,1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            with autocast():
                recon_t = model(corr_t.to(DEVICE)).cpu()

        error_t = (clean_t - recon_t).abs() * 3   # amplify for visibility
        imgs    = [clean_t, corr_t, recon_t, error_t.clamp(0,1)]
        label   = f"{rec['corruption_type']} / {rec.get('severity_level','')}"

        for col, img_t in enumerate(imgs):
            axes[row][col].imshow(tensor_to_np(img_t))
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_ylabel(label, fontsize=7, rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    save_path = str(BASE_DIR / "task1_reconstruction_grid.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved → {save_path}  ← add to IEEE report")
    return save_path

# 12 representative examples
t1_grid_path = plot_visual_grid(model_t1, test_npy, test_manifest, n=12,
                                title="Task 1 – Universal AE Reconstructions")

# 4 failure cases (pick high severity records with low SSIM)
failure_indices = []
model_t1.eval()
for idx, rec in enumerate(test_manifest):
    if rec.get("severity_level") == "high" and rec["corruption_type"] != "clean":
        clean_np = np.load(rec["image_path"])
        corr_np  = corrupt_from_record(clean_np, rec)
        clean_t  = torch.from_numpy(clean_np).permute(2,0,1).float().unsqueeze(0)/255.0
        corr_t   = torch.from_numpy(corr_np).permute(2,0,1).float().unsqueeze(0)/255.0
        with torch.no_grad():
            with autocast(): recon_t = model_t1(corr_t.to(DEVICE)).cpu()
        s = ssim_val(recon_t, clean_t).item()
        failure_indices.append((s, idx))

failure_indices.sort()                              # lowest SSIM = worst cases
worst_idxs = [i for _, i in failure_indices[:4]]
t1_fail_path = plot_visual_grid(model_t1, test_npy,
                                [test_manifest[i] for i in worst_idxs] + [test_manifest[worst_idxs[0]]]*8,
                                n=4, title="Task 1 – 4 Failure Cases (Lowest SSIM)")
print(f"✓ Failure case analysis complete.  Worst SSIM: {failure_indices[0][0]:.4f}")

# Log to W&B
run_viz_t1 = wandb.init(project=WB_PROJECT, name="task1-visualisations",
                         tags=["task1","visualisations"], reinit=True)
wandb.log({"task1/reconstruction_grid": wandb.Image(t1_grid_path),
           "task1/failure_cases":       wandb.Image(t1_fail_path)})
wandb.finish()

# %% CELL 19 ─────────────────────────── Task 1: ONNX export + verification
# Expected runtime: ~2 min

import onnx
import onnxruntime as ort

ONNX_T1 = str(MODELS_DIR / "task1_universal.onnx")
model_t1.eval()

dummy  = torch.randn(1, 3, 128, 128, device=DEVICE)
torch.onnx.export(
    model_t1, dummy, ONNX_T1,
    input_names=["corrupted_image"],
    output_names=["restored_image"],
    opset_version=17,
    dynamic_axes={"corrupted_image": {0: "batch"}, "restored_image": {0: "batch"}},
)
onnx.checker.check_model(ONNX_T1)
print(f"✓ ONNX model saved & checked → {ONNX_T1}")

# Verify parity
sess  = ort.InferenceSession(ONNX_T1, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name
rng   = np.random.default_rng(0)
all_pass = True
for i in range(16):
    x_np   = rng.random((1, 3, 128, 128)).astype(np.float32)
    pt_out = model_t1(torch.from_numpy(x_np).to(DEVICE)).cpu().detach().numpy()
    ort_out = sess.run([oname], {iname: x_np})[0]
    diff   = np.abs(pt_out - ort_out).max()
    ok     = np.allclose(pt_out, ort_out, atol=1e-4, rtol=1e-3)
    if not ok:
        print(f"  ✗ sample {i}: max_diff={diff:.2e}")
        all_pass = False

if all_pass:
    print("✓ ONNX parity verified (all 16 samples within atol=1e-4)")
else:
    print("✗ ONNX parity check FAILED – check export ops")

# %% CELL 20 ─────────────────────────── Task 2: Classifier model

class CorruptionClassifier(nn.Module):
    """4-class classifier: clean, salt_pepper, blur, occlusion."""
    def __init__(self, base_channels=32, dropout=0.3):
        super().__init__()
        C = base_channels
        def cb(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
            )
        self.features = nn.Sequential(
            cb(3, C), cb(C, C*2), cb(C*2, C*4), cb(C*4, C*8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(C*8, 4))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x): return self.head(self.pool(self.features(x)))
    def predict_proba(self, x): return torch.softmax(self.forward(x), dim=1)

_clf = CorruptionClassifier(32, 0.3).to(DEVICE)
_y   = _clf(torch.randn(2, 3, 128, 128).to(DEVICE))
assert _y.shape == (2, 4)
params_clf = sum(p.numel() for p in _clf.parameters() if p.requires_grad)
print(f"✓ Classifier OK  params={params_clf/1e6:.2f}M")
del _clf, _y

# %% CELL 21 ─────────────────────────── Task 2: Classifier Optuna
# Expected runtime: 15 trials × ~6 min = ~1.5 hours on T4

import torch.nn as nn

STORAGE_CLF = f"sqlite:///{OPTUNA_DIR}/task2_clf.db"
CE_LOSS     = nn.CrossEntropyLoss()

def train_clf_epoch(model, loader, optimiser, scaler, device):
    model.train()
    total_loss = 0; correct = 0; n = 0
    for corrupted, _, labels in loader:
        corrupted = corrupted.to(device); labels = labels.to(device)
        optimiser.zero_grad(set_to_none=True)
        with autocast():
            logits = model(corrupted)
            loss   = CE_LOSS(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser); scaler.update()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)
    return total_loss / len(loader), correct / n

def val_clf_epoch(model, loader, device):
    model.eval()
    correct = 0; n = 0
    with torch.no_grad():
        for corrupted, _, labels in loader:
            corrupted = corrupted.to(device); labels = labels.to(device)
            with autocast():
                logits = model(corrupted)
            correct += (logits.argmax(1) == labels).sum().item()
            n       += labels.size(0)
    return correct / n

def clf_objective(trial):
    lr            = trial.suggest_float("lr",           1e-4, 5e-3,  log=True)
    batch_size    = trial.suggest_categorical("batch_size",   [32, 64, 128])
    base_channels = trial.suggest_categorical("base_channels",[16, 32, 64])
    dropout       = trial.suggest_float("dropout",      0.0,  0.5,  step=0.1)
    weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    model     = CorruptionClassifier(base_channels, dropout).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler    = GradScaler()

    # Balanced sampler for training
    train_ds  = PetDataset(train_npy, deterministic=True, seed=0)
    labels    = [train_ds[i][2] for i in range(min(len(train_ds), 2000))]
    sampler   = make_balanced_sampler(labels)
    train_ld  = DataLoader(train_ds, batch_size=batch_size,
                           sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    val_ld    = DataLoader(PetDataset(val_npy, manifest=val_manifest),
                           batch_size=batch_size, shuffle=False, num_workers=2)

    best_acc  = 0.0
    for epoch in range(5):
        train_clf_epoch(model, train_ld, optimiser, scaler, DEVICE)
        acc = val_clf_epoch(model, val_ld, DEVICE)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_acc = max(best_acc, acc)
    return best_acc

study_clf = optuna.create_study(
    study_name="task2_classifier",
    direction="maximize",
    storage=STORAGE_CLF,
    load_if_exists=True,
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=2),
)
print(f"▶ Starting Classifier Optuna (already have {len(study_clf.trials)} trials)")
study_clf.optimize(clf_objective, n_trials=15, show_progress_bar=True)

best_clf = study_clf.best_params
print("\n─── Classifier Best Trial ───")
print(f"  Accuracy: {study_clf.best_value:.4f}")
for k, v in best_clf.items(): print(f"  {k}: {v}")

# %% CELL 22 ─────────────────────────── Task 2: Classifier final training + evaluation
# Expected runtime: ~20 min

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import seaborn as sns

T2_CLF_EPOCHS = 20

model_clf    = CorruptionClassifier(best_clf["base_channels"], best_clf["dropout"]).to(DEVICE)
opt_clf      = torch.optim.AdamW(model_clf.parameters(),
                                  lr=best_clf["lr"], weight_decay=best_clf["weight_decay"])
sch_clf      = CosineAnnealingLR(opt_clf, T_max=T2_CLF_EPOCHS)
scaler_clf   = GradScaler()

# Balanced training loader
train_ds_clf = PetDataset(train_npy, deterministic=True, seed=0)
labels_full  = [train_ds_clf[i][2] for i in range(len(train_ds_clf))]
sampler_clf  = make_balanced_sampler(labels_full)
train_ld_clf = DataLoader(train_ds_clf, batch_size=best_clf["batch_size"],
                           sampler=sampler_clf, num_workers=2, pin_memory=True, drop_last=True)
val_ld_clf   = DataLoader(PetDataset(val_npy, manifest=val_manifest),
                           batch_size=best_clf["batch_size"], shuffle=False, num_workers=2)

CKPT_T2_CLF.mkdir(parents=True, exist_ok=True)
ckpt_clf  = CKPT_T2_CLF / "checkpoint.pt"
start_ep  = 0
if ckpt_clf.exists():
    s = torch.load(str(ckpt_clf), map_location=DEVICE)
    model_clf.load_state_dict(s["model"])
    opt_clf.load_state_dict(s["optimiser"])
    start_ep = s["epoch"] + 1
    print(f"▶ Resumed classifier from epoch {s['epoch']}")

run_clf = wandb.init(project=WB_PROJECT, name="task2-classifier",
                     config=best_clf, tags=["task2","classifier"], reinit=True)
best_clf_acc = 0.0
for epoch in range(start_ep, T2_CLF_EPOCHS):
    tr_loss, tr_acc = train_clf_epoch(model_clf, train_ld_clf, opt_clf, scaler_clf, DEVICE)
    val_acc         = val_clf_epoch(model_clf, val_ld_clf, DEVICE)
    sch_clf.step()
    is_best = val_acc > best_clf_acc
    if is_best: best_clf_acc = val_acc
    wandb.log({"train/loss": tr_loss, "train/acc": tr_acc, "val/acc": val_acc, "epoch": epoch})
    state = {"epoch": epoch, "model": model_clf.state_dict(),
             "optimiser": opt_clf.state_dict(), "metrics": {"val_acc": val_acc},
             "config": best_clf}
    torch.save(state, str(ckpt_clf))
    if is_best: torch.save(state, str(CKPT_T2_CLF / "best.pt"))
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{T2_CLF_EPOCHS}  tr_loss={tr_loss:.4f}  "
              f"tr_acc={tr_acc:.3f}  val_acc={val_acc:.3f}  {'★' if is_best else ''}")
wandb.finish()

# ── Full test-set evaluation ────────────────────────────────────────────────
model_clf.eval()
test_ld_clf = DataLoader(PetDataset(test_npy, manifest=test_manifest),
                          batch_size=64, shuffle=False, num_workers=2)
all_preds, all_labels = [], []
with torch.no_grad():
    for corrupted, _, labels in test_ld_clf:
        with autocast():
            logits = model_clf(corrupted.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

print("\n─── Classifier Metrics ───")
print(classification_report(all_labels, all_preds,
                             target_names=CORRUPTION_TYPES, digits=4))

cm = confusion_matrix(all_labels, all_preds, normalize="true")
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CORRUPTION_TYPES, yticklabels=CORRUPTION_TYPES, ax=ax)
ax.set_title("Task 2 – Normalised Confusion Matrix"); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
cm_path = str(BASE_DIR / "task2_confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.show()
print(f"✓ Confusion matrix saved → {cm_path}")

run_cm = wandb.init(project=WB_PROJECT, name="task2-clf-eval",
                    tags=["task2","evaluation"], reinit=True)
wandb.log({"task2/confusion_matrix": wandb.Image(cm_path),
           "task2/test_accuracy": accuracy_score(all_labels, all_preds)})
wandb.finish()

# %% CELL 23 ─────────────────────────── Task 2: Specialist AE – shared Optuna
# Uses a MIXED corruption dataset (picks all 3 types) for architecture search.
# Expected runtime: 15 trials × ~6 min = ~1.5 hours

STORAGE_SPEC = f"sqlite:///{OPTUNA_DIR}/task2_specialist.db"

def spec_objective(trial):
    lr            = trial.suggest_float("lr",            1e-4, 1e-2,  log=True)
    batch_size    = trial.suggest_categorical("batch_size",    [16, 32, 64])
    bottleneck_dim = trial.suggest_int("bottleneck_dim",       64, 512, step=64)
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
    alpha         = trial.suggest_float("alpha",         0.50, 0.95, step=0.05)

    # Architecture search: use a random subset of salt+blur+occlusion only
    class SpecDataset(PetDataset):
        def __getitem__(self, idx):
            c, x, l = super().__getitem__(idx)
            # resample until not clean (for architecture search we want non-clean)
            if l == 0:
                rng = np.random.default_rng(idx + 10000)
                img = np.load(self.npy_paths[idx])
                nc  = str(rng.choice(["salt_pepper","blur","occlusion"]))
                sev = _sample_severity(nc, rng)
                cim = corrupt_from_severity(img, nc, sev, rng)
                c   = torch.from_numpy(cim).permute(2,0,1).float()/255.0
                l   = LABEL_TO_IDX[nc]
            return c, x, l

    model     = UniversalAE(base_channels, bottleneck_dim, 0.0).to(DEVICE)
    criterion = ReconLoss(alpha)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = GradScaler()

    ds        = SpecDataset(train_npy[:len(train_npy)//2])
    ld        = DataLoader(ds, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    val_ds    = PetDataset(val_npy, manifest=val_manifest)
    val_ld    = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    best_ssim = 0.0
    for epoch in range(5):
        train_one_epoch_t1(model, ld, criterion, optimiser, scaler)
        _, val_ssim = val_one_epoch_t1(model, val_ld, criterion)
        trial.report(val_ssim, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        best_ssim = max(best_ssim, val_ssim)
    return best_ssim

study_spec = optuna.create_study(
    study_name="task2_specialist_arch",
    direction="maximize",
    storage=STORAGE_SPEC,
    load_if_exists=True,
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=2),
)
print(f"▶ Specialist Optuna (already have {len(study_spec.trials)} trials)")
study_spec.optimize(spec_objective, n_trials=15, show_progress_bar=True)

best_spec = study_spec.best_params
print("\n─── Specialist Architecture Best Config ───")
print(f"  SSIM: {study_spec.best_value:.4f}")
for k, v in best_spec.items(): print(f"  {k}: {v}")

# %% CELL 24 ─────────────────────────── Task 2: Train 3 specialists independently
# Expected runtime: ~3 epochs × 20 epochs = ~3 hours total

def train_specialist(corruption_type, best_spec, epochs=20):
    """
    Train one specialist AE on its specific corruption type only.
    Returns the trained model.
    """
    print(f"\n─── Training {corruption_type} Specialist ───")
    # Dataset: only images with the target corruption type
    class SingleCorrDataset(Dataset):
        def __init__(self, npy_paths, ctype):
            self.paths = npy_paths
            self.ctype = ctype
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = np.load(self.paths[idx])
            rng = np.random.default_rng()
            sev = _sample_severity(self.ctype, rng)
            cor = corrupt_from_severity(img, self.ctype, sev, rng)
            c_t = torch.from_numpy(cor).permute(2,0,1).float()/255.0
            x_t = torch.from_numpy(img).permute(2,0,1).float()/255.0
            return c_t, x_t, LABEL_TO_IDX[self.ctype]

    model     = UniversalAE(best_spec["base_channels"],
                            best_spec["bottleneck_dim"], 0.0).to(DEVICE)
    criterion = ReconLoss(best_spec["alpha"])
    optimiser = torch.optim.AdamW(model.parameters(),
                                   lr=best_spec["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs)
    scaler    = GradScaler()
    bs        = best_spec["batch_size"]

    train_ld  = DataLoader(SingleCorrDataset(train_npy, corruption_type),
                            batch_size=bs, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True)
    val_ld    = DataLoader(PetDataset(val_npy, manifest=val_manifest),
                            batch_size=bs, shuffle=False, num_workers=2)

    ckpt_dir  = CKPT_T2_SPEC / corruption_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_f    = ckpt_dir / "checkpoint.pt"
    start_ep  = 0
    if ckpt_f.exists():
        s = torch.load(str(ckpt_f), map_location=DEVICE)
        model.load_state_dict(s["model"])
        optimiser.load_state_dict(s["optimiser"])
        start_ep = s["epoch"] + 1
        print(f"  Resumed from epoch {s['epoch']}")

    run = wandb.init(project=WB_PROJECT, name=f"task2-specialist-{corruption_type}",
                     config=best_spec, tags=["task2","specialist", corruption_type],
                     reinit=True)
    best_ssim = 0.0
    for epoch in range(start_ep, epochs):
        train_one_epoch_t1(model, train_ld, criterion, optimiser, scaler)
        val_loss, val_ssim = val_one_epoch_t1(model, val_ld, criterion)
        scheduler.step()
        is_best = val_ssim > best_ssim
        if is_best: best_ssim = val_ssim
        wandb.log({"val/ssim": val_ssim, "val/loss": val_loss, "epoch": epoch})
        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimiser": optimiser.state_dict(),
                 "metrics": {"val_ssim": val_ssim}, "config": best_spec}
        torch.save(state, str(ckpt_f))
        if is_best: torch.save(state, str(ckpt_dir / "best.pt"))
        if (epoch+1) % 5 == 0:
            print(f"  {corruption_type} | epoch {epoch+1}/{epochs}  ssim={val_ssim:.4f}  {'★' if is_best else ''}")
    wandb.finish()
    print(f"  Best SSIM: {best_ssim:.4f}")
    return model

model_salt = train_specialist("salt_pepper", best_spec, epochs=20)
model_blur = train_specialist("blur",        best_spec, epochs=20)
model_occ  = train_specialist("occlusion",   best_spec, epochs=20)
print("✓ All 3 specialists trained")

# %% CELL 25 ─────────────────────────── Task 2: Oracle vs predicted routing evaluation

from sklearn.metrics import accuracy_score

def hard_route_infer(corrupted_t, model_clf, model_salt, model_blur, model_occ,
                     routing_label=None):
    """
    Perform hard routing.
    routing_label: if given, use oracle routing (ground-truth label).
    Returns (x_hat, probs, routed_class)
    """
    with torch.no_grad():
        with autocast():
            probs = model_clf.predict_proba(corrupted_t)   # (B, 4)
    if routing_label is not None:
        routed = routing_label                             # oracle
    else:
        routed = int(probs.argmax(1).item())               # predicted

    ctype = IDX_TO_LABEL[routed]
    with torch.no_grad():
        with autocast():
            if ctype == "clean":
                x_hat = corrupted_t                        # identity bypass
            elif ctype == "salt_pepper":
                x_hat = model_salt(corrupted_t)
            elif ctype == "blur":
                x_hat = model_blur(corrupted_t)
            else:
                x_hat = model_occ(corrupted_t)
    return x_hat, probs, routed

# Evaluate on test manifest
model_clf.eval(); model_salt.eval(); model_blur.eval(); model_occ.eval()

oracle_ssim    = defaultdict(list)
predicted_ssim = defaultdict(list)
routing_errors = []

for rec in tqdm(test_manifest[:500], desc="Routing eval"):   # subset for speed
    img_np   = np.load(rec["image_path"])
    corr_np  = corrupt_from_record(img_np, rec)
    clean_t  = torch.from_numpy(img_np).permute(2,0,1).float().unsqueeze(0)/255.0
    corr_t   = torch.from_numpy(corr_np).permute(2,0,1).float().unsqueeze(0)/255.0

    true_label = LABEL_TO_IDX[rec["corruption_type"]]
    corr_dev   = corr_t.to(DEVICE)

    # Oracle routing
    out_oracle, _, _  = hard_route_infer(corr_dev, model_clf, model_salt, model_blur, model_occ,
                                         routing_label=true_label)
    oracle_ssim[rec["corruption_type"]].append(ssim_val(out_oracle.cpu(), clean_t).item())

    # Predicted routing
    out_pred, probs, routed = hard_route_infer(corr_dev, model_clf, model_salt, model_blur, model_occ)
    predicted_ssim[rec["corruption_type"]].append(ssim_val(out_pred.cpu(), clean_t).item())

    if routed != true_label and rec["corruption_type"] != "clean":
        routing_errors.append({
            "true": CORRUPTION_TYPES[true_label],
            "predicted": CORRUPTION_TYPES[routed],
            "ssim_oracle": oracle_ssim[rec["corruption_type"]][-1],
            "ssim_predicted": predicted_ssim[rec["corruption_type"]][-1],
            "image_path": rec["image_path"],
        })

print("\n─── Routing Evaluation (SSIM) ───")
print(f"{'Type':<15} {'Oracle':>8} {'Predicted':>10} {'Delta':>7}")
for ctype in CORRUPTION_TYPES:
    o = np.mean(oracle_ssim[ctype])    if oracle_ssim[ctype]    else 0
    p = np.mean(predicted_ssim[ctype]) if predicted_ssim[ctype] else 0
    print(f"  {ctype:<13} {o:>8.4f} {p:>10.4f} {p-o:>+7.4f}")

print(f"\n  Routing errors: {len(routing_errors)}"
      f" / {len([r for r in test_manifest[:500] if r['corruption_type']!='clean'])}")
if routing_errors:
    print("\n  5 worst routing failures (oracle SSIM - predicted SSIM):")
    routing_errors.sort(key=lambda e: e["ssim_oracle"] - e["ssim_predicted"], reverse=True)
    for e in routing_errors[:5]:
        print(f"    {e['true']} → {e['predicted']}  "
              f"oracle_ssim={e['ssim_oracle']:.3f}  pred_ssim={e['ssim_predicted']:.3f}")

# %% CELL 26 ─────────────────────────── Task 2: ONNX export (classifier + 3 specialists)

def export_model_onnx(model, path, name):
    model.eval()
    dummy = torch.randn(1, 3, 128, 128, device=DEVICE)
    torch.onnx.export(model, dummy, path,
        input_names=["image"], output_names=["output"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}})
    onnx.checker.check_model(path)
    print(f"✓ {name} → {path}")
    # Quick parity check (2 samples)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng  = np.random.default_rng(99)
    for _ in range(4):
        x_np  = rng.random((1, 3, 128, 128)).astype(np.float32)
        pt_out = model(torch.from_numpy(x_np).to(DEVICE)).cpu().detach().numpy()
        ort_out = sess.run(None, {"image": x_np})[0]
        assert np.allclose(pt_out, ort_out, atol=1e-4), f"{name} parity check FAILED"
    print(f"  ✓ Parity OK (atol=1e-4)")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
export_model_onnx(model_clf,  str(MODELS_DIR/"task2_classifier.onnx"), "Classifier")
export_model_onnx(model_salt, str(MODELS_DIR/"task2_salt.onnx"),       "Salt specialist")
export_model_onnx(model_blur, str(MODELS_DIR/"task2_blur.onnx"),       "Blur specialist")
export_model_onnx(model_occ,  str(MODELS_DIR/"task2_occlusion.onnx"),  "Occlusion specialist")
print("\n✓ Task 2 ONNX export complete")

# %% CELL 27 ─────────────────────────── MILESTONE 1 EVIDENCE CHECKLIST
# Copy this output into your lab notebook before saying "MILESTONE 1 DONE"

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              MILESTONE 1 – EVIDENCE CHECKLIST                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  DATA                                                                ║
║  ☐ data/manifests/split.json         (train/val/test paths, seed=42)║
║  ☐ data/manifests/val_manifest.json  (1 record/image)               ║
║  ☐ data/manifests/test_manifest.json (10 records/image)             ║
║  ☐ Screenshot: manifest determinism test PASSED                      ║
║  ☐ Screenshot: corruption unit test PASSED                           ║
║                                                                      ║
║  TASK 1 – Universal AE                                               ║
║  ☐ Screenshot: Optuna search space + n_trials + best_trial table    ║
║  ☐ W&B: training/val loss curve (30 epochs)                         ║
║  ☐ W&B: val SSIM curve                                              ║
║  ☐ Figure: 12-example reconstruction grid (clean|corrupt|recon|err) ║
║  ☐ Figure: 4 failure cases with discussion                          ║
║  ☐ Table: SSIM + L1 per corruption type × severity level            ║
║  ☐ File: task1_universal.onnx on Drive                              ║
║  ☐ Screenshot: ONNX parity "all 16 samples within atol=1e-4"        ║
║                                                                      ║
║  TASK 2 – Hard Routing                                               ║
║  ☐ Screenshot: Classifier Optuna + best_trial                       ║
║  ☐ Figure: normalised 4×4 confusion matrix                          ║
║  ☐ Table: accuracy, macro P/R/F1, per-class metrics                 ║
║  ☐ Screenshot: Specialist Optuna + best architecture config          ║
║  ☐ Table: Oracle vs Predicted routing SSIM per corruption type      ║
║  ☐ List: ≥5 routing error examples with discussion                  ║
║  ☐ Files: task2_{classifier,salt,blur,occlusion}.onnx on Drive      ║
║  ☐ Screenshot: all 4 ONNX parity checks PASSED                      ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  VIVA NOTES – 8 likely questions (Task 1)                           ║
║                                                                      ║
║  Q1: Why no skip connections?                                        ║
║      A: Spec mandates genuine compressed bottleneck; unrestricted    ║
║         skip paths would let the decoder bypass the bottleneck and   ║
║         trivially copy encoder activations, violating the autoencoder║
║         definition.  (Vincent et al., 2010, JMLR)                   ║
║                                                                      ║
║  Q2: Why L1 + SSIM instead of just MSE?                             ║
║      A: MSE minimises pixel-wise squared error, producing blurry     ║
║         outputs.  L1 is sharper and more outlier-robust.  SSIM      ║
║         preserves structure, luminance, and contrast – perceptually  ║
║         important for images.  (Zhao et al., 2017, IEEE Trans CI)    ║
║                                                                      ║
║  Q3: How does Optuna's MedianPruner work?                            ║
║      A: After n_startup_trials it prunes a trial at step s if its   ║
║         intermediate value is below the median of all other trials   ║
║         at the same step.  Reduces wasted computation on bad HPs.   ║
║                                                                      ║
║  Q4: What does the bottleneck dimension control?                     ║
║      A: It controls how much the latent is compressed relative to   ║
║         the spatial feature volume (8C×8×8).  Smaller = more        ║
║         compression = more regularisation but risk of info loss.    ║
║                                                                      ║
║  Q5: Why AdamW and not Adam?                                         ║
║      A: AdamW correctly decouples weight decay from the adaptive     ║
║         gradient update (Loshchilov & Hutter, 2019).  Plain Adam    ║
║         conflates L2 regularisation with momentum, under-regularising║
║         in practice.                                                 ║
║                                                                      ║
║  Q6: How is AMP used here?                                           ║
║      A: torch.cuda.amp.autocast() casts eligible ops to float16,   ║
║         GradScaler prevents underflow in the backward pass.  Gives   ║
║         ~1.5× speedup on T4 with ~same convergence.                 ║
║                                                                      ║
║  Q7: How did you verify corruption determinism?                      ║
║      A: Re-ran generate_val_manifest and generate_test_manifest      ║
║         with identical seeds; compared SHA-256 hashes of JSON       ║
║         outputs – both matched.  Also re-applied corrupt_from_record ║
║         twice per record and confirmed pixel-level equality.         ║
║                                                                      ║
║  Q8: Why ConvTranspose2d in decoder rather than bilinear upsample?  ║
║      A: ConvTranspose2d learns upsampling filters jointly with the  ║
║         rest of the network – can produce sharper reconstructions.   ║
║         (Zeiler & Fergus, 2014, ECCV)                               ║
║                                                                      ║
║  VIVA NOTES – 5 likely questions (Task 2)                           ║
║                                                                      ║
║  Q9: Why use WeightedRandomSampler for the classifier?               ║
║      A: Without balancing, ~25% clean images dominate training;      ║
║         the model becomes biased toward the majority class.  WRS     ║
║         over-samples minority classes so each minibatch has ~equal   ║
║         class frequencies.                                           ║
║                                                                      ║
║  Q10: What is oracle routing and why run it?                         ║
║       A: Oracle routing uses the ground-truth corruption label       ║
║          (from the manifest) to select the expert – showing the      ║
║          maximum achievable performance if classification were        ║
║          perfect.  The gap between oracle and predicted routing is   ║
║          entirely due to classifier errors.                          ║
║                                                                      ║
║  Q11: Why train specialists on a single corruption type?             ║
║       A: A model trained only on salt-and-pepper can memorise the   ║
║          specific statistical structure of that noise and learn the  ║
║          exact inverse mapping, outperforming a universal model on   ║
║          that type.                                                  ║
║                                                                      ║
║  Q12: Why share the Optuna search for specialist architecture?       ║
║       A: Running 3 separate 15-trial studies would triple compute.   ║
║          The optimal architecture depends mainly on image statistics  ║
║          (128×128 RGB), not corruption type – only the learned       ║
║          weights differ.                                             ║
║                                                                      ║
║  Q13: What ONNX opset version did you use and why?                   ║
║       A: Opset 17 (current stable).  All ops used (Conv, BN, ReLU, ║
║          Linear, ConvTranspose2d) are supported at opset 17.         ║
║          Dynamic axes allow variable batch size.                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

  After collecting all evidence above, reply: MILESTONE 1 DONE
""")

# Common errors & fixes are printed at the end for quick reference
print("""
─── COMMON ERRORS & FIXES ──────────────────────────────────────────────────

  ERROR: "CUDA out of memory during Optuna"
  FIX:   Reduce batch_size choices to [8, 16, 32], or add:
         torch.cuda.empty_cache() at the start of each trial.

  ERROR: "RuntimeError: Expected all tensors to be on the same device"
  FIX:   Add .to(DEVICE) to any tensor before feeding to model.

  ERROR: "Session expired" (Colab disconnects mid-training)
  FIX:   Checkpoint is saved every epoch to Drive; re-run the cell –
         load_checkpoint() will resume from last epoch automatically.
         Optuna SQLite on Drive is also persistent across sessions.

  ERROR: "torch.onnx.export: TracerWarning: Converting a tensor to..."
  FIX:   Replace Python int/bool control flow in forward() with tensor ops.
         The current model has no such flow – warning is safe to ignore.

  ERROR: "ONNX parity check FAILED (max_diff > 1e-4)"
  FIX:   Re-run export with opset_version=16 (fallback).  If persists,
         check for in-place ops that conflict with ONNX tracing.

  ERROR: "val_manifest.json not found"
  FIX:   Re-run Cell 9 (generate_val_manifest).  Manifests are small
         JSON files; commit them to git so they survive session resets.
""")
