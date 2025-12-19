"""
Unified Data Module

Includes:
- Dataset download & split fixing
- Duplicate/leakage analysis
- Preprocessing & encoding
- NPY-backed PyTorch Dataset

Design assumptions:
- Dataset may contain duplicates across splits
- Validation split may be unreliable or undersized
- Training data should always dominate split priority
- Exclusion is non-destructive (logical, not physical deletion)
- Dataset < 2GB, CPU-only training
"""

# =========================
# Standard Library
# =========================
import os
import shutil
from pathlib import Path
from collections import defaultdict

# =========================
# Third Party
# =========================
import yaml
import cv2
import kagglehub
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

# =========================
# Local Utilities
# =========================
from utils import FileUtils

# =========================
# Constants
# =========================
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ============================================================
# Dataset Manager
# ============================================================
class CXRDatasetManager:
    """Handles dataset download, physical layout, and split fixing."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.splits = ["train", "val", "test"]

    def _format_path(self, path):
        try:
            return f".\\{os.path.relpath(path, start=os.getcwd())}"
        except ValueError:
            return str(path)

    def exists(self):
        return all((self.data_dir / s).exists() for s in self.splits)

    def download(self, kaggle_id: str):
        display_path = self._format_path(self.data_dir)

        if self.exists():
            print(f"✅ Dataset found at: {display_path}")
            return

        print(f"⬇️ Downloading {kaggle_id}...")
        cache_path = kagglehub.dataset_download(kaggle_id)
        cache_root = Path(cache_path)

        inner_root = cache_root
        for root, dirs, _ in os.walk(cache_root):
            if "train" in dirs:
                inner_root = Path(root)
                break

        print(f"📦 Moving data to {display_path}...")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        for split in self.splits:
            shutil.copytree(
                inner_root / split,
                self.data_dir / split,
                dirs_exist_ok=True
            )

    def fix_splits(self):
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        if not val_dir.exists() or not test_dir.exists():
            return

        count = lambda d: sum(
            1 for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )

        num_val = count(val_dir)
        num_test = count(test_dir)

        if num_val < 50 and num_test > 100:
            print(f"⚠️ Swapping VAL ({num_val}) with TEST ({num_test})")
            tmp = self.data_dir / "_swap_tmp"
            val_dir.rename(tmp)
            test_dir.rename(val_dir)
            tmp.rename(test_dir)
            print("🔄 Swap complete.")


# ============================================================
# Analyzer (Duplicates / Leakage)
# ============================================================
class CXRAnalyzer:
    """Pure analysis layer."""

    def __init__(self, data_dir: Path):
        self.root = data_dir
        self.splits = ["train", "val", "test"]
        self.excluded_files = set()

    def build_exclusion_registry(self):
        print("🔍 Checking for duplicates/leakage...")
        hashes = {}
        all_images = []

        for split in self.splits:
            split_dir = self.root / split
            if split_dir.exists():
                all_images.extend(split_dir.rglob("*"))

        for img in tqdm(all_images, desc="Hashing"):
            if FileUtils.is_image(img):
                h = FileUtils.phash(img)
                if h:
                    hashes[str(img)] = h

        hash_groups = defaultdict(list)
        for path, h in hashes.items():
            hash_groups[str(h)].append(path)

        for paths in hash_groups.values():
            if len(paths) > 1:
                self._resolve_conflict(paths)

        if self.excluded_files:
            print(f"🚫 Excluded {len(self.excluded_files)} duplicates")
        else:
            print("✨ Dataset is clean")

    def _resolve_conflict(self, paths):
        def priority(p):
            p = p.replace("\\", "/")
            for split, prio in SPLIT_PRIORITY.items():
                if f"/{split}/" in p:
                    return prio
            return 99

        paths.sort(key=priority)
        for loser in paths[1:]:
            self.excluded_files.add(loser)

    def iter_valid_images(self, split):
        split_dir = self.root / split
        if not split_dir.exists():
            return
        for p in split_dir.rglob("*"):
            if FileUtils.is_image(p) and str(p) not in self.excluded_files:
                yield p


# ============================================================
# Preprocessing Pipeline
# ============================================================
class Preprocessor:
    def __init__(self, project_root):
        self.root = Path(project_root).resolve()
        self.config_path = self.root / "configs" / "config.yaml"

        if not self.config_path.exists():
            raise FileNotFoundError(f"❌ Missing config: {self.config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        raw_cfg = self.config["data"]["raw_path"]
        cand_1 = (self.root / raw_cfg).resolve()
        cand_2 = (self.root / "data" / "CXR").resolve()

        if cand_1.exists():
            self.raw_path = cand_1
        elif cand_2.exists():
            print(f"⚠️ Falling back to {cand_2}")
            self.raw_path = cand_2
        else:
            raise FileNotFoundError("❌ Dataset not found")

        self.proc_path = self.root / "data" / "processed"
        self.img_size = self.config["data"].get("img_size", 320)
        self.classes = self.config["data"]["classes"]

        print(f"🎯 Raw: {self.raw_path}")
        print(f"💾 Processed: {self.proc_path}")

        self.analyzer = CXRAnalyzer(self.raw_path)
        self.analyzer.build_exclusion_registry()

    def process_split(self, split):
        print(f"\n⚙️ Processing {split.upper()}")
        images, labels = [], []

        files = list(self.analyzer.iter_valid_images(split))
        if not files:
            print("⚠️ No valid files")
            return

        for img_path in tqdm(files, desc="Encoding", unit="img"):
            cls = img_path.parent.name
            if cls not in self.classes:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, 0)

            images.append(img)
            labels.append(self.classes[cls])

        if images:
            self.proc_path.mkdir(parents=True, exist_ok=True)
            X = np.array(images, np.float32)
            y = np.array(labels, np.int64)

            np.save(self.proc_path / f"X_{split}.npy", X)
            np.save(self.proc_path / f"y_{split}.npy", y)

            print(f"✅ Saved {X.shape}")

    def run(self):
        if self.proc_path.exists():
            print("🧹 Clearing old processed data")
            shutil.rmtree(self.proc_path)

        for split in ["train", "val", "test"]:
            self.process_split(split)


# ============================================================
# PyTorch Dataset
# ============================================================
class NpyDataset(Dataset):
    """Fast RAM-backed dataset."""

    def __init__(self, processed_path, split, transform=None):
        root = Path(processed_path)
        x_path = root / f"X_{split}.npy"
        y_path = root / f"y_{split}.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"❌ Missing {x_path}")

        print(f"⏳ Loading {split} into RAM...", end=" ")
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        print(f"Done {self.X.shape}")

        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.X[idx])
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
# ============================================================
