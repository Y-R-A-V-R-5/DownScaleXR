#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing script for DownScaleXR (pre-split dataset).
Usage: python scripts/data_preprocessing.py
Features:
- Downloads dataset if missing
- Processes raw CXR dataset into numpy arrays
- Deduplicates invalid images
- Saves X_{split}.npy and y_{split}.npy
- Displays tqdm progress with emojis
"""

from pathlib import Path
from tqdm import tqdm
import sys

# ---- Single sys.path fix ----
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import Preprocessor

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw_data_path = project_root / "data" / "CXR"

    print("🛠️ Starting Data Preprocessing...")
    preprocessor = Preprocessor(project_root)

    # Preprocess each split
    for split in ["train", "val", "test"]:
        print(f"\n🔄 Processing split: {split.upper()} 🩺")
        files = list(preprocessor.analyzer.iter_valid_images(split))
        if not files:
            print(f"⚠️ No valid images in {split}")
            continue

        for _ in tqdm(range(1), desc=f"Encoding {split}", colour='cyan', ascii=True):
            preprocessor.process_split(split)

    print("\n✅ Data preprocessing complete!")