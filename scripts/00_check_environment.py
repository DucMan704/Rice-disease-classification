#!/usr/bin/env python
"""Run first (also called automatically by run_all.py). Verifies TF/GPU and
that the dataset can actually be found before anything else happens."""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from src.pipeline import check_environment
from src.data_utils import build_or_load_manifest

if __name__ == "__main__":
    check_environment()
    split = build_or_load_manifest()
    print("Classes:", split["class_names"])
    print("train/val/test sizes:", len(split["train"]), len(split["val"]), len(split["test"]))
