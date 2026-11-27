#!/usr/bin/env python
"""Builds (or reuses) the reproducible stratified train/val/test split."""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from src.pipeline import prepare_data

if __name__ == "__main__":
    prepare_data()
