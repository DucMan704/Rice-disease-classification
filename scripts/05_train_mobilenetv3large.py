#!/usr/bin/env python
"""Trains, evaluates and Grad-CAMs a single model: MobileNetV3Large.
Can be run standalone (e.g. python scripts/05_train_mobilenetv3large.py) or is called
automatically, in order, by run_all.py."""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from src.pipeline import prepare_data, train_single_model, check_environment

if __name__ == "__main__":
    check_environment()
    train_ds, val_ds, test_ds, class_names, test_files_labels = prepare_data()
    train_single_model("MobileNetV3Large", train_ds, val_ds, test_ds, class_names, test_files_labels)
