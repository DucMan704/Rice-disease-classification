#!/usr/bin/env python
"""Trains, evaluates and Grad-CAMs a single model: DenseNet121.
Can be run standalone (e.g. python scripts/04_train_densenet121.py) or is called
automatically, in order, by run_all.py."""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from src.pipeline import prepare_data, train_single_model, check_environment

if __name__ == "__main__":
    check_environment()
    train_ds, val_ds, test_ds, class_names, test_files_labels = prepare_data()
    train_single_model("DenseNet121", train_ds, val_ds, test_ds, class_names, test_files_labels)
