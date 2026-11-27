#!/usr/bin/env python
"""Trains, evaluates and Grad-CAMs a single model: ConvNeXtTiny.
Can be run standalone (e.g. python scripts/06_train_convnexttiny.py) or is called
automatically, in order, by run_all.py."""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

from src.pipeline import prepare_data, train_single_model, check_environment

if __name__ == "__main__":
    check_environment()
    train_ds, val_ds, test_ds, class_names, test_files_labels = prepare_data()
    train_single_model("ConvNeXtTiny", train_ds, val_ds, test_ds, class_names, test_files_labels)
