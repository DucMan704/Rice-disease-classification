#!/usr/bin/env python
"""Builds the final cross-model comparison table + chart from whatever
outputs/metrics/*_metrics.json files exist so far."""
import sys
sys.path.insert(0, "..")
sys.path.insert(0, ".")

import config
from src.compare import build_comparison

if __name__ == "__main__":
    build_comparison(config.MODEL_NAMES)
