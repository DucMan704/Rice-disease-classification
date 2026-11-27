"""
src/utils.py
=============
Small, dependency-light helpers used across the pipeline:
logging setup, seeding, directory creation, safe json read/write, timers.
"""

import os
import json
import random
import logging
import time

import numpy as np

import config


def ensure_all_dirs():
    for path in config.DIRS.values():
        os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


_LOGGERS = {}


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to both stdout and a per-name log file under
    outputs/logs/. Safe to call repeatedly for the same name."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    ensure_all_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        log_path = os.path.join(config.DIRS["logs"], f"{name}.log")
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


class Timer:
    """Context manager for timing a block and logging it.

    with Timer(logger, "training EfficientNetB0"):
        ...
    """

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"START: {self.label}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        mins = elapsed / 60.0
        if exc_type is None:
            self.logger.info(f"DONE : {self.label} ({mins:.1f} min)")
        else:
            self.logger.error(f"FAILED: {self.label} after {mins:.1f} min -> {exc_val}")
        return False  # do not suppress exceptions
