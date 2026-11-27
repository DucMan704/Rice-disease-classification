#!/usr/bin/env python
"""
run_all.py
===========
THE single entry point for this project.

    python run_all.py

runs, in order:
  1. environment check (TF version, GPU, output dirs)
  2. dataset discovery + reproducible stratified train/val/test split
  3. for each of the 5 models: train (2-phase transfer learning) -> evaluate
     -> confusion matrix -> ROC curves -> prediction examples -> Grad-CAM
     (each model is wrapped in try/except so one failure does not stop
     the rest of the pipeline)
  4. final cross-model comparison table + chart

Safe to re-run: any model that already has saved metrics is skipped, so you
can resume a partially completed run (e.g. after a Kaggle session timeout)
without redoing finished work. Pass --force to retrain everything anyway.
"""

import argparse
import sys

sys.path.insert(0, ".")

import config
from src.utils import get_logger, ensure_all_dirs

logger = get_logger("run_all")


def maybe_install_requirements():
    """Kaggle images already ship every package this project needs
    (tensorflow, scikit-learn, matplotlib, pillow, pandas, numpy). This is
    just a defensive check for anyone running outside Kaggle."""
    required = ["tensorflow", "sklearn", "matplotlib", "PIL", "numpy", "pandas"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        logger.warning(f"Missing packages detected: {missing}. Attempting pip install from requirements.txt ...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)


def main():
    parser = argparse.ArgumentParser(description="Run the full 5-model rice-disease training pipeline.")
    parser.add_argument("--models", nargs="+", default=config.MODEL_NAMES,
                         help=f"Subset of models to run. Default: all of {config.MODEL_NAMES}")
    parser.add_argument("--force", action="store_true",
                         help="Retrain even if metrics already exist for a model.")
    args = parser.parse_args()

    ensure_all_dirs()
    maybe_install_requirements()

    logger.info("=" * 70)
    logger.info("RICE LEAF DISEASE CLASSIFICATION - FULL TRAINING PIPELINE")
    logger.info(f"Models to run: {args.models}")
    logger.info(f"Outputs will be written to: {config.OUTPUT_ROOT}")
    logger.info("=" * 70)

    from src.pipeline import run_full_pipeline
    results = run_full_pipeline(model_names=args.models, force=args.force)

    failed = [m for m, s in results.items() if s != "OK"]
    if failed:
        logger.warning(f"Finished with {len(failed)} failed model(s): {failed}. "
                        f"Check outputs/logs/ and outputs/metrics/*_FAILED.json for details.")
        sys.exit(1 if len(failed) == len(results) else 0)
    else:
        logger.info("All models completed successfully. See outputs/comparison/ for the final summary.")


if __name__ == "__main__":
    main()
