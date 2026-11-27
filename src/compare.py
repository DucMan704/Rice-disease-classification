"""
src/compare.py
================
Reads every outputs/metrics/<model>_metrics.json produced so far and builds
a single comparison table (CSV) plus a bar chart, so you can see at a glance
which of the 5 architectures performed best.
"""

import os
import csv

import config
from src.utils import get_logger, load_json

logger = get_logger("compare")


def build_comparison(model_names=None):
    model_names = model_names or config.MODEL_NAMES

    rows = []
    for name in model_names:
        path = os.path.join(config.DIRS["metrics"], f"{name}_metrics.json")
        if not os.path.exists(path):
            logger.warning(f"No metrics found for {name} (path missing: {path}) - skipping in comparison.")
            continue
        m = load_json(path)
        rows.append({
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "precision_macro": round(m["precision_macro"], 4),
            "recall_macro": round(m["recall_macro"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "precision_weighted": round(m["precision_weighted"], 4),
            "recall_weighted": round(m["recall_weighted"], 4),
            "f1_weighted": round(m["f1_weighted"], 4),
            "n_test_samples": m["n_test_samples"],
        })

    if not rows:
        logger.warning("No model metrics available yet - comparison skipped.")
        return None

    rows.sort(key=lambda r: r["f1_macro"], reverse=True)

    csv_path = os.path.join(config.DIRS["comparison"], "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Comparison table saved -> {csv_path}")

    _plot_comparison(rows)

    best = rows[0]
    logger.info(f"BEST MODEL (by macro F1): {best['model']} "
                f"(acc={best['accuracy']}, f1_macro={best['f1_macro']})")

    return rows


def _plot_comparison(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [r["model"] for r in rows]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        vals = [r[metric] for r in rows]
        ax.bar(x + i * width, vals, width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(config.DIRS["comparison"], "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Comparison chart saved -> {out_path}")
