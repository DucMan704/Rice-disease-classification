"""
src/train.py
=============
Generic two-phase transfer-learning training routine used for all 5 models:

  Phase 1 (head-only):   backbone frozen, train the new classification head.
  Phase 2 (fine-tune):   unfreeze the top fraction of the backbone and train
                          the whole thing at a low learning rate.

Produces: trained model, combined history (csv + json), training-curve plot.
"""

import os
import csv
import json

import config
from src.utils import get_logger, Timer, ensure_all_dirs
from src.models import unfreeze_top_fraction

logger = get_logger("train")


def _callbacks(model_name, checkpoint_path, monitor="val_accuracy"):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    return [
        EarlyStopping(
            monitor=monitor, mode="max", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path, monitor=monitor, mode="max",
            save_best_only=True, save_weights_only=False, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=1e-8, verbose=1,
        ),
    ]


def _history_to_rows(history_dict, phase_label, epoch_offset):
    rows = []
    n = len(next(iter(history_dict.values())))
    for i in range(n):
        row = {"epoch": epoch_offset + i + 1, "phase": phase_label}
        for k, v in history_dict.items():
            row[k] = v[i]
        rows.append(row)
    return rows


def train_model(model_name, model, base_model, train_ds, val_ds):
    """Runs phase 1 + phase 2 training. Returns (model, history_rows list-of-dict)."""
    from tensorflow import keras

    ensure_all_dirs()
    ckpt_path = os.path.join(config.DIRS["models"], f"{model_name}_best.keras")
    final_path = os.path.join(config.DIRS["models"], f"{model_name}_final.keras")

    all_rows = []

    # ---------------- Phase 1: frozen backbone, train head ----------------
    with Timer(logger, f"[{model_name}] phase 1 (head training, {config.EPOCHS_HEAD} epochs)"):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR_HEAD),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS_HEAD,
            callbacks=_callbacks(model_name, ckpt_path),
            verbose=2,
        )
        all_rows += _history_to_rows(hist1.history, "head", 0)

    # ---------------- Phase 2: unfreeze top layers, fine-tune -------------
    n_unfrozen, n_total = unfreeze_top_fraction(base_model)
    logger.info(f"[{model_name}] fine-tune phase: unfroze {n_unfrozen}/{n_total} backbone layers")

    with Timer(logger, f"[{model_name}] phase 2 (fine-tune, {config.EPOCHS_FINE_TUNE} epochs)"):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR_FINE_TUNE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS_FINE_TUNE,
            callbacks=_callbacks(model_name, ckpt_path),
            verbose=2,
        )
        all_rows += _history_to_rows(hist2.history, "fine_tune", config.EPOCHS_HEAD)

    # model already holds the best weights (restore_best_weights=True in both phases)
    model.save(final_path)
    logger.info(f"[{model_name}] saved final model -> {final_path}")
    logger.info(f"[{model_name}] best checkpoint -> {ckpt_path}")

    _save_history(model_name, all_rows)
    _plot_curves(model_name, all_rows)

    return model, all_rows


def _save_history(model_name, rows):
    csv_path = os.path.join(config.DIRS["history"], f"{model_name}_history.csv")
    json_path = os.path.join(config.DIRS["history"], f"{model_name}_history.json")

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    logger.info(f"[{model_name}] history saved -> {csv_path}")


def _plot_curves(model_name, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    acc = [r.get("accuracy") for r in rows]
    val_acc = [r.get("val_accuracy") for r in rows]
    loss = [r.get("loss") for r in rows]
    val_loss = [r.get("val_loss") for r in rows]
    phase_boundary = config.EPOCHS_HEAD + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, acc, label="train")
    axes[0].plot(epochs, val_acc, label="val")
    axes[0].axvline(phase_boundary, color="gray", linestyle="--", linewidth=1, label="fine-tune start")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, loss, label="train")
    axes[1].plot(epochs, val_loss, label="val")
    axes[1].axvline(phase_boundary, color="gray", linestyle="--", linewidth=1, label="fine-tune start")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Training curves - {model_name}")
    fig.tight_layout()

    out_path = os.path.join(config.DIRS["plots_curves"], f"{model_name}_training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"[{model_name}] training curves saved -> {out_path}")
