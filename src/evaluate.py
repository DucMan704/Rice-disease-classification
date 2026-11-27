"""
src/evaluate.py
=================
Computes and saves everything the project spec asks for on the test set:
accuracy / precision / recall / F1, confusion matrix, ROC curves
(one-vs-rest per class + macro/micro average), and a grid of correct vs.
incorrect prediction examples.
"""

import os

import numpy as np

import config
from src.utils import get_logger, save_json

logger = get_logger("evaluate")


def _predict_all(model, test_ds):
    y_true, y_prob = [], []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_prob.append(preds)
        y_true.append(batch_y.numpy())
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def evaluate_model(model_name, model, test_ds, class_names):
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report, confusion_matrix,
    )

    y_true, y_pred, y_prob = _predict_all(model, test_ds)

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "n_test_samples": int(len(y_true)),
        "classification_report": report_dict,
    }

    metrics_path = os.path.join(config.DIRS["metrics"], f"{model_name}_metrics.json")
    save_json(metrics, metrics_path)

    report_path = os.path.join(config.DIRS["metrics"], f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report_text)

    logger.info(
        f"[{model_name}] TEST  acc={acc:.4f}  "
        f"precision(macro)={precision_macro:.4f}  recall(macro)={recall_macro:.4f}  f1(macro)={f1_macro:.4f}"
    )

    _plot_confusion_matrix(model_name, y_true, y_pred, class_names)
    _plot_roc_curves(model_name, y_true, y_prob, class_names)

    return metrics, y_true, y_pred, y_prob


def _plot_confusion_matrix(model_name, y_true, y_pred, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 3, 1.1 * len(class_names) + 2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (row-normalised) - {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                    ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = os.path.join(config.DIRS["plots_cm"], f"{model_name}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"[{model_name}] confusion matrix saved -> {out_path}")


def _plot_roc_curves(model_name, y_true, y_prob, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        # label_binarize collapses to a single column for binary case
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, cls in enumerate(class_names):
        ax.plot(fpr[i], tpr[i], lw=1.3, label=f"{cls} (AUC={roc_auc[i]:.3f})")
    ax.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", lw=2.5,
            label=f"micro-avg (AUC={roc_auc['micro']:.3f})")
    ax.plot(fpr["macro"], tpr["macro"], color="navy", linestyle=":", lw=2.5,
            label=f"macro-avg (AUC={roc_auc['macro']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (one-vs-rest) - {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(config.DIRS["plots_roc"], f"{model_name}_roc_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"[{model_name}] ROC curves saved -> {out_path}")


def plot_prediction_examples(model_name, test_files_labels, y_true, y_pred, class_names, n=config.PRED_EXAMPLES_GRID):
    """Grid of sample test images with true vs predicted label, green/red
    coded for correct/incorrect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    idx = np.arange(len(test_files_labels))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(idx)
    idx = idx[:n]

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for ax_i, i in enumerate(idx):
        ax = axes[ax_i]
        path, _ = test_files_labels[i]
        true_name = class_names[y_true[i]]
        pred_name = class_names[y_pred[i]]
        correct = true_name == pred_name
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "image\nunavailable", ha="center", va="center")
        ax.set_title(f"true: {true_name}\npred: {pred_name}",
                     color="green" if correct else "red", fontsize=9)
        ax.axis("off")

    for ax_i in range(len(idx), len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(f"Prediction examples - {model_name}")
    fig.tight_layout()
    out_path = os.path.join(config.DIRS["plots_pred"], f"{model_name}_prediction_examples.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"[{model_name}] prediction examples saved -> {out_path}")
