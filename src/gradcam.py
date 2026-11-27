"""
src/gradcam.py
================
Architecture-agnostic Grad-CAM. Instead of hard-coding a "last conv layer"
name per backbone (which changes between TF/Keras versions), we walk the
model's layers and automatically pick the last layer whose output is a
4D tensor (batch, H, W, C) - this works the same way for EfficientNet,
ResNet, DenseNet, MobileNetV3 and ConvNeXt without any per-model tuning.
"""

import os

import numpy as np

import config
from src.utils import get_logger

logger = get_logger("gradcam")


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("Could not find a 4D (conv-like) layer for Grad-CAM.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def _overlay_heatmap(img_rgb_uint8, heatmap, alpha=0.45):
    import matplotlib.cm as cm
    from PIL import Image

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (img_rgb_uint8.shape[1], img_rgb_uint8.shape[0])
    )
    heatmap_arr = np.array(heatmap_img)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_arr]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    overlay = np.uint8(img_rgb_uint8 * (1 - alpha) + jet_heatmap * alpha)
    return overlay


def generate_gradcam_examples(model_name, model, test_files_labels, y_true, y_pred, class_names):
    """Saves Grad-CAM overlays for a handful of correctly and incorrectly
    classified test images into outputs/plots/gradcam/{correct,incorrect}/.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    try:
        last_conv_layer_name = find_last_conv_layer(model)
    except ValueError as e:
        logger.warning(f"[{model_name}] Grad-CAM skipped: {e}")
        return

    correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    rng = np.random.default_rng(config.SEED)
    rng.shuffle(correct_idx)
    rng.shuffle(incorrect_idx)

    correct_idx = correct_idx[: config.GRADCAM_NUM_CORRECT]
    incorrect_idx = incorrect_idx[: config.GRADCAM_NUM_INCORRECT]

    if not incorrect_idx:
        logger.info(f"[{model_name}] no misclassified test samples found - skipping incorrect Grad-CAM set.")

    for label, idx_list, out_dir in [
        ("correct", correct_idx, config.DIRS["gradcam_correct"]),
        ("incorrect", incorrect_idx, config.DIRS["gradcam_incorrect"]),
    ]:
        if not idx_list:
            continue
        for rank, i in enumerate(idx_list):
            path, _ = test_files_labels[i]
            try:
                pil_img = Image.open(path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
            except Exception as e:
                logger.warning(f"[{model_name}] could not open {path}: {e}")
                continue
            img_arr = np.array(pil_img).astype("float32")
            batch = np.expand_dims(img_arr, axis=0)

            heatmap, pred_idx = make_gradcam_heatmap(
                batch, model, last_conv_layer_name, pred_index=int(y_pred[i])
            )
            overlay = _overlay_heatmap(np.array(pil_img).astype("uint8"), heatmap)

            true_name = class_names[y_true[i]]
            pred_name = class_names[y_pred[i]]

            fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
            axes[0].imshow(pil_img)
            axes[0].set_title("original", fontsize=9)
            axes[0].axis("off")
            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM", fontsize=9)
            axes[1].axis("off")
            fig.suptitle(f"true={true_name} | pred={pred_name}", fontsize=10,
                         color=("green" if label == "correct" else "red"))
            fig.tight_layout()

            out_path = os.path.join(out_dir, f"{model_name}_{label}_{rank:02d}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    logger.info(f"[{model_name}] Grad-CAM examples saved -> {config.DIRS['gradcam_correct']} / gradcam/incorrect")
