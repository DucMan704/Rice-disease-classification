"""
src/pipeline.py
=================
Top-level orchestration used by both run_all.py and the individual
scripts/0X_train_<model>.py wrappers.
"""

import os
import gc
import traceback

import config
from src.utils import get_logger, ensure_all_dirs, set_global_seed, Timer, save_json
from src.data_utils import build_or_load_manifest, build_tf_datasets
from src.models import MODEL_BUILDERS
from src.train import train_model
from src.evaluate import evaluate_model, plot_prediction_examples
from src.gradcam import generate_gradcam_examples

logger = get_logger("pipeline")


def check_environment():
    ensure_all_dirs()
    set_global_seed()

    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            logger.info(f"GPU available: {g}")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    else:
        logger.warning(
            "No GPU detected - training will be slow. On Kaggle: "
            "Settings (right panel) -> Accelerator -> GPU."
        )
    return len(gpus) > 0


def prepare_data():
    with Timer(logger, "data discovery + split"):
        split = build_or_load_manifest()
    with Timer(logger, "building tf.data pipelines"):
        train_ds, val_ds, test_ds, class_names, test_files_labels = build_tf_datasets(split)
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    return train_ds, val_ds, test_ds, class_names, test_files_labels


def _metrics_exist(model_name):
    return os.path.exists(os.path.join(config.DIRS["metrics"], f"{model_name}_metrics.json"))


def train_single_model(model_name, train_ds, val_ds, test_ds, class_names, test_files_labels, force=False):
    """Builds, trains, evaluates and Grad-CAMs a single model.
    Idempotent: if metrics already exist for this model, it is skipped
    unless force=True (so re-running run_all.py safely resumes)."""

    if not force and _metrics_exist(model_name):
        logger.info(f"[{model_name}] metrics already exist - skipping (use force=True to redo).")
        return True

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_BUILDERS.keys())}")

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()

        with Timer(logger, f"[{model_name}] build model"):
            model, base_model = MODEL_BUILDERS[model_name](num_classes=len(class_names))
            logger.info(f"[{model_name}] total params: {model.count_params():,}")

        model, history_rows = train_model(model_name, model, base_model, train_ds, val_ds)

        with Timer(logger, f"[{model_name}] evaluation on test set"):
            metrics, y_true, y_pred, y_prob = evaluate_model(model_name, model, test_ds, class_names)

        plot_prediction_examples(model_name, test_files_labels, y_true, y_pred, class_names)

        with Timer(logger, f"[{model_name}] Grad-CAM generation"):
            generate_gradcam_examples(model_name, model, test_files_labels, y_true, y_pred, class_names)

        del model, base_model
        gc.collect()
        tf.keras.backend.clear_session()

        logger.info(f"[{model_name}] === COMPLETE ===")
        return True

    except Exception as e:
        logger.error(f"[{model_name}] FAILED: {e}")
        logger.error(traceback.format_exc())
        save_json(
            {"model": model_name, "status": "FAILED", "error": str(e)},
            os.path.join(config.DIRS["metrics"], f"{model_name}_FAILED.json"),
        )
        return False


def run_full_pipeline(model_names=None, force=False):
    model_names = model_names or config.MODEL_NAMES

    check_environment()
    train_ds, val_ds, test_ds, class_names, test_files_labels = prepare_data()

    results = {}
    for name in model_names:
        ok = train_single_model(name, train_ds, val_ds, test_ds, class_names, test_files_labels, force=force)
        results[name] = "OK" if ok else "FAILED"

    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    for name, status in results.items():
        logger.info(f"  {name:20s} -> {status}")
    logger.info("=" * 60)

    from src.compare import build_comparison
    build_comparison(model_names)

    return results
