"""
src/data_utils.py
==================
Everything related to finding the dataset on disk, building a stratified
train/val/test split (once, reproducibly), and turning that split into
tf.data.Dataset objects ready to feed into any of the 5 models.

Design goals:
- Zero manual path configuration: we search /kaggle/input recursively.
- Works whether the Kaggle dataset ships as flat `ClassName/*.jpg` folders,
  or as `train/ClassName/*.jpg`, `val/...`, `test/...` -- we merge everything
  found and re-split ourselves so every model sees the EXACT same split
  (required for a fair comparative analysis).
- The split is written to disk (manifest CSV) the first time and reused on
  every subsequent run, so re-running the pipeline is reproducible and the
  5 models are always compared on identical data.
"""

import os
import csv
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

import config
from src.utils import get_logger

logger = get_logger("data_utils")

# Folder names that are split-indicators, not class names. If a leaf folder
# happens to be named one of these, we fall back to its parent's name.
_SPLIT_LIKE_NAMES = {
    "train", "training", "val", "valid", "validation",
    "test", "testing", "images", "image", "data", "dataset",
}


def _candidate_base_dirs():
    """Directories to search for image class-folders."""
    bases = []
    if config.DATA_DIR_OVERRIDE:
        bases.append(config.DATA_DIR_OVERRIDE)
    if os.path.isdir(config.KAGGLE_INPUT_DIR):
        bases.append(config.KAGGLE_INPUT_DIR)
    # local fallback for testing outside Kaggle
    local_fallback = os.path.join(config.PROJECT_ROOT, "data", "raw")
    if os.path.isdir(local_fallback):
        bases.append(local_fallback)
    return bases


def discover_class_images():
    """Walk every candidate base dir and group image files by inferred
    class name (the name of the leaf folder that directly contains them).

    Returns: dict[class_name] -> list[abs file paths]
    """
    class_files = defaultdict(list)

    for base in _candidate_base_dirs():
        for dirpath, dirnames, filenames in os.walk(base):
            imgs = [f for f in filenames if f.endswith(config.IMG_EXTENSIONS)]
            if not imgs:
                continue
            # Only treat this as a "class leaf" if it has no sub-directories
            # itself (i.e. the images live directly here).
            if dirnames:
                continue

            class_name = os.path.basename(dirpath.rstrip("/"))
            if class_name.lower() in _SPLIT_LIKE_NAMES:
                parent = os.path.basename(os.path.dirname(dirpath.rstrip("/")))
                if parent:
                    class_name = parent

            class_name = class_name.strip()
            for f in imgs:
                class_files[class_name].append(os.path.join(dirpath, f))

    return class_files


def _validate_and_report(class_files):
    if not class_files:
        raise FileNotFoundError(
            "Could not find any images under /kaggle/input (or DATA_DIR).\n"
            "-> Make sure you attached the 'Rice Disease Dataset' as a Kaggle "
            "notebook Input (Add Input -> search 'anshulm257/rice-disease-dataset').\n"
            "This is the ONE manual step Kaggle requires; the code cannot do it "
            "for you because Kaggle datasets must be attached through the UI."
        )

    logger.info(f"Discovered {len(class_files)} classes:")
    total = 0
    for cls, files in sorted(class_files.items()):
        logger.info(f"  - {cls}: {len(files)} images")
        total += len(files)
    logger.info(f"Total images found: {total}")

    # Drop classes with too few images to stratify into train/val/test
    MIN_PER_CLASS = 6
    dropped = [c for c, f in class_files.items() if len(f) < MIN_PER_CLASS]
    for c in dropped:
        logger.warning(f"Dropping class '{c}' - only {len(class_files[c])} images (<{MIN_PER_CLASS})")
        del class_files[c]

    if len(class_files) < 2:
        raise RuntimeError("Fewer than 2 usable classes found - cannot train a classifier.")

    return class_files


def _manifest_path():
    return os.path.join(config.DIRS["data_split"], "manifest.csv")


def build_or_load_manifest():
    """Create the stratified train/val/test split once and persist it, or
    load the existing one if it already matches on disk. Returns a dict:
    { 'train': [(path, label), ...], 'val': [...], 'test': [...],
      'class_names': [sorted list] }
    """
    manifest_file = _manifest_path()

    if os.path.exists(manifest_file):
        logger.info(f"Found existing split manifest at {manifest_file}, reusing it.")
        rows = []
        with open(manifest_file) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        class_names = sorted(set(r["label"] for r in rows))
        split = {"train": [], "val": [], "test": []}
        missing = 0
        for r in rows:
            if not os.path.exists(r["filepath"]):
                missing += 1
                continue
            split[r["split"]].append((r["filepath"], r["label"]))
        if missing:
            logger.warning(
                f"{missing} files listed in manifest no longer exist on disk "
                f"(skipped). If this seems wrong, delete {manifest_file} and rerun."
            )
        split["class_names"] = class_names
        return split

    logger.info("No existing manifest found - discovering images and building a new split.")
    class_files = discover_class_images()
    class_files = _validate_and_report(class_files)
    class_names = sorted(class_files.keys())

    rows = []
    for cls in class_names:
        files = sorted(class_files[cls])
        labels = [cls] * len(files)

        # first carve off test set
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            files, labels, test_size=config.TEST_SPLIT, random_state=config.SEED, shuffle=True,
        )
        # then carve val out of the remainder
        val_ratio_of_remainder = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels, test_size=val_ratio_of_remainder,
            random_state=config.SEED, shuffle=True,
        )

        for fp, lb in zip(train_files, train_labels):
            rows.append({"filepath": fp, "label": lb, "split": "train"})
        for fp, lb in zip(val_files, val_labels):
            rows.append({"filepath": fp, "label": lb, "split": "val"})
        for fp, lb in zip(test_files, test_labels):
            rows.append({"filepath": fp, "label": lb, "split": "test"})

    os.makedirs(config.DIRS["data_split"], exist_ok=True)
    with open(manifest_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote split manifest ({len(rows)} images) to {manifest_file}")

    split = {"train": [], "val": [], "test": []}
    for r in rows:
        split[r["split"]].append((r["filepath"], r["label"]))
    split["class_names"] = class_names

    for s in ("train", "val", "test"):
        logger.info(f"  {s}: {len(split[s])} images")

    return split


def _make_label_lookup(class_names):
    return {name: i for i, name in enumerate(class_names)}


def _load_and_resize(path, label, img_size):
    import tensorflow as tf
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)
    return img, label


def _augment(img, label):
    import tensorflow as tf
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def build_tf_datasets(split, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE):
    """Turn the manifest split dict into tf.data.Dataset objects.

    Returns: train_ds, val_ds, test_ds, class_names, test_files_labels
    where test_files_labels is a plain python list of (filepath, label_str)
    kept around for Grad-CAM / prediction-example visualisation later.
    """
    import tensorflow as tf

    class_names = split["class_names"]
    lookup = _make_label_lookup(class_names)

    def to_arrays(pairs):
        paths = [p for p, _ in pairs]
        labels = [lookup[l] for _, l in pairs]
        return paths, labels

    train_paths, train_labels = to_arrays(split["train"])
    val_paths, val_labels = to_arrays(split["val"])
    test_paths, test_labels = to_arrays(split["test"])

    AUTOTUNE = tf.data.AUTOTUNE

    def make_ds(paths, labels, training, shuffle_buffer=1000):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(min(shuffle_buffer, len(paths)), seed=config.SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, l: _load_and_resize(p, l, img_size), num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(train_paths, train_labels, training=True)
    val_ds = make_ds(val_paths, val_labels, training=False)
    test_ds = make_ds(test_paths, test_labels, training=False)

    test_files_labels = list(zip(test_paths, [class_names[i] for i in test_labels]))

    return train_ds, val_ds, test_ds, class_names, test_files_labels
