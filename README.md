# Rice Leaf Disease Classification — 5-Model Comparative Study

Automated, reproducible training pipeline that trains and evaluates **5 CNN
architectures** (EfficientNetB0, ResNet50, DenseNet121, MobileNetV3-Large,
ConvNeXt-Tiny) on the [Rice Disease Dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)
(Bacterial Leaf Blight, Brown Spot, Leaf Blast, Neck Blast, Leaf Scald, Rice
Hispa, Narrow Brown Leaf Spot) and produces a full comparative report for
each model: accuracy, precision, recall, F1, ROC curves, confusion matrix,
training curves, prediction examples, and Grad-CAM (correct + incorrect).

Designed to run **start-to-finish on Kaggle with zero code edits** after you
clone the repo.

---

## 1. What's in this ZIP

```
rice-disease-classification/
├── README.md                      <- this file
├── requirements.txt                <- dependencies (already on Kaggle images)
├── .gitignore
├── config.py                       <- ALL settings live here (paths auto-detected)
├── run_all.py                      <- SINGLE ENTRY POINT — run this
├── kaggle_notebook.ipynb           <- ready-made Kaggle notebook (3 cells)
├── src/
│   ├── utils.py                    <- logging, seeding, dirs, json helpers
│   ├── data_utils.py                <- dataset auto-discovery + stratified split + tf.data pipelines
│   ├── models.py                    <- builders for the 5 architectures + fine-tune unfreeze logic
│   ├── train.py                     <- 2-phase transfer-learning training loop
│   ├── evaluate.py                  <- metrics, confusion matrix, ROC curves, prediction examples
│   ├── gradcam.py                   <- architecture-agnostic Grad-CAM
│   ├── pipeline.py                  <- orchestrates one model end-to-end, and all 5 in sequence
│   └── compare.py                   <- builds the final cross-model comparison table/chart
└── scripts/                        <- optional: run stages individually
    ├── 00_check_environment.py
    ├── 01_prepare_data.py
    ├── 02_train_efficientnetb0.py
    ├── 03_train_resnet50.py
    ├── 04_train_densenet121.py
    ├── 05_train_mobilenetv3large.py
    ├── 06_train_convnexttiny.py
    └── 07_compare_models.py
```

### What each piece does

| File | Purpose |
|---|---|
| `config.py` | Every path and hyperparameter (image size, batch size, epochs, learning rates, split ratios). Auto-detects whether it's running on Kaggle and where to read/write. **This is the only file you'd ever want to hand-edit**, and you don't need to for it to work. |
| `run_all.py` | The single command that runs the entire pipeline: environment check → data split → train+evaluate+Grad-CAM for all 5 models (in order, each isolated in a try/except) → final comparison. |
| `src/data_utils.py` | Recursively searches `/kaggle/input` for image folders, groups images by class regardless of whether the dataset ships as flat `ClassName/` folders or already-split `train/val/test` folders, builds **one** stratified 70/15/15 split (saved as `outputs/data_split/manifest.csv`) so every model is trained/evaluated on identical data, and turns that into `tf.data` pipelines with on-the-fly augmentation. |
| `src/models.py` | Builds each of the 5 backbones from `tf.keras.applications` (ImageNet weights, no manual downloads needed) with a shared classification head, plus the logic to unfreeze the top layers for fine-tuning. |
| `src/train.py` | Two-phase training: (1) frozen backbone, train the new head; (2) unfreeze the top ~30% of the backbone and fine-tune at a low learning rate. Uses EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint. Saves history CSV/JSON and a training-curves PNG. |
| `src/evaluate.py` | Runs the trained model on the held-out test set and saves: accuracy/precision/recall/F1 (macro + weighted), a full `classification_report`, confusion matrix heatmap, one-vs-rest ROC curves (per class + micro/macro average), and a grid of sample predictions. |
| `src/gradcam.py` | Generates Grad-CAM heatmap overlays for correctly and incorrectly classified test images. Automatically finds the right convolutional layer for **any** of the 5 architectures (no hard-coded layer names, so it won't break across TF/Keras versions). |
| `src/pipeline.py` | Wires everything together, skips models that are already done (so re-running is safe/resumable), and logs a final summary. |
| `src/compare.py` | Reads every model's saved metrics and produces `outputs/comparison/model_comparison.csv` and `.png` ranking all 5 models. |
| `scripts/0X_*.py` | Thin wrappers so you can run a single stage (e.g. just retrain ConvNeXt-Tiny) without running the whole pipeline again. |
| `kaggle_notebook.ipynb` | A ready-made 2-cell Kaggle notebook: clone repo → `python run_all.py`. |

---

## 2. Upload the project to GitHub

1. Unzip the file locally.
2. Create a **new GitHub repository** (e.g. `rice-disease-classification`), public or private.
3. From inside the unzipped folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: 5-model rice disease classification pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
   (Or just use GitHub's "Upload files" web UI if you prefer not to use git locally.)

---

## 3. Set up the Kaggle Notebook

Create a new Kaggle Notebook, then do **two one-time UI steps** (Kaggle
requires these to be done through the interface — no code can do this for
you, so this is the only "manual" part of the whole process):

1. **Attach the dataset**: right panel → **Add Input** → search
   `anshulm257/rice-disease-dataset` → **Add**.
2. **Turn on GPU + Internet**: right panel → **Accelerator** → GPU (T4x2 or
   P100) → **Internet** → On (needed once, to download ImageNet-pretrained
   weights for the 5 backbones).

That's it — no manual path editing, no pip installs, no config changes.

### Option A — copy-paste 2 cells (simplest)

```python
# Cell 1
!rm -rf rice-disease-classification
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git rice-disease-classification
%cd rice-disease-classification
```
```python
# Cell 2
!python run_all.py
```

### Option B — import the included notebook directly

New Notebook → **File → Import Notebook → GitHub** → paste the raw URL of
`kaggle_notebook.ipynb` from your repo. Edit the repo URL in the first cell,
then **Run All**.

---

## 4. What happens when it runs

`run_all.py` executes, fully automatically and in this exact order:

1. **Environment check** — logs TensorFlow version, confirms GPU is visible,
   creates all output folders.
2. **Data preparation** — discovers every class folder under
   `/kaggle/input`, merges duplicates across any train/val/test folders the
   dataset ships with, and builds one stratified 70% / 15% / 15% split
   (fixed seed = reproducible). Written once to
   `outputs/data_split/manifest.csv` and reused on every future run.
3. **Model 1/5 — EfficientNetB0**: build → train (head, then fine-tune) →
   evaluate → confusion matrix → ROC curves → prediction examples →
   Grad-CAM (correct + incorrect) → save model.
4. **Model 2/5 — ResNet50** — same steps.
5. **Model 3/5 — DenseNet121** — same steps.
6. **Model 4/5 — MobileNetV3-Large** — same steps.
7. **Model 5/5 — ConvNeXt-Tiny** — same steps.
8. **Final comparison** — aggregates all 5 models' metrics into one ranked
   table and bar chart.

If any single model errors out (e.g. an OOM on a very large backbone), the
pipeline logs the failure, saves an `*_FAILED.json` marker, and **keeps
going with the next model** rather than stopping the whole run. Re-running
`python run_all.py` afterwards will skip every model that already finished
and only retry the one(s) that failed (pass `--force` to retrain everything
regardless).

You can also train one model at a time:
```bash
python scripts/02_train_efficientnetb0.py
python scripts/05_train_mobilenetv3large.py
```
or restrict `run_all.py` to a subset:
```bash
python run_all.py --models EfficientNetB0 DenseNet121
```

---

## 5. Where outputs are saved

Everything is written under `outputs/` (inside the cloned repo, i.e.
`/kaggle/working/rice-disease-classification/outputs/` on Kaggle — this
becomes part of your saved notebook output when you click **Save Version**):

```
outputs/
├── logs/                       run_all.log, pipeline.log, per-module logs
├── data_split/manifest.csv     the exact file->class->split assignment used by every model
├── models/
│   ├── EfficientNetB0_best.keras     (best checkpoint, by val_accuracy)
│   ├── EfficientNetB0_final.keras    (final model after both training phases)
│   └── ... (one pair per model)
├── history/
│   ├── EfficientNetB0_history.csv    per-epoch loss/accuracy for both training phases
│   └── EfficientNetB0_history.json
├── metrics/
│   ├── EfficientNetB0_metrics.json          accuracy/precision/recall/F1 + full classification report
│   └── EfficientNetB0_classification_report.txt
├── plots/
│   ├── training_curves/EfficientNetB0_training_curves.png
│   ├── confusion_matrix/EfficientNetB0_confusion_matrix.png
│   ├── roc_curve/EfficientNetB0_roc_curve.png
│   ├── prediction_examples/EfficientNetB0_prediction_examples.png
│   └── gradcam/
│       ├── correct/EfficientNetB0_correct_00.png ...
│       └── incorrect/EfficientNetB0_incorrect_00.png ...
└── comparison/
    ├── model_comparison.csv    all 5 models ranked by macro F1
    └── model_comparison.png    bar chart comparing accuracy/precision/recall/F1 across models
```

**Trained models**: `outputs/models/*_final.keras` (load with
`tf.keras.models.load_model(path)`).

**Final "which model wins" answer**: `outputs/comparison/model_comparison.csv`
and `.png` — sorted by macro F1, best model logged at the end of the run.

---

## 6. Kaggle-specific notes

- **Dataset must be attached via the UI** — Kaggle does not allow
  downloading someone else's dataset through the API without your Kaggle
  credentials, so "Add Input" in the notebook is unavoidable and is the one
  manual step in this whole workflow.
- **Internet must be ON** — `tf.keras.applications.*` downloads ImageNet
  weights from Google's servers the first time each model is built.
- **GPU quota** — Kaggle gives ~30 GPU-hours/week. With the default settings
  (`config.py`: 8 head epochs + 12 fine-tune epochs per model, early
  stopping enabled) all 5 models typically finish well inside a single
  ~9-12 hour session on a T4/P100. If you're tight on quota, lower
  `EPOCHS_HEAD` / `EPOCHS_FINE_TUNE` in `config.py`, or run models in
  separate sessions using the `scripts/0X_train_*.py` wrappers — the
  manifest and any already-finished models are reused automatically.
- **Session interruption / resuming** — because the split manifest and every
  model's outputs are written to disk as soon as they finish, and
  `run_all.py` skips anything with existing metrics, you can safely stop a
  session and start a new one, re-clone the repo (or reuse
  `/kaggle/working` if it's the same session), and rerun `python run_all.py`
  to pick up where you left off.
- **Reproducibility** — a fixed seed (`config.SEED = 42`) controls the
  train/val/test split, shuffling, and TensorFlow's own RNG, so results are
  reproducible across runs of the same environment.

---

## 7. Tuning (optional)

Everything tunable lives in `config.py`:

- `IMG_SIZE`, `BATCH_SIZE`
- `VAL_SPLIT`, `TEST_SPLIT`
- `EPOCHS_HEAD`, `EPOCHS_FINE_TUNE`, `FINE_TUNE_UNFREEZE_FRACTION`
- `LR_HEAD`, `LR_FINE_TUNE`
- `EARLY_STOPPING_PATIENCE`, `REDUCE_LR_PATIENCE`, `REDUCE_LR_FACTOR`
- `GRADCAM_NUM_CORRECT` / `GRADCAM_NUM_INCORRECT`, `PRED_EXAMPLES_GRID`

No other file needs to change for a normal run.
