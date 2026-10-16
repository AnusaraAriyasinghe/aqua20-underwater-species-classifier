# Shared model code - `scripts/preprocessing.py`, `scripts/features.py`, `scripts/eval_utils.py`

Every member's model notebook imports these three files, so **everyone should be able to explain them**.
They are the bridge between the Progress Review I preprocessing and the Final Evaluation models.

## Role in the pipeline

```
load_aqua20()                                   scripts/data_utils.py
  -> Label encoding (0-19)                      HF ClassLabel
  -> Fixed stratified train / val split         preprocessing.get_split()  -> data/splits/split_indices.csv
  -> Resize -> CLAHE [on/off] (uint8)           preprocessing.load_images() -> data/cache/*.npz
       |
       |-- Branch A: images in (M3 CNN)
       |     -> Rescaling(1/255) + augmentation layers inside the Keras model (training only)
       |
       '-- Branch B: feature vectors in (M1 LR, M2 SVM, M4 PCA+KNN, M5 MLP, M6 RF)
             -> features.load_features("hog_color" | "pixels32", clahe)  -> data/features/*.npz
             -> features.dev_set(): train + val combined for k-fold CV
  -> eval_utils: same metrics, plots and results file for everyone
       -> results/outputs/model_results/<member>.csv, results/outputs/predictions/<member>.csv,
          results/model_figures/<member>_*.png
```

## `scripts/preprocessing.py`

| Function | What it does | Why |
|---|---|---|
| `apply_clahe(image)` | CLAHE on the L channel of LAB colour space (clip 2.0, 8x8 tiles) | Identical to the group pipeline; corrects underwater colour cast and low contrast without distorting colours |
| `get_split(dataset)` | Loads `data/splits/split_indices.csv` (or recreates it with `random_state=42`, stratified 85/15) | Every member uses **exactly the same images** for train / val / test, so the group comparison is fair |
| `load_images(size, clahe)` | RGB -> resize to `size x size` -> optional CLAHE, returned as uint8 arrays per split; cached in `data/cache/` | Decoding 8,171 JPEGs takes ~30 s; the cache makes later runs instant. uint8 keeps memory 4x lower than float32 |
| `normalize(images)` | uint8 0-255 -> float32 0-1 | The group's normalization step |
| `make_augmentation()` | Keras `RandomFlip`, `RandomRotation(0.08)`, `RandomZoom(0.10)`, `RandomBrightness(0.15)` | Identical to the group pipeline; used as layers so they only run during `fit()` |

**Split safety check:** the split file stores each image's label. `get_split()` compares those labels with
the downloaded dataset and stops with an error if they differ, so a changed dataset can never silently give a
different split. It also checks the sizes: 5,575 train / 984 validation / 1,612 test.

## `scripts/features.py`

| Feature type | How it is computed | Size |
|---|---|---|
| `hog_color` | 128x128 image -> grayscale -> **HOG** (9 orientations, 16x16 cells, 2x2 blocks, L2-Hys) = 1,764 values, plus **HSV colour histograms** (32 bins each for hue, saturation, value, normalized) = 96 values | 1,860 |
| `pixels32` | 32x32 RGB image -> normalized 0-1 -> flattened | 3,072 |

- **HOG** (Histogram of Oriented Gradients) describes **shape and edges**: in each 16x16 cell it counts how
  strongly edges point in each of 9 directions. Fins, tentacles and body outlines become numbers.
- **Colour histograms** describe **colour distribution** (e.g. orange clownfish vs grey shark), which HOG
  ignores because it works on grayscale.
- **Raw pixels** are the simplest possible features - a baseline that shows how much HOG helps.
- `flipped=True` builds features of horizontally flipped images, used only in M6's augmentation variety.
- `dev_set(data)` joins train + validation (6,559 images) for k-fold cross-validation. The test set
  (1,612 images) is never part of any tuning.

Run `python scripts/features.py` once to build every feature file (~2 minutes), or let the notebooks build
them on first use.

## `scripts/eval_utils.py`

| Function | Purpose |
|---|---|
| `set_seeds()` | Fixes Python, NumPy and TensorFlow random seeds (42) |
| `stratified_kfold(n)` | `StratifiedKFold(n, shuffle=True, random_state=42)` - every member gets the **same folds** |
| `compute_metrics()` | Accuracy, macro F1, weighted F1 |
| `fit_search()` | Fits a `GridSearchCV` / `RandomizedSearchCV`, takes the best CV macro F1, scores the refitted best model on the test set, returns one results row |
| `cross_validate_keras()` | Stratified k-fold CV for Keras models (fixed epochs, optional per-fold scaling) |
| `class_weights()` | "balanced" class weights for Keras |
| `plot_confusion_matrix()` | Row-normalized confusion matrix (diagonal = recall per species) |
| `plot_variety_comparison()` | Validation vs test macro F1 for every variety |
| `save_results()` | Writes this member's rows to `results/outputs/model_results/<member>.csv` (re-runs overwrite it) and marks the best variety. One file per member, so results from different PCs can simply be collected into one folder |
| `load_all_results()` | Combines every member's results file into one table for the group comparison |
| `save_predictions()` | Saves the best model's test predictions for the group per-species comparison |

## Evaluation protocol (the same for all six members)

- **Main metric - macro F1:** the F1 score of each of the 20 species, averaged with equal weight. AQUA20 is
  heavily imbalanced (2,200 fish vs 20 octopus training images), so a model that always predicts "fish"
  would get ~33% accuracy but a macro F1 near 0.03. Macro F1 rewards getting rare species right.
- **Also reported:** accuracy (easy to explain) and weighted F1 (F1 weighted by class size).
- **Model selection never uses the test set.** scikit-learn models: stratified 5-fold CV on train + val.
  Keras models: hold-out validation set for comparing varieties, then stratified k-fold CV of the winner.
- **Test set used once per variety** for the final reported score.

## Why this design

- **Q: Why not just call `train_test_split` in every notebook?**
  A: It would give the same split only if every laptop had the same library versions and dataset order. The
  committed split file removes that risk and the label check catches dataset changes.
- **Q: Why are `data/cache/` and `data/features/` not in Git?**
  A: They are large (up to 400 MB) and can be rebuilt from the dataset in about 2 minutes. Only the tiny split
  file (90 KB) is committed.
- **Q: Why is StandardScaler inside the sklearn Pipeline instead of applied once?**
  A: Inside a Pipeline it is refitted on the training folds only in every CV split, so the validation fold
  never influences the scaling (no data leakage).
- **Q: Why combine train and validation for cross-validation?**
  A: More data per fold gives more reliable scores, and k-fold CV already provides the validation role. The
  test set stays separate.
