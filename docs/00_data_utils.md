# Module: `scripts/data_utils.py` — Shared Data Loader

## Role in the pipeline
This is not one of the six graded techniques — it's the shared foundation every notebook
(all six individual notebooks + `group_pipeline.ipynb`) imports from. It guarantees every
member loads and preprocesses raw images the exact same way, which is what makes the final
integration in `group_pipeline.ipynb` consistent.

## Flow
```
Hugging Face Hub ("taufiktrf/AQUA20")
        |
        v
 load_dataset(..., cache_dir="data/raw")   <-- load_aqua20()
        |
        v
 DatasetDict{ 'train': Dataset, 'test': Dataset }
        |
        v
 get_class_names()  -> ['species_0', 'species_1', ..., 'species_19']
 to_numpy(image)    -> RGB numpy array, ready for OpenCV / NumPy code
```

## Line-by-line walkthrough

```python
DATASET_NAME = "taufiktrf/AQUA20"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
```
- `DATASET_NAME` is the exact Hugging Face Hub identifier (`<owner>/<dataset>`), taken straight
  from the dataset's URL.
- `__file__` is this file's own path. `.resolve()` turns it into an absolute path (removes any
  `..` or symlink ambiguity). `.parent` steps up from `data_utils.py` to `scripts/`; `.parent`
  again steps up to `GROUP_ID/`. Appending `"data" / "raw"` always gives `GROUP_ID/data/raw`,
  **no matter which notebook or working directory calls this function**. This avoids the
  classic bug where a relative path (`"../data/raw"`) breaks depending on where the script is run
  from.

```python
def load_aqua20():
    return load_dataset(DATASET_NAME, cache_dir=str(RAW_DATA_DIR))
```
- `load_dataset` is a function from Hugging Face's `datasets` library.
- First call: it downloads the dataset's Parquet files from the Hub over HTTPS and stores them
  under `cache_dir`. Parquet is a columnar binary format — here it stores an `image` column
  (raw image bytes) and a `label` column (integers) for each of the 8,171 rows.
- Every call after the first: since the files already exist in `cache_dir`, it loads from disk
  instead of the network — fast, and works offline once cached.
- The return value is a `datasets.DatasetDict`, which behaves like a Python dict with two keys:
  `'train'` (6,559 rows) and `'test'` (1,612 rows). Under the hood each `Dataset` is backed by
  Apache Arrow, which memory-maps the file rather than loading all 8,171 images into RAM at
  once — accessing `dataset['train'][5]` only decodes that one row on demand.

```python
def get_class_names(dataset_dict):
    return dataset_dict["train"].features["label"].names
```
- `.features` is the *schema* of the dataset — column names and types, published by the dataset
  author alongside the data itself.
- `.features["label"]` is a `ClassLabel` object: Hugging Face's built-in type for categorical
  labels. Internally it stores both the ordered list of class-name strings and the integer code
  each one maps to.
- `.names` returns that ordered list, e.g. `class_names[7]` gives you the species name for label
  id `7`. This is what lets every notebook print human-readable species names instead of bare
  integers.

```python
def to_numpy(image, size=None):
    image = image.convert("RGB")
    if size is not None:
        image = image.resize(size)
    return np.array(image)
```
- `image` here is a `PIL.Image.Image` object (what `datasets` automatically decodes the raw
  bytes into when you access a row's `'image'` field).
- `.convert("RGB")` **defensively normalizes color mode**. Real-world image datasets are not
  guaranteed to be uniformly stored — some may be grayscale (`"L"` mode, 1 channel) or have a
  transparency channel (`"RGBA"`, 4 channels). Every downstream operation (CLAHE, normalization,
  augmentation, CNN input) assumes exactly 3 channels, so this line prevents shape-mismatch
  crashes later in the pipeline for any image that isn't already plain RGB.
- `.resize(size)` is optional — only applied if a `size` tuple like `(224, 224)` is passed in.
  Left as `None` by default so this same helper can be reused for both "give me the raw size"
  (image resizing / EDA) and "give me a fixed size" (everything downstream of resizing) use
  cases.
- `np.array(image)` converts the PIL object into a NumPy array of shape `(height, width, 3)`
  with `uint8` values 0–255 — the format OpenCV (`cv2`) and NumPy-based operations expect.

## Why this design (shared module vs. copy-pasting into each notebook)
- **Single source of truth.** If the dataset name, cache location, or RGB-handling logic ever
  needs to change, it's a one-line fix in `data_utils.py` instead of six separate edits.
- **Consistency requirement.** The Progress Review I group mark explicitly rewards "logical flow
  and collaboration" in the integrated pipeline — a shared loader is the most direct way to prove
  all six members' work operates on identically-prepared data.
- **Memory efficiency.** Relying on `datasets`' Arrow-backed lazy loading (rather than, say,
  manually downloading all images into a Python list) means the notebooks don't need to hold
  8,171 full-resolution images in memory at once.

## Viva prep questions
1. **Q: Why is `RAW_DATA_DIR` computed from `__file__` instead of a hardcoded string?**
   A: So the path resolves correctly regardless of which notebook imports this module or what
   the current working directory is — avoids "works on my machine" path bugs.
2. **Q: What would happen if you skipped `.convert("RGB")` on a grayscale image in this dataset?**
   A: `np.array(image)` would produce a 2D array `(height, width)` instead of 3D
   `(height, width, 3)`, which would crash any downstream code expecting 3 color channels
   (e.g. `cv2.cvtColor(..., COLOR_RGB2LAB)`).
3. **Q: Why use Hugging Face's `datasets` library instead of manually downloading and unzipping
   image files, like the sample Cats-vs-Dogs project did?**
   A: AQUA20 is published in Parquet format on the Hub specifically for use with `datasets`;
   it handles caching, lazy/memory-mapped loading, and label schema for us, whereas the sample
   project's dataset was distributed as a plain folder of JPEGs the group iterated manually.
4. **Q: What is `ClassLabel` and why does it matter here?**
   A: It's Hugging Face's schema type for categorical columns — it stores the mapping between
   integer label ids and human-readable class names, which we retrieve via `get_class_names()`
   instead of hardcoding species names ourselves.
5. **Q: Is the entire dataset loaded into RAM when `load_aqua20()` returns?**
   A: No — Arrow-backed `Dataset` objects are memory-mapped from disk; individual rows are only
   decoded into memory when indexed/accessed.
