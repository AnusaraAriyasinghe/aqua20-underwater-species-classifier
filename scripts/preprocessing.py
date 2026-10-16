"""Shared preprocessing for every model notebook.

Reproduces the group pipeline (notebooks/group_pipeline.ipynb) as importable functions:
label encoding -> fixed stratified train/val split -> resize -> CLAHE (optional) -> normalize
-> training-only augmentation (Keras models only).

The train/validation split is stored in data/splits/split_indices.csv and committed to Git, so
every member trains and evaluates on exactly the same images. The dataset itself is never
committed; load_aqua20() downloads it into data/raw/ on the first run.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import load_aqua20, get_class_names, to_numpy

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_FILE = PROJECT_DIR / "data" / "splits" / "split_indices.csv"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

RANDOM_STATE = 42
VAL_SIZE = 0.15
NUM_CLASSES = 20
EXPECTED_SIZES = {"train": 5575, "val": 984, "test": 1612}


def apply_clahe(image_array):
    """CLAHE on the L channel of LAB colour space (same settings as the group pipeline)."""
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def _create_split(dataset):
    labels = np.array(dataset["train"]["label"])
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    split = pd.DataFrame({"index": indices, "label": labels, "split": "train"})
    split.loc[val_idx, "split"] = "val"
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_FILE, index=False)
    return split


def get_split(dataset):
    """Return (train_idx, val_idx) into dataset['train'], loading the committed split file.

    If the file is missing it is recreated with the same random_state. The stored labels are
    compared against the loaded dataset, so a changed dataset fails loudly instead of silently
    producing a different split.
    """
    split = pd.read_csv(SPLIT_FILE) if SPLIT_FILE.exists() else _create_split(dataset)
    labels = np.array(dataset["train"]["label"])
    if len(split) != len(labels) or not np.array_equal(split["label"].to_numpy(), labels):
        raise ValueError(
            "The downloaded AQUA20 dataset does not match data/splits/split_indices.csv. "
            "Delete data/raw/ and re-download, or check the dataset version."
        )
    train_idx = split.loc[split["split"] == "train", "index"].to_numpy()
    val_idx = split.loc[split["split"] == "val", "index"].to_numpy()
    sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(dataset["test"])}
    if sizes != EXPECTED_SIZES:
        raise ValueError(f"Unexpected split sizes {sizes}, expected {EXPECTED_SIZES}")
    return train_idx, val_idx


def preprocess_image(pil_image, size, clahe):
    """RGB conversion -> resize -> optional CLAHE. Returns uint8 (normalization happens later)."""
    image = to_numpy(pil_image, size=(size, size))
    if clahe:
        image = apply_clahe(image)
    return image


def load_images(size, clahe, verbose=True):
    """Return a dict of uint8 image arrays and labels for the train / val / test splits.

    Keys: X_train, y_train, X_val, y_val, X_test, y_test, class_names.
    Results are cached in data/cache/ (not committed) so later runs load in seconds.
    """
    cache_file = CACHE_DIR / f"images_{size}px_{'clahe' if clahe else 'raw'}.npz"
    dataset = load_aqua20()
    class_names = get_class_names(dataset)
    train_idx, val_idx = get_split(dataset)

    if cache_file.exists():
        data = dict(np.load(cache_file))
    else:
        if verbose:
            print(f"Preprocessing images at {size}x{size}, CLAHE={clahe} (first run only)...")
        train_split, test_split = dataset["train"], dataset["test"]
        # Iterate example by example so only one full-size image is decoded in memory at a time.
        train_images = np.stack([preprocess_image(ex["image"], size, clahe) for ex in train_split])
        test_images = np.stack([preprocess_image(ex["image"], size, clahe) for ex in test_split])
        train_labels = np.array(train_split["label"])
        data = {
            "X_train": train_images[train_idx], "y_train": train_labels[train_idx],
            "X_val": train_images[val_idx], "y_val": train_labels[val_idx],
            "X_test": test_images, "y_test": np.array(test_split["label"]),
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **data)

    data["class_names"] = class_names
    if verbose:
        print(f"{size}x{size} CLAHE={clahe}: train {data['X_train'].shape}, "
              f"val {data['X_val'].shape}, test {data['X_test'].shape}")
    return data


def normalize(images):
    """Scale uint8 pixels from 0-255 to 0-1 (float32)."""
    return images.astype("float32") / 255.0


def make_augmentation():
    """Training-only augmentation layers, identical to the group pipeline.

    Used as the first layers of a Keras model: they are active during fit() and switched off
    automatically during evaluate() / predict(), so validation and test images stay unchanged.
    Expects inputs already normalized to [0, 1].
    """
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
    ], name="data_augmentation")
