"""Shared helpers for loading the AQUA20 dataset across all preprocessing notebooks."""
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATASET_NAME = "taufiktrf/AQUA20"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_aqua20():
    """Load the AQUA20 dataset (train + test splits) from Hugging Face.

    Returns a `datasets.DatasetDict` with 'train' and 'test' splits.
    Each item has an 'image' (PIL.Image) and 'label' (int, 0-19) field.
    """
    return load_dataset(DATASET_NAME, cache_dir=str(RAW_DATA_DIR))


def get_class_names(dataset_dict):
    """Return the list of species names indexed by label id."""
    return dataset_dict["train"].features["label"].names


def to_numpy(image, size=None):
    """Convert a PIL image (any mode/size) to an RGB numpy array, optionally resized."""
    image = image.convert("RGB")
    if size is not None:
        image = image.resize(size)
    return np.array(image)
