"""Feature extraction for the classical models (Branch B: M1 LR, M2 SVM, M4 PCA+KNN, M5 MLP, M6 RF).

Two feature types, each with and without CLAHE:
- "hog_color": HOG shape/texture descriptor (128x128 grayscale) + HSV colour histograms -> 1,860 values
- "pixels32":  the image itself, resized to 32x32 RGB, normalized to 0-1 and flattened -> 3,072 values

Features are cached in data/features/ (not committed). Run `python scripts/features.py` once to
build everything up front, or let the notebooks build what they need on first use.
"""
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

from preprocessing import PROJECT_DIR, load_images, normalize

FEATURE_DIR = PROJECT_DIR / "data" / "features"
FEATURE_TYPES = ("hog_color", "pixels32")
HOG_IMAGE_SIZE = 128
PIXEL_IMAGE_SIZE = 32
HIST_BINS = 32


def hog_color_vector(image):
    """HOG of the grayscale image + per-channel HSV histograms, for one uint8 RGB image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_vector = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                     block_norm="L2-Hys")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ranges = [180, 256, 256]  # OpenCV hue is 0-179
    hists = [np.histogram(hsv[..., c], bins=HIST_BINS, range=(0, ranges[c]))[0] for c in range(3)]
    color_vector = np.concatenate(hists).astype("float32") / (image.shape[0] * image.shape[1])
    return np.concatenate([hog_vector.astype("float32"), color_vector])


def hog_color_features(images):
    return np.stack([hog_color_vector(image) for image in images])


def pixel_features(images):
    return normalize(images).reshape(len(images), -1)


def load_features(kind, clahe, flipped=False, verbose=True):
    """Return a dict with X_train, y_train, X_val, y_val, X_test, y_test, class_names.

    flipped=True returns features of horizontally flipped images (used only by M6's
    flip-augmentation variety, and only ever for training folds).
    """
    if kind not in FEATURE_TYPES:
        raise ValueError(f"kind must be one of {FEATURE_TYPES}")
    tag = f"{kind}_{'clahe' if clahe else 'raw'}{'_flipped' if flipped else ''}"
    cache_file = FEATURE_DIR / f"{tag}.npz"
    size = HOG_IMAGE_SIZE if kind == "hog_color" else PIXEL_IMAGE_SIZE

    if cache_file.exists():
        images = None
        data = dict(np.load(cache_file))
        data["class_names"] = load_images(size, clahe, verbose=False)["class_names"]
    else:
        images = load_images(size, clahe, verbose=verbose)
        extract = hog_color_features if kind == "hog_color" else pixel_features
        if verbose:
            print(f"Extracting {tag} features (first run only)...")
        data = {}
        for split in ("train", "val", "test"):
            X = images[f"X_{split}"]
            if flipped:
                X = X[:, :, ::-1, :]  # flip left-right
            data[f"X_{split}"] = extract(X)
            data[f"y_{split}"] = images[f"y_{split}"]
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **data)
        data["class_names"] = images["class_names"]

    if verbose:
        print(f"Features {tag}: train {data['X_train'].shape}, val {data['X_val'].shape}, "
              f"test {data['X_test'].shape}")
    return data


def dev_set(data):
    """Combine train + val into one development set for k-fold cross-validation.

    The test split is kept completely separate and is only used for the final scores.
    """
    X = np.concatenate([data["X_train"], data["X_val"]])
    y = np.concatenate([data["y_train"], data["y_val"]])
    return X, y


if __name__ == "__main__":
    for kind in FEATURE_TYPES:
        for clahe in (False, True):
            load_features(kind, clahe)
    load_features("hog_color", clahe=True, flipped=True)
    load_features("hog_color", clahe=False, flipped=True)
    print("All features cached in", FEATURE_DIR)
