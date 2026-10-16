"""Shared evaluation helpers so every member reports results in exactly the same way.

Main metric: macro F1 (every species counts equally, which matters because AQUA20 is heavily
imbalanced: 2,200 'fish' training images vs 20 'octopus'). Accuracy and weighted F1 are also reported.
"""
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import PROJECT_DIR, RANDOM_STATE

# One results file per member (results/outputs/model_results/<member>.csv), so members can run their
# notebooks on separate PCs and the files can simply be collected into one folder - no merging needed.
RESULTS_DIR = PROJECT_DIR / "results" / "outputs" / "model_results"
PREDICTIONS_DIR = PROJECT_DIR / "results" / "outputs" / "predictions"
FIGURE_DIR = PROJECT_DIR / "results" / "model_figures"

RESULT_COLUMNS = [
    "member", "model", "variety", "preprocessing", "hyperparameters", "validation",
    "val_macro_f1", "val_macro_f1_std", "cv_macro_f1", "cv_macro_f1_std",
    "test_accuracy", "test_macro_f1", "test_weighted_f1", "train_time_s", "is_best",
]


def set_seeds(seed=RANDOM_STATE):
    """Fix Python, NumPy and (if installed) TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def stratified_kfold(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def best_cv_score(search):
    """Mean and std of the macro-F1 CV score for the best setting of a fitted *SearchCV."""
    i = search.best_index_
    return search.cv_results_["mean_test_score"][i], search.cv_results_["std_test_score"][i]


def class_weights(y):
    """'balanced' class weights as a dict, for Keras model.fit(class_weight=...)."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


def cross_validate_keras(make_model, X, y, n_splits, epochs, batch_size=64, use_class_weight=False,
                         prepare=None, make_callbacks=None, verbose=True):
    """Stratified k-fold CV for a Keras model. Returns (mean macro F1, std, per-fold scores).

    make_model() must return a freshly compiled model. Each fold trains for a fixed number of
    epochs (the best epoch found with early stopping on the hold-out validation set), so no fold
    ever sees its own test fold during training. prepare(X_tr, X_te) -> (X_tr, X_te) lets the
    caller fit preprocessing such as StandardScaler on the training fold only. make_callbacks() returns
    fresh Keras callbacks (e.g. a learning-rate schedule) for each fold.
    """
    scores = []
    for fold, (tr, te) in enumerate(stratified_kfold(n_splits).split(X, y), start=1):
        X_tr, X_te = X[tr], X[te]
        if prepare is not None:
            X_tr, X_te = prepare(X_tr, X_te)
        model = make_model()
        model.fit(X_tr, y[tr], epochs=epochs, batch_size=batch_size, verbose=0,
                  callbacks=make_callbacks() if make_callbacks is not None else None,
                  class_weight=class_weights(y[tr]) if use_class_weight else None)
        y_pred = model.predict(X_te, verbose=0).argmax(axis=1)
        scores.append(f1_score(y[te], y_pred, average="macro"))
        if verbose:
            print(f"  fold {fold}/{n_splits}: macro F1 = {scores[-1]:.4f}")
    return float(np.mean(scores)), float(np.std(scores)), scores


def make_row(variety, preprocessing, hyperparameters, validation, val_mean, val_std, test_metrics,
             seconds, cv_mean=np.nan, cv_std=np.nan):
    """One results-table row. For sklearn models the validation score IS the 5-fold CV score."""
    print(f"{variety}: validation macro F1 = {val_mean:.3f}"
          + (f" +/- {val_std:.3f}" if not np.isnan(val_std) else "")
          + f" | test macro F1 = {test_metrics['macro_f1']:.3f}"
          + f" | test accuracy = {test_metrics['accuracy']:.3f} | {seconds:.0f}s")
    print(f"    best hyperparameters: {hyperparameters}")
    return {
        "variety": variety, "preprocessing": preprocessing, "hyperparameters": str(hyperparameters),
        "validation": validation, "val_macro_f1": val_mean, "val_macro_f1_std": val_std,
        "cv_macro_f1": cv_mean, "cv_macro_f1_std": cv_std,
        "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"], "train_time_s": seconds,
    }


def fit_search(search, variety, preprocessing, X_dev, y_dev, X_test, y_test):
    """Fit a GridSearchCV / RandomizedSearchCV on the development set and score it on the test set.

    The search picks hyperparameters by k-fold CV macro F1, then refits the best setting on the
    whole development set; only that refitted model ever sees the test set.
    """
    with Timer() as t:
        search.fit(X_dev, y_dev)
    cv_mean, cv_std = best_cv_score(search)
    y_pred = search.predict(X_test)
    params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    n_folds = search.n_splits_
    row = make_row(variety, preprocessing, params, f"{n_folds}-fold stratified CV", cv_mean, cv_std,
                   compute_metrics(y_test, y_pred), t.seconds, cv_mean, cv_std)
    return row, y_pred


def search_table(search, top=10):
    """The best `top` hyperparameter settings tried by a fitted search, as a tidy DataFrame."""
    cv = pd.DataFrame(search.cv_results_)
    params = cv["params"].apply(lambda p: {k.split("__")[-1]: v for k, v in p.items()})
    table = pd.DataFrame({"hyperparameters": params.astype(str), "cv_macro_f1": cv["mean_test_score"],
                          "cv_std": cv["std_test_score"], "fit_time_s": cv["mean_fit_time"]})
    return table.sort_values("cv_macro_f1", ascending=False).head(top).reset_index(drop=True)


class Timer:
    """with Timer() as t: ...  then t.seconds"""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def print_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, digits=3, zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Row-normalized confusion matrix (each row = share of that true species), saved to results/."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, annot_kws={"size": 7},
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar_kws={"label": "Recall"})
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title(title)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=120, bbox_inches="tight")
    plt.show()


def plot_variety_comparison(results, title, filename):
    """Bar chart: validation macro F1 (with std error bars) vs test macro F1, one pair per variety."""
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(results)), 4.5))
    ax.bar(x - 0.2, results["val_macro_f1"], 0.4, yerr=results["val_macro_f1_std"].fillna(0),
           capsize=3, label="Validation macro F1", color="#4C72B0")
    ax.bar(x + 0.2, results["test_macro_f1"], 0.4, label="Test macro F1", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(results["variety"], rotation=20, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=120, bbox_inches="tight")
    plt.show()


def save_results(member, model_name, results, best_variety=None):
    """Write this member's results to results/outputs/model_results/<member>.csv (re-runs overwrite it).

    The best variety is the one with the highest validation macro F1 (never the test score),
    unless best_variety names it explicitly.
    """
    results = results.reset_index(drop=True).copy()
    results.insert(0, "model", model_name)
    results.insert(0, "member", member)
    if best_variety is None:
        best_variety = results.loc[results["val_macro_f1"].idxmax(), "variety"]
    results["is_best"] = results["variety"] == best_variety
    results = results[RESULT_COLUMNS]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{member}.csv"
    results.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved {len(results)} rows for {member} to {path}")


def load_all_results():
    """Combine every member's results file into one table (used by the group comparison notebook)."""
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No results found in {RESULTS_DIR} - run the member notebooks first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def save_predictions(member, model_name, variety, y_true, y_pred):
    """Test-set predictions of the best variety, used by the group comparison notebook."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "model": model_name, "variety": variety}) \
        .to_csv(PREDICTIONS_DIR / f"{member}.csv", index=False)
