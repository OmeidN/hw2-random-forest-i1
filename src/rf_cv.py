"""
Manual 3-fold cross-validation for Random Forest.
Uses StratifiedKFold only for index generation; training/prediction/metrics are manual.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.metrics_utils import compute_metrics


def run_manual_3fold_cv(
    X, y, n_estimators, max_features, min_samples_leaf, random_state=42
):
    """
    Manual 3-fold CV: StratifiedKFold for indices only.
    For each fold: train RF on train indices, predict on test indices,
    compute confusion matrix, accuracy, precision, recall, F1.
    Also store train/test class counts per fold.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        # Use .iloc for DataFrame indexing
        if hasattr(X, "iloc"):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
        else:
            X_train_fold = X[train_idx]
            X_test_fold = X[test_idx]
        if hasattr(y, "iloc"):
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
        else:
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_test_fold)

        metrics = compute_metrics(y_test_fold, y_pred)

        train_class_0 = int((y_train_fold == 0).sum())
        train_class_1 = int((y_train_fold == 1).sum())
        test_class_0 = int((y_test_fold == 0).sum())
        test_class_1 = int((y_test_fold == 1).sum())

        fold_results.append({
            "fold": fold_num,
            "train_class_0": train_class_0,
            "train_class_1": train_class_1,
            "test_class_0": test_class_0,
            "test_class_1": test_class_1,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    return fold_results


def cv_results_to_dataframe(results):
    """Convert list of fold result dicts to a DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "fold": r["fold"],
            "train_class_0": r["train_class_0"],
            "train_class_1": r["train_class_1"],
            "test_class_0": r["test_class_0"],
            "test_class_1": r["test_class_1"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        })
    return pd.DataFrame(rows)


def summarize_cv_results(results):
    """
    Compute mean (and optionally std) of accuracy, precision, recall, f1 across folds.
    Returns a dict with mean_accuracy, mean_precision, mean_recall, mean_f1.
    """
    df = cv_results_to_dataframe(results)
    return {
        "mean_accuracy": df["accuracy"].mean(),
        "mean_precision": df["precision"].mean(),
        "mean_recall": df["recall"].mean(),
        "mean_f1": df["f1"].mean(),
        "std_accuracy": df["accuracy"].std(),
        "std_precision": df["precision"].std(),
        "std_recall": df["recall"].std(),
        "std_f1": df["f1"].std(),
    }
