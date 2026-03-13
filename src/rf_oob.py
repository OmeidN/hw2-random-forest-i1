"""
Random Forest training with Out-of-Bag (OOB) estimation.
Grid search over hyperparameters; each model uses OOB for evaluation.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.metrics_utils import compute_metrics


def run_oob_grid(X_train, y_train, random_state=42):
    """
    Train Random Forest models with different hyperparameters.
    Each model uses bootstrap=True, oob_score=True.
    OOB predictions are derived from oob_decision_function_, threshold 0.5.
    Returns a list of result dicts (each with model, params, metrics).
    """
    results = []

    n_estimators_list = [200, 500, 1000]
    max_features_list = ["sqrt", "log2", 0.1, 0.2]
    min_samples_leaf_list = [1, 2, 5]

    for n_estimators in n_estimators_list:
        for max_features in max_features_list:
            for min_samples_leaf in min_samples_leaf_list:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=True,
                    oob_score=True,
                    random_state=random_state,
                    n_jobs=-1,
                )
                clf.fit(X_train, y_train)

                # OOB predictions: use probability for class 1, threshold 0.5
                oob_probs = clf.oob_decision_function_[:, 1]
                oob_pred = (oob_probs >= 0.5).astype(int)
                metrics = compute_metrics(y_train, oob_pred)

                results.append({
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                    "oob_score_builtin": clf.oob_score_,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "model": clf,
                })

    return results


def oob_results_to_dataframe(results):
    """
    Convert the list of OOB result dicts to a DataFrame (params + metrics).
    Accuracy etc. will be numeric; format when printing if you need 5 decimals.
    """
    rows = []
    for r in results:
        rows.append({
            "n_estimators": r["n_estimators"],
            "max_features": str(r["max_features"]),
            "min_samples_leaf": r["min_samples_leaf"],
            "oob_score": r["oob_score_builtin"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        })
    return pd.DataFrame(rows)


def get_best_oob_model(results, sort_by="f1"):
    """
    Return the single best result dict from OOB results, sorted by sort_by (default f1).
    Returns the full result dict so you can get result["model"] and result["accuracy"], etc.
    """
    if not results:
        return None
    valid = [s for s in sort_by.split(",") if s in results[0]]
    sort_key = valid[0] if valid else "f1"
    best = max(results, key=lambda x: x[sort_key])
    return best
