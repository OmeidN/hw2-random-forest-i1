import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from src.metrics_utils import compute_metrics


def run_manual_3fold_cv(X, y, n_estimators, max_features, min_samples_leaf, random_state=42):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_test_fold)

        metrics = compute_metrics(y_test_fold, y_pred)

        fold_results.append({
            "fold": fold_num,
            "train_class_0": int((y_train_fold == 0).sum()),
            "train_class_1": int((y_train_fold == 1).sum()),
            "test_class_0": int((y_test_fold == 0).sum()),
            "test_class_1": int((y_test_fold == 1).sum()),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    return fold_results
