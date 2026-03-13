from sklearn.ensemble import RandomForestClassifier
from src.metrics_utils import compute_metrics


def run_oob_grid(X_train, y_train, random_state=42):
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
                    oob_score=True,
                    bootstrap=True,
                    random_state=random_state,
                    n_jobs=-1
                )

                clf.fit(X_train, y_train)

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
