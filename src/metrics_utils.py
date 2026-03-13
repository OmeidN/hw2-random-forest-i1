"""
Metrics computation and formatting for classification.
Ensures accuracy and other metrics are shown to 5 decimal places where required.
"""
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Default positive class for binary classification
DEFAULT_POS_LABEL = 1


def compute_metrics(y_true, y_pred, pos_label=DEFAULT_POS_LABEL):
    """
    Compute confusion matrix, accuracy, precision, recall, F1.
    Returns a dict with all values.
    """
    cm = confusion_matrix(y_true, y_pred)
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=pos_label, zero_division=0
        ),
    }


def format_metric(value, decimals=5):
    """Format a numeric metric to the given number of decimal places."""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def metrics_dict_to_frame(metrics_dict, model_name=None):
    """
    Convert a metrics dict (accuracy, precision, recall, f1) to a small DataFrame.
    Optionally include a model_name column.
    """
    row = {
        "accuracy": metrics_dict["accuracy"],
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "f1": metrics_dict["f1"],
    }
    if model_name is not None:
        row["model"] = model_name
    df = pd.DataFrame([row])
    if model_name is not None:
        cols = ["model", "accuracy", "precision", "recall", "f1"]
        df = df[cols]
    return df


def confusion_matrix_to_frame(cm, index_labels=None, columns_labels=None):
    """
    Convert a 2x2 confusion matrix to a labeled DataFrame.
    Default labels: ['0', '1'] for binary.
    """
    if index_labels is None:
        index_labels = ["0", "1"]
    if columns_labels is None:
        columns_labels = ["0", "1"]
    return pd.DataFrame(
        cm,
        index=index_labels,
        columns=columns_labels,
    )
