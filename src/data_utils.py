"""
Data loading, auditing, and train/verification split utilities.
Handles the original DB audit and creation of training vs verification sets.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import RANDOM_STATE, POS_LABEL, NEG_LABEL


def load_data(csv_path):
    """
    Load a CSV file into a pandas DataFrame.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def detect_label_column(df):
    """
    Detect the label column robustly: prefer 'label', accept 'Label'.
    Returns the actual column name used in the DataFrame.
    """
    cols_lower = [c for c in df.columns if str(c).strip().lower() == "label"]
    if cols_lower:
        # Return the actual column name as in the DataFrame
        for c in df.columns:
            if str(c).strip().lower() == "label":
                return c
    # Fallback: check for exact match
    if "label" in df.columns:
        return "label"
    if "Label" in df.columns:
        return "Label"
    raise ValueError(
        "No label column found. Expected a column named 'label' or 'Label'. "
        f"Columns: {list(df.columns)}"
    )


def basic_audit(df, label_col):
    """
    Perform a basic audit of the dataset.
    Returns a dict with sample count, feature count, class counts, missing values,
    dtype summary, and sanity checks (numeric features, PII heuristic, etc.).
    """
    n_samples = len(df)
    feature_cols = [c for c in df.columns if c != label_col]
    n_features = len(feature_cols)

    # Class counts and percentages
    class_counts = df[label_col].value_counts().sort_index().to_dict()
    total = n_samples
    class_pct = {
        k: (v / total * 100) if total else 0
        for k, v in class_counts.items()
    }

    # Missing values
    missing_total = int(df.isna().sum().sum())
    missing_per_col = df.isna().sum()

    # Dtype summary (excluding label)
    X = df[feature_cols]
    dtypes = X.dtypes
    all_numeric = all(
        pd.api.types.is_numeric_dtype(d) for d in dtypes
    )
    dtype_summary = dtypes.value_counts().to_dict()
    # Convert to string for serialization
    dtype_summary = {str(k): int(v) for k, v in dtype_summary.items()}

    # Simple PII heuristic: columns that might be identifiers
    pii_heuristic = []
    for c in df.columns:
        c_lower = str(c).lower()
        if any(
            x in c_lower
            for x in ["id", "name", "patient", "sample_id", "subject"]
        ):
            pii_heuristic.append(c)
    has_pii_heuristic = len(pii_heuristic) > 0

    # Samples vs features: recommend at least 10x samples than features
    samples_ok = n_samples >= 10 * n_features if n_features else True

    # Class balance: flag if either class is < 10%
    min_pct = min(class_pct.values()) if class_pct else 0
    unbalanced = min_pct < 10.0

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "class_counts": class_counts,
        "class_pct": class_pct,
        "missing_total": missing_total,
        "missing_per_col": missing_per_col,
        "dtype_summary": dtype_summary,
        "all_features_numeric": all_numeric,
        "pii_heuristic_columns": pii_heuristic,
        "has_pii_heuristic": has_pii_heuristic,
        "samples_at_least_10x_features": samples_ok,
        "class_unbalanced_under_10pct": unbalanced,
        "columns": list(df.columns),
        "feature_columns": feature_cols,
    }


def audit_summary_table(df, label_col):
    """
    Build a summary table (DataFrame) suitable for display from audit results.
    """
    audit = basic_audit(df, label_col)
    rows = [
        ("Number of samples", audit["n_samples"]),
        ("Number of features", audit["n_features"]),
        ("Class counts", str(audit["class_counts"])),
        ("Class percentages (%)", str(audit["class_pct"])),
        ("Missing values (total)", audit["missing_total"]),
        ("All features numeric", audit["all_features_numeric"]),
        ("PII heuristic columns found", audit["pii_heuristic_columns"] or "None"),
        ("Samples >= 10× features", audit["samples_at_least_10x_features"]),
        ("Class unbalanced (<10% in any class)", audit["class_unbalanced_under_10pct"]),
    ]
    return pd.DataFrame(rows, columns=["Audit item", "Value"])


def split_train_verification(df, label_col, random_state=RANDOM_STATE):
    """
    Create verification set by removing exactly:
    - 1 random sample with class 1 (i1)
    - 1 random sample with class 0 (non-i1)
    The remaining data is the training set.
    Returns (training_df, verification_df).
    """
    pos = df[df[label_col] == POS_LABEL]
    neg = df[df[label_col] == NEG_LABEL]
    if len(pos) < 1:
        raise ValueError("No positive (class 1) samples to hold out.")
    if len(neg) < 1:
        raise ValueError("No negative (class 0) samples to hold out.")

    pos_sample = pos.sample(n=1, random_state=random_state)
    neg_sample = neg.sample(n=1, random_state=random_state)
    verification_df = pd.concat([pos_sample, neg_sample]).copy()
    training_df = df.drop(index=verification_df.index).copy()

    return training_df, verification_df


def get_xy(df, label_col):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()
    return X, y


def save_dataframe(df, path):
    """Save a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
