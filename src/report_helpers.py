"""
Helpers for saving result tables to CSV, figures, and formatting report text.
"""
import pandas as pd
from pathlib import Path

from src.config import OUTPUT_DATA_DIR, FIGURES_DIR
from src.metrics_utils import format_metric


def save_result_table_to_csv(df, filename, output_dir=None):
    """Save a DataFrame to CSV in the outputs directory."""
    out_dir = output_dir or OUTPUT_DATA_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def save_top10_feature_plot(feature_df, filepath=None):
    """
    Create a horizontal bar plot of top 10 features and save as PNG.
    feature_df should have 'feature' and 'gini_importance' columns (e.g. from get_feature_importance_table).
    """
    import matplotlib.pyplot as plt

    top10 = feature_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top10)), top10["gini_importance"].values, align="center")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["feature"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 10 features (Random Forest)")
    plt.tight_layout()

    if filepath is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filepath = FIGURES_DIR / "top10_features.png"
    else:
        filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return filepath


def software_tools_text():
    """Return a short text block listing software tools used."""
    return (
        "Software: Python 3, pandas, NumPy, scikit-learn (RandomForestClassifier, "
        "StratifiedKFold), matplotlib, seaborn. "
        "Environment: Jupyter Notebook or Python script."
    )


def format_audit_notes(audit_dict):
    """
    Format audit dict into a few lines of text for the report.
    """
    lines = [
        f"Samples: {audit_dict['n_samples']}, Features: {audit_dict['n_features']}. ",
        f"Class counts: {audit_dict['class_counts']}. ",
        f"Missing values: {audit_dict['missing_total']}. ",
        f"All features numeric: {audit_dict['all_features_numeric']}. ",
        f"Samples >= 10× features: {audit_dict['samples_at_least_10x_features']}. ",
        f"Class unbalanced (<10%): {audit_dict['class_unbalanced_under_10pct']}.",
    ]
    return "\n".join(lines)


def feature_ranking_discussion_starter(overlap_summary):
    """
    Return a short discussion starter for the feature ranking section,
    incorporating the ground-truth overlap summary.
    """
    return (
        "Feature ranking (Gini importance) from the best Random Forest model. "
        "Biology ground truth for i1 cluster involves COL5A2, NDNF, and FAT1; "
        "COL5A2 family genes (e.g. COL5A2, COL5A2.1) are considered together. "
        f"Overlap with top 10: {overlap_summary['summary']}"
    )
