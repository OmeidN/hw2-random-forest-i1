"""
Feature importance ranking from a trained Random Forest (Gini importance).
Ground-truth overlap: COL5A2, NDNF, FAT1 and COL5A2 family (prefix match).
"""
import pandas as pd


def get_feature_importance_table(model, feature_names):
    """
    Build a DataFrame of feature names and Gini importance, sorted descending.
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": model.feature_importances_,
    }).sort_values("gini_importance", ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    return importance_df


# Biology ground truth: genes of interest for i1 cluster
GROUND_TRUTH_GENES = ["COL5A2", "NDNF", "FAT1"]
COL5A2_PREFIX = "COL5A2"  # COL5A2 family = prefix match


def summarize_ground_truth_overlap(feature_df, top_n=10):
    """
    Check whether any of the top_n ranked features contain COL5A2, NDNF, or FAT1.
    COL5A2 family: treat by prefix (e.g. COL5A2, COL5A2.1, COL5A2.2).
    Returns a dict with lists of matching feature names and a short summary.
    """
    top = feature_df.head(top_n)
    features_list = top["feature"].astype(str).tolist()

    matches = {
        "COL5A2_family": [],
        "NDNF": [],
        "FAT1": [],
    }
    for f in features_list:
        f_upper = f.upper()
        if f_upper.startswith(COL5A2_PREFIX):
            matches["COL5A2_family"].append(f)
        if "NDNF" in f_upper:
            matches["NDNF"].append(f)
        if "FAT1" in f_upper:
            matches["FAT1"].append(f)

    summary_parts = []
    if matches["COL5A2_family"]:
        summary_parts.append(
            f"COL5A2 family (prefix): {', '.join(matches['COL5A2_family'])}"
        )
    if matches["NDNF"]:
        summary_parts.append(f"NDNF: {', '.join(matches['NDNF'])}")
    if matches["FAT1"]:
        summary_parts.append(f"FAT1: {', '.join(matches['FAT1'])}")
    if not summary_parts:
        summary_parts.append(
            "None of the biology ground-truth genes (COL5A2, NDNF, FAT1) "
            "appear in the top-ranked features."
        )

    return {
        "matches": matches,
        "summary": "; ".join(summary_parts),
        "top_features": features_list,
    }
