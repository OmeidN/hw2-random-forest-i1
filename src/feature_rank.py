import pandas as pd


def get_feature_importance_table(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": model.feature_importances_
    }).sort_values("gini_importance", ascending=False)

    return importance_df
