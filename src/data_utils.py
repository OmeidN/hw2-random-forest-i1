import pandas as pd
from src.config import RANDOM_STATE, LABEL_COL, POS_LABEL, NEG_LABEL


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def basic_audit(df, label_col=LABEL_COL):
    n_samples = len(df)
    n_features = df.shape[1] - 1
    class_counts = df[label_col].value_counts().to_dict()
    missing_total = int(df.isna().sum().sum())

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "class_counts": class_counts,
        "missing_total": missing_total,
        "columns": list(df.columns),
    }


def split_train_verification(df, label_col=LABEL_COL):
    pos_sample = df[df[label_col] == POS_LABEL].sample(n=1, random_state=RANDOM_STATE)
    neg_sample = df[df[label_col] == NEG_LABEL].sample(n=1, random_state=RANDOM_STATE)

    verification_df = pd.concat([pos_sample, neg_sample]).copy()
    training_df = df.drop(index=verification_df.index).copy()

    return training_df, verification_df


def get_xy(df, label_col=LABEL_COL):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y
