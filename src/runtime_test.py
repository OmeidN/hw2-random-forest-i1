"""
Run the best trained model on the 2 held-out verification samples.
Returns a DataFrame with true_class, predicted_class, prob_class_0, prob_class_1, correct.
"""
import pandas as pd


def run_verification_predictions(model, X_verify, y_verify):
    """
    Predict on verification set; return DataFrame with true class, predicted class,
    class probabilities, and whether the prediction was correct.
    """
    pred = model.predict(X_verify)
    prob = model.predict_proba(X_verify)

    results = pd.DataFrame({
        "true_class": y_verify.values,
        "predicted_class": pred,
        "prob_class_0": prob[:, 0],
        "prob_class_1": prob[:, 1],
        "correct": (pred == y_verify.values),
    })
    return results
