from pathlib import Path

import joblib
import pandas as pd

from explain import explain_xgboost, sigmoid

# Locate and load the saved model, threshold, and feature names.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]
fraud_class_index = list(model.classes_).index(1)


def predict(values, decision_threshold=None):
    """Score one transaction and explain it.

    `values` maps each model feature to a number. Returns the fraud probability,
    the decision at `decision_threshold` (the saved threshold by default), the
    base probability, and every feature's contribution in log-odds, largest first.
    """
    decision_threshold = threshold if decision_threshold is None else decision_threshold

    # Arrange the transaction features in the model's expected order.
    transaction_df = pd.DataFrame([[values[f] for f in features]], columns=features)

    fraud_probability = float(model.predict_proba(transaction_df)[0][fraud_class_index])
    base_values, contributions = explain_xgboost(model, transaction_df)

    factors = sorted(
        (
            {"feature": f, "value": float(values[f]), "contribution": float(c)}
            for f, c in zip(features, contributions[0])
        ),
        key=lambda factor: abs(factor["contribution"]),
        reverse=True,
    )

    return {
        "fraud_probability": fraud_probability,
        "threshold": decision_threshold,
        "prediction": "Fraudulent" if fraud_probability >= decision_threshold else "Legitimate",
        "base_probability": float(sigmoid(base_values[0])),
        "factors": factors,
    }
