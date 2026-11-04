import numpy as np
from xgboost import DMatrix


def explain_xgboost(model, X):
    """Split an XGBoost prediction into per-feature contributions (TreeSHAP).

    Contributions are in log-odds: positive values push the transaction towards
    fraud, negative values away from it. The base value plus all contributions
    equals the model's log-odds, so sigmoid(base + sum) is the fraud probability.
    """
    shap_values = model.get_booster().predict(DMatrix(X), pred_contribs=True)
    # The last column is the base value shared by every transaction.
    return shap_values[:, -1], shap_values[:, :-1]


def sigmoid(log_odds):
    return 1.0 / (1.0 + np.exp(-log_odds))
