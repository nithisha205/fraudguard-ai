import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudguard.data import (
    cross_val_folds,
    load_data,
    make_xgboost,
    out_of_fold_probabilities,
    split_data,
    total_cost,
)

CANDIDATE_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def add_hour_features(X):
    """Hour of day from Time (seconds since the first transaction), as hour and as a 24-hour circle."""
    X = X.copy()
    hour = (X["Time"] // 3600) % 24
    X["Hour"] = hour
    X["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return X


# Compare feature sets with 5-fold cross-validation on training + validation data.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_dev = pd.concat([X_train, X_val])
y_dev = pd.concat([y_train, y_val])
amounts = X_dev["Amount"].to_numpy()

# Trees split on order, so log(Amount) would change nothing; only new information is tested.
feature_sets = {
    "Original 30 features": X_dev,
    "+ hour of day": add_hour_features(X_dev),
    "+ hour of day, without Time": add_hour_features(X_dev).drop(columns=["Time"]),
}

print(f"--- Feature Experiment: XGBoost, 5-fold CV ({int(y_dev.sum())} fraud) ---")
print(f"{'Feature set':30} | {'Avg prec':>8} | {'Fold range':>13} | {'Best thr':>8} | {'Cost':>8} | Missed")

for name, features in feature_sets.items():
    probabilities = out_of_fold_probabilities(make_xgboost(), features, y_dev)
    costs = [total_cost(y_dev, probabilities >= t, amounts) for t in CANDIDATE_THRESHOLDS]
    best = int(np.argmin(costs))
    flagged = probabilities >= CANDIDATE_THRESHOLDS[best]

    fold_scores = [
        average_precision_score(y_dev.iloc[idx], probabilities[idx])
        for _, idx in cross_val_folds().split(features, y_dev)
    ]

    print(
        f"{name:30} | {average_precision_score(y_dev, probabilities):8.4f} | "
        f"{min(fold_scores):.3f}-{max(fold_scores):.3f} | {CANDIDATE_THRESHOLDS[best]:8.2f} | "
        f"${costs[best]:7,.0f} | {int(((y_dev == 1) & ~flagged).sum())}"
    )
