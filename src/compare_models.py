import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    RANDOM_STATE,
    REVIEW_COST,
    load_data,
    make_random_forest,
    make_xgboost,
    out_of_fold_probabilities,
    split_data,
    total_cost,
)

CANDIDATE_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

# Compare models with 5-fold cross-validation on the training + validation data.
# The test set stays untouched for the final evaluation.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_dev = pd.concat([X_train, X_val])
y_dev = pd.concat([y_train, y_val])
amounts = X_dev["Amount"].to_numpy()

models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "Random Forest": make_random_forest(),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    ),
    "XGBoost": make_xgboost(),
    "XGBoost + isotonic calibration": CalibratedClassifierCV(
        make_xgboost(), method="isotonic", cv=3
    ),
}

print(f"--- Model Comparison (5-fold CV, {len(y_dev):,} transactions, {int(y_dev.sum())} fraud) ---")
print(f"Costs: missed fraud = transaction amount, false alarm = ${REVIEW_COST:.0f} review\n")
print(
    f"{'Model':32} | {'Avg prec':>8} | {'Brier':>7} | {'Best thr':>8} | {'Cost':>8} | "
    f"{'Missed':>6} | {'False al':>8} | {'Exp-cost rule':>13}"
)

for name, model in models.items():
    probabilities = out_of_fold_probabilities(model, X_dev, y_dev)

    # Rule 1: one threshold for every transaction, chosen to minimise cost.
    costs = [total_cost(y_dev, probabilities >= t, amounts) for t in CANDIDATE_THRESHOLDS]
    best_threshold = CANDIDATE_THRESHOLDS[int(np.argmin(costs))]
    flagged = probabilities >= best_threshold

    # Rule 2: flag when the expected fraud loss is larger than the review cost.
    expected_cost_rule = total_cost(y_dev, probabilities * amounts >= REVIEW_COST, amounts)

    print(
        f"{name:32} | {average_precision_score(y_dev, probabilities):8.4f} | "
        f"{brier_score_loss(y_dev, probabilities):7.5f} | {best_threshold:8.2f} | "
        f"${min(costs):7,.0f} | {int(((y_dev == 1) & ~flagged).sum()):6d} | "
        f"{int(((y_dev == 0) & flagged).sum()):8d} | ${expected_cost_rule:12,.0f}"
    )

print(f"\nTotal fraud amount in these transactions: ${amounts[y_dev.to_numpy() == 1].sum():,.0f}")
print("Best thresholds are chosen on the same predictions they are scored on, so costs are slightly optimistic.")
