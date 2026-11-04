import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data import load_data, make_xgboost, split_data, total_cost

CANDIDATE_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def evaluate(name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train on the past, pick the cost-optimal threshold on validation, score the test set once."""
    model = make_xgboost().fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_costs = [
        total_cost(y_val, val_probabilities >= t, X_val["Amount"]) for t in CANDIDATE_THRESHOLDS
    ]
    threshold = CANDIDATE_THRESHOLDS[int(np.argmin(val_costs))]

    probabilities = model.predict_proba(X_test)[:, 1]
    flagged = probabilities >= threshold
    fraud = y_test.to_numpy() == 1

    print(
        f"{name:28} | {average_precision_score(y_test, probabilities):8.4f} | "
        f"{roc_auc_score(y_test, probabilities):7.4f} | {threshold:9.2f} | "
        f"{int((fraud & flagged).sum()):3d} / {int(fraud.sum()):3d} | {int((~fraud & flagged).sum()):12d} | "
        f"${total_cost(y_test, flagged, X_test['Amount']):9,.2f} of ${X_test['Amount'][fraud].sum():9,.2f}"
    )


df, X, y = load_data()

print("--- Random vs Time-based Split: XGBoost, 60% train / 20% validation / 20% test ---")
print(
    f"{'Split':28} | {'Avg prec':>8} | {'ROC-AUC':>7} | {'Threshold':>9} | {'Caught':>9} | "
    f"{'False alarms':>12} | Cost / total fraud amount"
)

# Random split: test transactions come from the same period as training.
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
evaluate("Random (stratified)", X_train, y_train, X_val, y_val, X_test, y_test)

# Time-based split: train on the earliest 60%, validate on the next 20%, test on the latest 20%.
# This is how the model would be used: trained on the past, scoring future transactions.
order = np.argsort(X["Time"].to_numpy(), kind="stable")
train_end, val_end = int(len(order) * 0.6), int(len(order) * 0.8)
train_idx, val_idx, test_idx = order[:train_end], order[train_end:val_end], order[val_end:]
evaluate(
    "Time-based (past -> future)",
    X.iloc[train_idx], y.iloc[train_idx],
    X.iloc[val_idx], y.iloc[val_idx],
    X.iloc[test_idx], y.iloc[test_idx],
)

hours = X["Time"].iloc[test_idx] / 3600
print(
    f"\nTime-based test period: hours {hours.min():.1f}-{hours.max():.1f} of the 48-hour dataset, "
    f"fraud rate {y.iloc[test_idx].mean() * 100:.3f}% vs {y.iloc[train_idx].mean() * 100:.3f}% in training."
)
