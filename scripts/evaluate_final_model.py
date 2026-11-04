import joblib
import matplotlib

matplotlib.use("Agg")  # Save figures to files only; no GUI window.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from fraudguard.data import (  # noqa: E402
    FIGURES_DIR,
    MODEL_PATH,
    REVIEW_COST,
    load_data,
    make_random_forest,
    make_xgboost,
    out_of_fold_probabilities,
    split_data,
    total_cost,
)
from fraudguard.explain import explain_xgboost  # noqa: E402

CANDIDATE_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def cheapest_threshold(y_true, probabilities, amounts):
    costs = [total_cost(y_true, probabilities >= t, amounts) for t in CANDIDATE_THRESHOLDS]
    return float(CANDIDATE_THRESHOLDS[int(np.argmin(costs))]), costs


# Load the dataset. Training + validation data is used for development; the test set once, at the end.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_dev = pd.concat([X_train, X_val])
y_dev = pd.concat([y_train, y_val])
test_amounts = X_test["Amount"].to_numpy()

# Choose the threshold that minimises cost on out-of-fold predictions.
print("Choosing the threshold with 5-fold cross-validation...")
dev_probabilities = out_of_fold_probabilities(make_xgboost(), X_dev, y_dev)
threshold, dev_costs = cheapest_threshold(y_dev, dev_probabilities, X_dev["Amount"].to_numpy())
print(f"Cost-optimal threshold: {threshold:.2f}")

# Plot cross-validated cost against threshold for the report.
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(CANDIDATE_THRESHOLDS, dev_costs)
plt.axvline(threshold, linestyle="--", color="gray", label=f"Chosen threshold = {threshold:.2f}")
plt.title(f"Cross-validated Cost by Threshold (review cost = ${REVIEW_COST:.0f})")
plt.xlabel("Decision threshold")
plt.ylabel("Missed fraud amount + review cost ($)")
plt.legend()
plt.tight_layout()
cost_plot_path = FIGURES_DIR / "threshold_cost.png"
plt.savefig(cost_plot_path, dpi=150)
plt.close()
print(f"Cost plot saved to: {cost_plot_path}")

# Train the final XGBoost model on all development data.
print("\nTraining final XGBoost model...")
model = make_xgboost()
model.fit(X_dev, y_dev)

# Baseline for comparison: the original Random Forest, threshold chosen on validation.
print("Training Random Forest baseline...")
baseline = make_random_forest().fit(X_train, y_train)
baseline_threshold, _ = cheapest_threshold(
    y_val, baseline.predict_proba(X_val)[:, 1], X_val["Amount"].to_numpy()
)
baseline_probabilities = baseline.predict_proba(X_test)[:, 1]

# Evaluate the chosen threshold once on the held-out test set.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_flagged = test_probabilities >= threshold

print(f"\n--- Final Test Results: XGBoost (threshold={threshold:.2f}) ---")
print(classification_report(y_test, test_flagged.astype(int), digits=4, zero_division=0))

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, test_flagged.astype(int)))

test_average_precision = average_precision_score(y_test, test_probabilities)
test_roc_auc = roc_auc_score(y_test, test_probabilities)
test_cost = total_cost(y_test, test_flagged, test_amounts)

print(f"Average precision: {test_average_precision:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"Test transactions: {len(y_test):,}")
print(f"Fraud cases in test set: {int(y_test.sum())}")

print("\n--- Test Set: Baseline vs Final ---")
print("Model                          | Threshold | Avg prec | Caught | Missed | False alarms |  Cost")
for name, probabilities, t in [
    ("Random Forest (baseline)", baseline_probabilities, baseline_threshold),
    ("XGBoost (final)", test_probabilities, threshold),
]:
    flagged = probabilities >= t
    fraud = y_test.to_numpy() == 1
    print(
        f"{name:30} | {t:9.2f} | {average_precision_score(y_test, probabilities):8.4f} | "
        f"{int((fraud & flagged).sum()):6d} | {int((fraud & ~flagged).sum()):6d} | "
        f"{int((~fraud & flagged).sum()):12d} | ${total_cost(y_test, flagged, test_amounts):,.2f}"
    )
print(f"Total fraud amount in test set: ${test_amounts[y_test.to_numpy() == 1].sum():,.2f}")

# Look at the fraud the model missed: how confident was it, and why?
missed = X_test[(y_test.to_numpy() == 1) & ~test_flagged]
if len(missed):
    missed_probabilities = model.predict_proba(missed)[:, 1]
    _, contributions = explain_xgboost(model, missed)

    print(f"\n--- Missed Fraud ({len(missed)} transactions, ${missed['Amount'].sum():,.2f}) ---")
    print(f"Median fraud probability of missed fraud: {np.median(missed_probabilities):.4f}")
    print("Largest missed fraud and the features that made them look legitimate:")
    for i in np.argsort(-missed["Amount"].to_numpy())[:5]:
        reasons = ", ".join(
            f"{missed.columns[j]} ({contributions[i, j]:+.2f})" for j in np.argsort(contributions[i])[:3]
        )
        print(
            f"  ${missed['Amount'].iloc[i]:8,.2f}  probability {missed_probabilities[i]:.4f}  "
            f"strongest 'legitimate' signals: {reasons}"
        )

# Save the trained model, its decision threshold, and how it was chosen.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "threshold": threshold,
        "features": list(X.columns),
        "threshold_method": f"minimum 5-fold CV cost (review cost ${REVIEW_COST:.2f})",
        "test_metrics": {
            "average_precision": round(float(test_average_precision), 4),
            "roc_auc": round(float(test_roc_auc), 4),
            "cost": round(test_cost, 2),
        },
    },
    MODEL_PATH,
)

print(f"\nTrained model saved to: {MODEL_PATH}")
