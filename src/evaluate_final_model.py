
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Locate and load the dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]

# Reserve the final test set first.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split the remaining data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

# Train Random Forest using training data only.
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Confirm the candidate threshold using validation data.
val_probabilities = model.predict_proba(X_val)[:, 1]
print(
    "\nValidation average precision:",
    f"{average_precision_score(y_val, val_probabilities):.4f}",
)

# Evaluate the chosen threshold once on the held-out test set.
threshold = 0.40
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

print(f"\n--- Final Test Results (threshold={threshold:.2f}) ---")
print(classification_report(
    y_test, test_predictions, digits=4, zero_division=0
))

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, test_predictions))

print(
    "Average precision:",
    f"{average_precision_score(y_test, test_probabilities):.4f}",
)
print(
    "ROC-AUC:",
    f"{roc_auc_score(y_test, test_probabilities):.4f}",
)
print(f"Test transactions: {len(y_test):,}")
print(f"Fraud cases in test set: {int(y_test.sum())}")


import joblib

# Save the trained model and its decision threshold.
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
joblib.dump(
    {
        "model": model,
        "threshold": threshold,
        "features": list(X.columns),
    },
    MODEL_PATH,
)

print(f"\nTrained model saved to: {MODEL_PATH}")