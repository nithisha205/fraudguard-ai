from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Load the dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Class"])
y = df["Class"]

# Create a reproducible 60% training / 20% validation / 20% test split.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

# Train Random Forest on training data only.
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Get fraud probabilities for validation transactions.
val_probabilities = model.predict_proba(X_val)[:, 1]

print("\n--- Random Forest: Validation Threshold Results ---")
print("Threshold | Precision | Recall | F1-score | False positives | Missed fraud")

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print(
        f"{threshold:9.2f} | {precision:9.4f} | {recall:6.4f} | "
        f"{f1:8.4f} | {fp:15,d} | {fn:12,d}"
    )

print(
    "\nValidation average precision:",
    f"{average_precision_score(y_val, val_probabilities):.4f}",
)
print(f"Validation transactions: {len(y_val):,}")
print(f"Final test transactions reserved: {len(y_test):,}")
