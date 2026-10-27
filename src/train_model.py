
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Locate and load the dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

# Separate features (X) from the target (y).
X = df.drop(columns=["Class"])
y = df["Class"]

# Keep the same reproducible, stratified 80/20 split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Build a pipeline: scale features, then train the classifier.
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

# Learn patterns using training data only.
print("Training Logistic Regression...")
model.fit(X_train, y_train)

# Evaluate on the held-out test set.
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

print("--- Confusion Matrix ---")
print("Rows = actual class; columns = predicted class")
print(confusion_matrix(y_test, y_pred))

print("\n--- Evaluation Metrics ---")
print(f"Average precision (PR-AUC summary): {average_precision_score(y_test, y_prob):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
