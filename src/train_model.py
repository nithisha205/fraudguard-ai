from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import RANDOM_STATE, load_data, split_data

# Baseline: Logistic Regression trained on the training split and
# checked on the validation split (the test split stays untouched).
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Build a pipeline: scale features, then train the classifier.
model = Pipeline(
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
)

# Learn patterns using training data only.
print("Training Logistic Regression...")
model.fit(X_train, y_train)

# Evaluate on the validation set.
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("\n--- Classification Report (Validation) ---")
print(classification_report(y_val, y_pred, digits=4, zero_division=0))

print("--- Confusion Matrix ---")
print("Rows = actual class; columns = predicted class")
print(confusion_matrix(y_val, y_pred))

print("\n--- Evaluation Metrics ---")
print(f"Average precision (PR-AUC summary): {average_precision_score(y_val, y_prob):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
