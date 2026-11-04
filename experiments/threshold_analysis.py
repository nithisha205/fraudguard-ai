from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threshold_report import print_threshold_table

from fraudguard.data import RANDOM_STATE, load_data, split_data

# Load the dataset and create the shared 60/20/20 split.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Train only on the training set.
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

print("Training Logistic Regression...")
model.fit(X_train, y_train)

# Get fraud probabilities for validation transactions.
val_probabilities = model.predict_proba(X_val)[:, 1]

print("\n--- Validation Results by Threshold ---")
print_threshold_table(y_val, val_probabilities)

print(
    "\nValidation average precision:",
    f"{average_precision_score(y_val, val_probabilities):.4f}",
)

print("\nFinal test set remains untouched for final evaluation.")
print(f"Validation transactions: {len(y_val):,}")
print(f"Final test transactions reserved: {len(y_test):,}")
