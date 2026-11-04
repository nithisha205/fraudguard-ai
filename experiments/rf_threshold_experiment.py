from sklearn.metrics import average_precision_score
from threshold_report import print_threshold_table

from fraudguard.data import load_data, make_random_forest, split_data

# Load the dataset and create the shared 60/20/20 split.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Train Random Forest on training data only.
model = make_random_forest()

print("Training Random Forest...")
model.fit(X_train, y_train)

# Get fraud probabilities for validation transactions.
val_probabilities = model.predict_proba(X_val)[:, 1]

print("\n--- Random Forest: Validation Threshold Results ---")
print_threshold_table(y_val, val_probabilities)

print(
    "\nValidation average precision:",
    f"{average_precision_score(y_val, val_probabilities):.4f}",
)
print(f"Validation transactions: {len(y_val):,}")
print(f"Final test transactions reserved: {len(y_test):,}")
