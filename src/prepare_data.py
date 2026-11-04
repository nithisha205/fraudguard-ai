from data import load_data, split_data

# Load the dataset and create the shared 60/20/20 split.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Report the sizes of each set.
print("--- Dataset Split ---")
print(f"Total rows: {len(df):,}")
print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Test rows: {len(X_test):,}")

print("\n--- Fraud Cases in Each Set ---")
for name, labels in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"{name} fraud cases: {int(labels.sum())}")
    print(f"{name} legitimate cases: {int((labels == 0).sum())}")

print("\n--- Fraud Percentage ---")
print(f"Original: {y.mean() * 100:.4f}%")
print(f"Training: {y_train.mean() * 100:.4f}%")
print(f"Validation: {y_val.mean() * 100:.4f}%")
print(f"Test: {y_test.mean() * 100:.4f}%")
