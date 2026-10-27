from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Locate the project and dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Load the dataset.
df = pd.read_csv(DATA_PATH)

# Separate input features (X) from the target (y).
X = df.drop(columns=["Class"])
y = df["Class"]

# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Report the sizes of each set.
print("--- Dataset Split ---")
print(f"Total rows: {len(df):,}")
print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")

print("\n--- Fraud Cases in Each Set ---")
print(f"Training fraud cases: {int(y_train.sum())}")
print(f"Training legitimate cases: {int((y_train == 0).sum())}")
print(f"Test fraud cases: {int(y_test.sum())}")
print(f"Test legitimate cases: {int((y_test == 0).sum())}")

print("\n--- Fraud Percentage ---")
print(f"Original: {y.mean() * 100:.4f}%")
print(f"Training: {y_train.mean() * 100:.4f}%")
print(f"Test: {y_test.mean() * 100:.4f}%")
