from pathlib import Path
import pandas as pd

# Locate the project folder and dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

# Load the CSV file into a Pandas DataFrame.
df = pd.read_csv(DATA_PATH)

# Inspect the dataset.
print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n--- First Five Transactions ---")
print(df.head())

print("\n--- Column Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum().sum())

print("\n--- Transaction Class Distribution ---")
print(df["Class"].value_counts().sort_index())

print("\n--- Fraud Percentage ---")
fraud_percentage = df["Class"].mean() * 100
print(f"{fraud_percentage:.4f}%")
