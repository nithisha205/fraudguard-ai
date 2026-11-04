from data import load_data

# Load the CSV file into a Pandas DataFrame.
df, X, y = load_data()

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
