import pandas as pd

from data import PROJECT_ROOT, RANDOM_STATE, load_data, split_data

SAMPLES_PATH = PROJECT_ROOT / "samples" / "sample_transactions.csv"

# Take a small sample from the test set for the dashboard: every fraud plus 200 legitimate transactions.
# The model never trained on these rows.
df, X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
test = X_test.assign(Class=y_test)

sample = pd.concat([
    test[test["Class"] == 1],
    test[test["Class"] == 0].sample(200, random_state=RANDOM_STATE),
]).sort_index()

SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(SAMPLES_PATH, index_label="Transaction")

print(f"Saved {len(sample)} transactions ({int(sample['Class'].sum())} fraud) to: {SAMPLES_PATH}")
