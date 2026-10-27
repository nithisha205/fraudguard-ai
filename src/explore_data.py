from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Locate the project folder and dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Create the output folder if it does not exist.
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the dataset.
df = pd.read_csv(DATA_PATH)

# Count legitimate and fraudulent transactions.
class_counts = df["Class"].value_counts().sort_index()
class_percentages = df["Class"].value_counts(normalize=True).sort_index() * 100

print("--- Transaction Counts ---")
print(class_counts.rename(index={0: "Legitimate", 1: "Fraud"}))

print("\n--- Transaction Percentages ---")
print(class_percentages.rename(index={0: "Legitimate", 1: "Fraud"}).round(4))

# Draw the bar chart.
labels = ["Legitimate (0)", "Fraud (1)"]
counts = [class_counts.get(0, 0), class_counts.get(1, 0)]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts)
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

# Add the exact count above each bar.
for bar in bars:
    count = int(bar.get_height())
    plt.annotate(
        f"{count:,}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
        xytext=(0, 3),
        textcoords="offset points"
    )

plt.tight_layout()

# Save the chart for our project report.
output_path = FIGURES_DIR / "class_distribution.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nChart saved to: {output_path}")


# Compare transaction amounts by class.
print("\n--- Transaction Amount Summary ---")
print(
    df.groupby("Class")["Amount"]
    .describe()
    .rename(index={0: "Legitimate", 1: "Fraud"})
)

# Draw box plots for transaction amounts.
plt.figure(figsize=(8, 5))

amounts_by_class = [
    df.loc[df["Class"] == 0, "Amount"],
    df.loc[df["Class"] == 1, "Amount"],
]

plt.boxplot(
    amounts_by_class,
    tick_labels=["Legitimate (0)", "Fraud (1)"],
    showfliers=False
)

plt.yscale("log")
plt.title("Transaction Amounts by Class (Log Scale)")
plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount (log scale)")
plt.tight_layout()

amount_plot_path = FIGURES_DIR / "amount_boxplot.png"
plt.savefig(amount_plot_path, dpi=150)
plt.show()

print(f"\nBox plot saved to: {amount_plot_path}")
