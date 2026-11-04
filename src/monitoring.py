import json

import matplotlib

matplotlib.use("Agg")  # Save figures to files only; no GUI window.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from data import FIGURES_DIR, PROJECT_ROOT, load_data, make_xgboost, total_cost  # noqa: E402
from drift import PSI_RETRAIN, PSI_WARNING, drift_status, psi  # noqa: E402
from predictor import threshold as THRESHOLD  # noqa: E402

MONITORING_PATH = PROJECT_ROOT / "reports" / "monitoring.json"
BLOCK_HOURS = 8
INITIAL_BLOCKS = 2  # The first model is trained on the first 16 hours.
SLIDING_BLOCKS = 2  # The sliding-window model always uses the latest 16 hours.

# Transactions follow a daily cycle, so each block is compared with the same hours one day earlier.
DAY_BLOCKS = 24 // BLOCK_HOURS

# Time only counts seconds, so it always "drifts"; it is left out of the drift check.
DRIFT_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def score_block(model, X, y):
    """Labelled results for one block at the saved decision threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    flagged = probabilities >= THRESHOLD
    fraud = y.to_numpy() == 1
    return {
        "average_precision": round(float(average_precision_score(y, probabilities)), 4),
        "recall": round(float((fraud & flagged).sum() / fraud.sum()), 4),
        "false_alarms": int((~fraud & flagged).sum()),
        "cost": round(total_cost(y, flagged, X["Amount"]), 2),
    }


# Split the 48 hours of transactions into 8-hour blocks, in time order.
df, X, y = load_data()
block = (X["Time"] // (BLOCK_HOURS * 3600)).astype(int)
n_blocks = int(block.max()) + 1

# The "deployed" model is trained once on the first blocks and never updated.
initial = block < INITIAL_BLOCKS
static_model = make_xgboost().fit(X[initial], y[initial])

results = []
print(f"--- Monitoring Simulation: {BLOCK_HOURS}-hour blocks, threshold {THRESHOLD} ---")
print(
    "Block  | Hours  | Fraud | Max feature PSI      | Score PSI | Status    | "
    "Avg precision: static / expanding / sliding"
)

for b in range(INITIAL_BLOCKS, n_blocks):
    current = block == b
    X_block, y_block = X[current], y[current]

    # Label-free drift signals: available as soon as transactions arrive.
    # The first day has no earlier day to compare with.
    reference = block == b - DAY_BLOCKS
    if reference.any():
        X_reference = X[reference]
        feature_psi = {f: psi(X_reference[f], X_block[f]) for f in DRIFT_FEATURES}
        top_features = sorted(feature_psi.items(), key=lambda item: item[1], reverse=True)[:5]
        score_psi = psi(
            static_model.predict(X_reference, output_margin=True),
            static_model.predict(X_block, output_margin=True),
        )
        # The fraud score is what drives decisions, so it sets the status; features explain why.
        status = drift_status(score_psi)
    else:
        top_features, score_psi, status = [("-", float("nan"))], float("nan"), "no reference"

    # Labelled results: only known later, once fraud is confirmed.
    expanding = block < b
    sliding = (block < b) & (block >= b - SLIDING_BLOCKS)
    strategies = {
        "static": static_model,
        "expanding": make_xgboost().fit(X[expanding], y[expanding]),
        "sliding": make_xgboost().fit(X[sliding], y[sliding]),
    }
    scores = {name: score_block(model, X_block, y_block) for name, model in strategies.items()}

    results.append({
        "block": b,
        "hours": [b * BLOCK_HOURS, (b + 1) * BLOCK_HOURS],
        "transactions": int(current.sum()),
        "fraud": int(y_block.sum()),
        "fraud_amount": round(float(X_block.loc[y_block == 1, "Amount"].sum()), 2),
        "score_psi": None if np.isnan(score_psi) else round(score_psi, 4),
        "top_feature_psi": {f: round(v, 4) for f, v in top_features if not np.isnan(v)},
        "status": status,
        "strategies": scores,
    })

    print(
        f"{b:6d} | {b * BLOCK_HOURS:2d}-{(b + 1) * BLOCK_HOURS:2d}h | {int(y_block.sum()):5d} | "
        f"{top_features[0][0]:>6} {top_features[0][1]:12.4f} | {score_psi:9.4f} | "
        f"{status:9} | "
        + " / ".join(f"{scores[s]['average_precision']:.4f}" for s in strategies)
    )

# Totals over all monitored blocks.
totals = {
    name: {
        "cost": round(sum(r["strategies"][name]["cost"] for r in results), 2),
        "caught_share": round(
            sum(r["strategies"][name]["recall"] * r["fraud"] for r in results)
            / sum(r["fraud"] for r in results), 4
        ),
    }
    for name in ["static", "expanding", "sliding"]
}
print("\nTotal over monitored blocks:")
for name, total in totals.items():
    print(f"  {name:9} cost ${total['cost']:9,.2f}  fraud caught {total['caught_share']:.1%}")

MONITORING_PATH.write_text(json.dumps({
    "block_hours": BLOCK_HOURS,
    "threshold": THRESHOLD,
    "psi_warning": PSI_WARNING,
    "psi_retrain": PSI_RETRAIN,
    "blocks": results,
    "totals": totals,
}, indent=2))
print(f"\nResults saved to: {MONITORING_PATH}")

# Chart: performance per block for each strategy, and the label-free drift signal.
labels = [f"{r['hours'][0]}-{r['hours'][1]}h" for r in results]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for name, marker in [("static", "o"), ("expanding", "s"), ("sliding", "^")]:
    top.plot(labels, [r["strategies"][name]["average_precision"] for r in results], marker=marker, label=name)
top.set_ylabel("Average precision")
top.set_title("Retraining strategies on each new 8-hour block")
top.legend()

bottom.bar(labels, [r["score_psi"] or 0 for r in results], label="Fraud score PSI", alpha=0.7)
bottom.plot(
    labels,
    [max(r["top_feature_psi"].values(), default=np.nan) for r in results],
    "k.-",
    label="Max feature PSI",
)
bottom.axhline(PSI_WARNING, color="orange", linestyle="--", label="Warning")
bottom.axhline(PSI_RETRAIN, color="red", linestyle="--", label="Retrain")
bottom.set_ylabel("PSI vs same hours a day earlier")
bottom.set_xlabel("Block")
bottom.legend(fontsize=8)
fig.tight_layout()
chart_path = FIGURES_DIR / "monitoring.png"
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"Chart saved to: {chart_path}")
