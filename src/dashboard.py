import altair as alt
import pandas as pd
import streamlit as st

import predictor
from data import PROJECT_ROOT, total_cost

SAMPLES_PATH = PROJECT_ROOT / "samples" / "sample_transactions.csv"
COST_PLOT_PATH = PROJECT_ROOT / "reports" / "figures" / "threshold_cost.png"
FACTORS_SHOWN = 10

st.set_page_config(page_title="FraudGuard AI", layout="wide")


@st.cache_data
def load_samples():
    """Held-out test transactions with their true labels and fraud probabilities."""
    samples = pd.read_csv(SAMPLES_PATH, index_col="Transaction")
    probabilities = predictor.model.predict_proba(samples[predictor.features])
    samples["Probability"] = probabilities[:, predictor.fraud_class_index]
    return samples


samples = load_samples()

# Sidebar: decision settings shared by every tab.
st.sidebar.title("FraudGuard AI")
st.sidebar.caption("XGBoost fraud detection with per-prediction explanations.")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(predictor.threshold),
    step=0.01,
    help=f"Flag a transaction when its fraud probability is at least this. Saved model: {predictor.threshold}.",
)
review_cost = st.sidebar.number_input(
    "Review cost per false alarm ($)", min_value=0.0, value=5.0, step=1.0
)

tab_check, tab_overview, tab_model = st.tabs(["Check a transaction", "Sample overview", "Model"])

with tab_check:
    show = st.radio(
        "Show", ["All", "Fraud only", "Legitimate only"], horizontal=True, label_visibility="collapsed"
    )
    choices = samples
    if show == "Fraud only":
        choices = samples[samples["Class"] == 1]
    elif show == "Legitimate only":
        choices = samples[samples["Class"] == 0]

    transaction_id = st.selectbox(
        "Transaction",
        choices.index,
        format_func=lambda i: (
            f"#{i} · ${samples.at[i, 'Amount']:,.2f} · "
            f"{'Fraud' if samples.at[i, 'Class'] == 1 else 'Legitimate'}"
        ),
    )
    values = samples.loc[transaction_id, predictor.features].to_dict()
    original = predictor.predict(values, threshold)

    # What-if: change the most influential features and the amount.
    with st.expander("What if? Change the most influential features"):
        edit_features = [f["feature"] for f in original["factors"][:3] if f["feature"] != "Amount"]
        columns = st.columns(len(edit_features) + 1)
        for column, feature in zip(columns, edit_features + ["Amount"]):
            values[feature] = column.number_input(
                feature, value=float(values[feature]), key=f"{transaction_id}-{feature}"
            )

    result = predictor.predict(values, threshold)
    probability = result["fraud_probability"]
    actual = "Fraud" if samples.at[transaction_id, "Class"] == 1 else "Legitimate"
    expected_loss = probability * values["Amount"]

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Fraud probability",
        f"{probability:.2%}",
        delta=None if probability == original["fraud_probability"]
        else f"{probability - original['fraud_probability']:+.2%}",
        delta_color="inverse",
    )
    metric_columns[1].metric("Decision", result["prediction"])
    metric_columns[2].metric("Actually", actual)
    metric_columns[3].metric(
        "Expected fraud loss",
        f"${expected_loss:,.2f}",
        help="Fraud probability × amount. Compare with the review cost.",
    )

    if result["prediction"] == "Fraudulent" and actual == "Legitimate":
        st.warning("False alarm: this legitimate transaction would be sent for review.")
    elif result["prediction"] == "Legitimate" and actual == "Fraud":
        st.error("Missed fraud: the model considers this transaction normal.")

    st.subheader("Why?")
    st.caption(
        f"Every transaction starts at the average fraud probability ({result['base_probability']:.3%}). "
        "Bars show how much each feature pushed it towards fraud (red) or legitimate (green), in log-odds."
    )
    factors = pd.DataFrame(result["factors"][:FACTORS_SHOWN])
    factors["Direction"] = factors["contribution"].map(
        lambda c: "Towards fraud" if c > 0 else "Towards legitimate"
    )
    chart = (
        alt.Chart(factors)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution (log-odds)"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Towards fraud", "Towards legitimate"], range=["#d64545", "#2f9e6b"]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                "feature",
                alt.Tooltip("value:Q", format=".3f"),
                alt.Tooltip("contribution:Q", format="+.3f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

with tab_overview:
    flagged = samples["Probability"] >= threshold
    fraud = samples["Class"] == 1

    st.caption(
        f"{len(samples)} held-out test transactions: all {int(fraud.sum())} test frauds "
        f"and {int((~fraud).sum())} random legitimate ones. Move the sidebar settings to see the trade-off."
    )
    overview_columns = st.columns(4)
    overview_columns[0].metric("Fraud caught", f"{int((fraud & flagged).sum())} / {int(fraud.sum())}")
    overview_columns[1].metric("Missed fraud", f"${samples.loc[fraud & ~flagged, 'Amount'].sum():,.2f}")
    overview_columns[2].metric("False alarms", int((~fraud & flagged).sum()))
    overview_columns[3].metric(
        "Total cost", f"${total_cost(fraud, flagged, samples['Amount'], review_cost):,.2f}"
    )

    table = samples[["Amount", "Probability", "Class"]].assign(
        Actual=fraud.map({True: "Fraud", False: "Legitimate"}),
        Decision=flagged.map({True: "Flagged", False: "Passed"}),
    )
    st.dataframe(
        table.drop(columns="Class").sort_values("Probability", ascending=False),
        column_config={
            "Amount": st.column_config.NumberColumn(format="$%.2f"),
            "Probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.4f"),
        },
        width="stretch",
    )

with tab_model:
    metrics = predictor.artifact.get("test_metrics", {})
    model_columns = st.columns(4)
    model_columns[0].metric("Model", type(predictor.model).__name__)
    model_columns[1].metric("Saved threshold", predictor.threshold)
    model_columns[2].metric("Test average precision", metrics.get("average_precision", "n/a"))
    model_columns[3].metric("Test ROC-AUC", metrics.get("roc_auc", "n/a"))
    st.caption(f"Threshold method: {predictor.artifact.get('threshold_method', 'manual')}.")
    if COST_PLOT_PATH.exists():
        st.image(str(COST_PLOT_PATH), caption="Cross-validated cost by threshold")
