import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudguard import api, predictor
from fraudguard.data import total_cost
from fraudguard.explain import explain_xgboost, sigmoid

client = TestClient(api.app)


@pytest.fixture
def transaction():
    return {feature: 0.0 for feature in predictor.features}


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] == 30


def test_predict_returns_probability_and_explanation(transaction):
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in {"Fraudulent", "Legitimate"}
    assert len(body["top_factors"]) == api.TOP_FACTORS


def test_suspicious_transaction_scores_higher(transaction):
    normal = client.post("/predict", json=transaction).json()

    # Strongly negative V14/V12/V10/V17 are typical of fraud in this dataset.
    suspicious = {**transaction, "V14": -15.0, "V12": -12.0, "V10": -10.0, "V17": -10.0}
    flagged = client.post("/predict", json=suspicious).json()

    assert flagged["fraud_probability"] > normal["fraud_probability"]

    # The changed features should be explained as pushing towards fraud.
    pushes = {f["feature"]: f["contribution"] for f in flagged["top_factors"]}
    assert pushes["V14"] > 0 and pushes["V17"] > 0


def test_explanation_adds_up_to_probability(transaction):
    transaction["V14"] = -8.0
    row = pd.DataFrame([transaction], columns=predictor.features)

    base, contributions = explain_xgboost(predictor.model, row)
    probability = predictor.model.predict_proba(row)[0][predictor.fraud_class_index]

    assert sigmoid(base[0] + contributions[0].sum()) == pytest.approx(probability, abs=1e-5)


def test_missing_feature_is_rejected(transaction):
    del transaction["Amount"]
    assert client.post("/predict", json=transaction).status_code == 422


def test_unexpected_feature_is_rejected(transaction):
    transaction["Merchant"] = 1.0
    assert client.post("/predict", json=transaction).status_code == 422


def test_non_numeric_value_is_rejected(transaction):
    transaction["Amount"] = "abc"
    assert client.post("/predict", json=transaction).status_code == 422


def test_total_cost_counts_missed_fraud_amount_and_reviews():
    y_true = [1, 1, 0, 0]
    flagged = [True, False, True, False]
    amounts = [100.0, 40.0, 7.0, 3.0]

    # One missed fraud ($40) plus one false alarm reviewed at $5.
    assert total_cost(y_true, flagged, amounts, review_cost=5.0) == 45.0
