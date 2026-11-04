import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, FiniteFloat, create_model

# Allow "uvicorn src.api:app" from the project root to import sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import predictor  # noqa: E402

# Decimals kept in responses; fraud probabilities are often tiny and the threshold is low.
PROBABILITY_DECIMALS = 6

# Number of features returned in each prediction's explanation.
TOP_FACTORS = 5

# Request schema: one required, finite number per model feature; unknown fields are rejected.
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {feature: 0.0 for feature in predictor.features}},
    ),
    **{feature: (FiniteFloat, ...) for feature in predictor.features},
)


class Factor(BaseModel):
    feature: str
    value: float
    contribution: float


class Prediction(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: str
    base_probability: float
    top_factors: list[Factor]


# Create the API application.
app = FastAPI(
    title="FraudGuard AI",
    description="An API for credit card fraud detection with per-prediction explanations.",
    version="1.2.0",
)


# Home endpoint.
@app.get("/")
def home():
    return {
        "message": "Welcome to FraudGuard AI!",
        "status": "API is running",
        "docs": "/docs",
    }


# Health endpoint with model details.
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": type(predictor.model).__name__,
        "threshold": predictor.threshold,
        "threshold_method": predictor.artifact.get("threshold_method", "manual"),
        "test_metrics": predictor.artifact.get("test_metrics", {}),
        "n_features": len(predictor.features),
    }


# Fraud prediction endpoint.
@app.post("/predict", response_model=Prediction)
def predict_transaction(transaction: Transaction):
    result = predictor.predict(transaction.model_dump())

    return Prediction(
        fraud_probability=round(result["fraud_probability"], PROBABILITY_DECIMALS),
        threshold=result["threshold"],
        prediction=result["prediction"],
        base_probability=round(result["base_probability"], PROBABILITY_DECIMALS),
        top_factors=[
            Factor(
                feature=factor["feature"],
                value=factor["value"],
                contribution=round(factor["contribution"], 4),
            )
            for factor in result["factors"][:TOP_FACTORS]
        ],
    )
