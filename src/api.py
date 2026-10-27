from pathlib import Path

import joblib

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the saved machine learning model.
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

# Load the saved model, threshold, and feature names.
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

print("FraudGuard AI model loaded successfully!")
print(f"Decision threshold: {threshold}")
print(f"Number of expected features: {len(features)}")
print(f"Model file: {MODEL_PATH}")

from fastapi import FastAPI

# Create the FastAPI application.
app = FastAPI(
    title="FraudGuard AI",
    description="An API for credit card fraud detection.",
    version="1.0.0",
)


# Home endpoint.
@app.get("/")
def home():
    return {
        "message": "Welcome to FraudGuard AI!",
        "status": "API is running",
        "model_loaded": model is not None,
    }
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate and load the saved model.
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

# Create the API application.
app = FastAPI(
    title="FraudGuard AI",
    description="An API for credit card fraud detection.",
    version="1.0.0",
)


# Home endpoint.
@app.get("/")
def home():
    return {
        "message": "Welcome to FraudGuard AI!",
        "status": "API is running",
        "model_loaded": model is not None,
    }


# Fraud prediction endpoint.
@app.post("/predict")
def predict_transaction(transaction: dict[str, float]):

    # Check which features are missing or unexpected.
    received_features = set(transaction.keys())
    required_features = set(features)

    missing_features = required_features - received_features
    extra_features = received_features - required_features

    if missing_features or extra_features:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": sorted(missing_features),
                "unexpected_features": sorted(extra_features),
                "message": "Provide exactly the 30 features expected by the model.",
            },
        )

    # Arrange the transaction features in the model's expected order.
    transaction_df = pd.DataFrame(
        [[transaction[feature] for feature in features]],
        columns=features,
    )

    # Calculate the probability that the transaction is fraudulent.
    probabilities = model.predict_proba(transaction_df)[0]

    # Find the probability associated with class 1 (fraud).
    fraud_class_index = list(model.classes_).index(1)
    fraud_probability = float(probabilities[fraud_class_index])

    # Classify the transaction using the saved threshold.
    prediction = (
        "Fraudulent"
        if fraud_probability >= threshold
        else "Legitimate"
    )

    # Return the prediction to the user.
    return {
        "fraud_probability": round(fraud_probability, 4),
        "threshold": threshold,
        "prediction": prediction,
    }