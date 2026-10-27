# FraudGuard AI — API Development and Testing

## 1. Overview

FraudGuard AI exposes a trained Random Forest classification model through a REST API built with FastAPI.

The API accepts transaction features, calculates an estimated fraud probability, and returns a classification based on the configured decision threshold.

## 2. Architecture

The prediction workflow is:

1. A client submits transaction features as JSON.
2. FastAPI receives the request.
3. The API validates that the expected features are present.
4. The features are arranged in the order expected by the trained model.
5. The Random Forest model calculates class probabilities.
6. The fraud probability is compared with the saved threshold.
7. The API returns the probability, threshold, and predicted classification.

## 3. Main Components

| Component | Purpose |
|---|---|
| `src/api.py` | Implements the API endpoints and prediction logic |
| `models/fraud_model.joblib` | Stores the trained model, decision threshold, and feature names |
| FastAPI | Handles API requests and responses |
| Uvicorn | Runs the local API server |
| Swagger UI | Provides interactive API documentation |

## 4. Available Endpoints

### GET /

**Purpose:** Confirm that the API is running.

Example response:

```json
{
  "message": "Welcome to FraudGuard AI!",
  "status": "API is running",
  "model_loaded": true
}
```

### POST /predict

**Purpose:** Receive transaction features and return a fraud prediction.

The request must contain all 30 input features:

- `Time`
- `V1` through `V28`
- `Amount`

The target column `Class` must not be included because it is the actual label used for evaluation, not an input feature.

Example request structure:

```json
{
  "Time": 0.0,
  "V1": -1.35,
  "V2": -0.07,
  "V3": 2.53,
  "V4": 1.37,
  "V5": -0.33,
  "V6": 0.46,
  "V7": 0.23,
  "V8": 0.09,
  "V9": 0.36,
  "V10": 0.09,
  "V11": -0.55,
  "V12": -0.61,
  "V13": -0.99,
  "V14": -0.31,
  "V15": 1.46,
  "V16": -0.47,
  "V17": 0.20,
  "V18": 0.02,
  "V19": 0.40,
  "V20": 0.25,
  "V21": -0.01,
  "V22": 0.27,
  "V23": -0.11,
  "V24": 0.06,
  "V25": 0.12,
  "V26": -0.18,
  "V27": 0.13,
  "V28": -0.02,
  "Amount": 149.62
}
```

The values above illustrate the request structure. For reproducible tests, use the complete transaction examples recorded from the dataset.

Example response structure:

```json
{
  "fraud_probability": 0.61,
  "threshold": 0.4,
  "prediction": "Fraudulent"
}
```

The probability in this example is illustrative and should not be interpreted as a result for every transaction.

## 5. Running the API Locally

Activate the project's virtual environment, then run the following command from the main project directory:

```powershell
python -m uvicorn src.api:app --reload
```

The API will be available locally at:

`http://127.0.0.1:8000`

The interactive documentation is available at:

`http://127.0.0.1:8000/docs`

Keep the terminal running while testing the API. Press `Ctrl + C` in the terminal to stop the server.

## 6. Manual Testing Procedure

The prediction endpoint was tested through the Swagger UI.

Testing procedure:

1. Start the FastAPI application.
2. Open `/docs` in a browser.
3. Expand `POST /predict`.
4. Click **Try it out**.
5. Enter the transaction features as JSON.
6. Click **Execute**.
7. Inspect the HTTP response and returned prediction.

## 7. Test Case 1: Fraudulent Transaction

**Input:** A transaction selected from the dataset with the actual label `Class = 1`.

**Observed API result:**
- Fraud probability: `0.61`
- Threshold: `0.40`
- Prediction: `Fraudulent`

**Outcome:** The prediction matched the known dataset label.

## 8. Test Case 2: Legitimate Transaction

**Input:** A transaction selected from the dataset with the actual label `Class = 0`.

**Observed API result:**
- Prediction: `Legitimate`

**Outcome:** The prediction matched the known dataset label.

The actual fraud probability for this second test should be recorded from the API response if a complete test record is required.

## 9. Current Limitations

- The API has been tested locally, not in a production environment.
- Only two individual transaction examples have been manually tested through the API.
- Manual tests do not replace automated testing.
- The endpoint checks for missing and unexpected feature names, but additional validation is needed for non-finite values and malformed inputs.
- The current API does not include authentication, rate limiting, or production monitoring.
- The predicted probability is not guaranteed to be calibrated.

## 10. Planned Testing Improvements

Future work includes:

1. Automated endpoint tests using `pytest`.
2. Validation tests for missing or unexpected features.
3. Tests for malformed and non-finite numeric inputs.
4. Tests for successful and unsuccessful prediction scenarios.
5. Integration testing between the API and the saved model.

## 11. Conclusion

The FastAPI application successfully loads the saved model and returns fraud predictions through the `POST /predict` endpoint.

The two manual tests demonstrated the complete request-to-prediction workflow using examples from the dataset. Further automated testing is needed to assess the reliability of the API across a wider range of inputs.