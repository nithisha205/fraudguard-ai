# FraudGuard AI — Project Development Log

## 1. Project Overview

FraudGuard AI is a machine learning application that estimates the probability of credit card fraud using a Random Forest classifier.

The project demonstrates an end-to-end machine learning workflow, including data exploration, model experimentation, evaluation, model persistence, and API-based inference.

**Current status:** The model has been trained and evaluated, and a local FastAPI application successfully returns predictions for manually tested transactions.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| API framework | FastAPI |
| Development server | Uvicorn |
| API testing | FastAPI Swagger UI |
| Development environment | Visual Studio Code |
| Version control | Git and GitHub |

## 3. Data Exploration

The project uses a credit card fraud detection dataset containing 284,807 transactions, 30 input features, and a target column named `Class`.

The target values are:
- `0`: Legitimate transaction
- `1`: Fraudulent transaction

Only 492 transactions are labeled as fraudulent, representing approximately 0.17% of the dataset.

This class imbalance influenced the modeling and evaluation strategy. Accuracy alone was not considered sufficient for assessing fraud detection performance.

The exploratory analysis examined the dataset structure, missing values, class distribution, and transaction amounts.

**Detailed findings:** See [`figures/eda_notes.md`](figures/eda_notes.md).

## 4. Model Experimentation

The project explored Logistic Regression and Random Forest as candidate classification algorithms.

The experiments examined:
- Performance differences between the two algorithms.
- The effect of class weighting on model training.
- How classification thresholds affect precision, recall, F1-score, false positives, and missed fraud cases.

Random Forest was selected as the final model candidate based on the recorded experimental results.

The selected model uses:
- 100 decision trees.
- `class_weight="balanced"`.
- `random_state=42`.
- `n_jobs=-1`.

A classification threshold of `0.40` was selected for the final evaluation and API prediction workflow.

The detailed experimental results should be maintained separately from this development log.

## 5. Model Evaluation

The selected Random Forest model was evaluated on a held-out test set using fraud-specific metrics and ranking metrics.

The recorded results were:
- Fraud precision: 0.8723.
- Fraud recall: 0.8367.
- Fraud F1-score: 0.8542.
- Accuracy: 0.9995.
- Average precision: 0.8603.
- ROC-AUC: 0.9623.

The recorded confusion matrix indicates 82 detected fraud cases, 16 missed fraud cases, and 12 legitimate transactions incorrectly flagged as fraudulent.

These figures should be verified against the final evaluation script output before publication.

**Detailed results and limitations:** See [`model_evaluation.md`](model_evaluation.md).

## 6. Model Persistence

The trained model artifact is saved at:

`models/fraud_model.joblib`

The saved artifact contains:
- The trained Random Forest classifier.
- The selected classification threshold.
- The expected input feature names.

The API loads this artifact when the application starts, allowing it to make predictions without retraining the model on every request.

## 7. API Implementation

The API is implemented in `src/api.py` using FastAPI.

### `GET /`

The home endpoint returns a welcome message, API status, and an indication of whether the model was loaded.

### `POST /predict`

The prediction endpoint:
1. Accepts transaction features as JSON.
2. Checks for missing or unexpected features.
3. Arranges the 30 input features in the model's expected order.
4. Calculates the estimated fraud probability.
5. Applies the saved classification threshold.
6. Returns the probability, threshold, and predicted classification.

The endpoint expects the model's 30 input features. The target column, `Class`, must not be included in the request.

FastAPI provides interactive documentation through the automatically generated `/docs` page.

## 8. Manual API Testing

The API was run locally and tested through the Swagger UI.

Two dataset transactions were submitted:

| Test case | Known dataset label | API result | Outcome |
|---|---|---|---|
| Fraudulent transaction | Fraudulent | Fraudulent | Correct |
| Legitimate transaction | Legitimate | Legitimate | Correct |

The fraudulent example returned a fraud probability of `0.61` against a threshold of `0.40`.

Both manual examples matched their known dataset labels. These tests confirmed basic prediction functionality but do not establish generalization performance or guarantee correct predictions on unseen transactions.

**Testing details:** See [`api_testing.md`](api_testing.md).

## 9. Current Project Status

### Completed

- Dataset inspection and exploratory data analysis.
- Model experimentation and threshold analysis.
- Random Forest training and held-out evaluation.
- Model artifact persistence.
- FastAPI application implementation.
- Model loading and prediction endpoints.
- Manual API tests using one known fraudulent transaction and one known legitimate transaction.
- Initial project and model documentation.

### Planned Improvements

1. Add automated tests using pytest.
2. Strengthen input validation and error handling.
3. Add a reproducible dependency file such as `requirements.txt`.
4. Improve the GitHub README and setup instructions.
5. Consider a Streamlit interface for interactive demonstrations.
6. Explore deployment, monitoring, and additional validation.

These are planned improvements, not completed features.

## 10. Limitations

- The project uses historical, anonymized transaction data.
- The `V1`–`V28` features do not expose their original business meanings.
- The test set contains a limited number of fraudulent transactions, so reported metrics have sampling uncertainty.
- Manual API testing covered only two example transactions.
- The API has been tested locally; production deployment has not been established.
- Model predictions are estimates and should not be treated as definitive fraud determinations.

Further validation, security controls, monitoring, and operational testing would be required before using this system in a real financial environment.