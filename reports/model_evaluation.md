# FraudGuard AI — Model Evaluation

## 1. Objective

The objective of this evaluation is to assess how effectively a Random Forest classifier identifies fraudulent credit card transactions.

Because fraud is rare compared with legitimate activity, the model must be evaluated using metrics that measure its ability to identify fraud while limiting false alarms.

## 2. Dataset and Split

The dataset contains 284,807 transactions, including 492 fraudulent transactions.

The target column is `Class`:
- `0`: Legitimate
- `1`: Fraudulent

The data was divided using stratified sampling with `random_state=42`.

The intended split was:
- Training set: 60%
- Validation set: 20%
- Test set: 20%

Stratification preserves approximately the same proportion of fraudulent transactions in each split.

The final model evaluation was performed on the held-out test set.

## 3. Model Configuration

The model is a Random Forest classifier with the following settings:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

### Why Random Forest?

Random Forest combines predictions from multiple decision trees. It can learn nonlinear relationships between transaction features and the fraud label.

### Why use class weighting?

The dataset is highly imbalanced. The `balanced` class-weighting option adjusts the importance of classes based on their frequency in the training data.

This can help the model learn from the less common fraud class.

## 4. Evaluation Results

The recorded test metrics are:

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Fraud precision | 87.23% |
| Fraud recall | 83.67% |
| Fraud F1-score | 85.42% |
| Average precision | 0.8603 |
| ROC-AUC | 0.9623 |

The classification threshold used for the final prediction API is 0.40.

These results should be verified against the saved output of the final evaluation script before being treated as the definitive report.

## 5. Understanding the Metrics

### Accuracy — 99.95%

Accuracy measures the proportion of all transactions classified correctly.

Accuracy can be misleading on highly imbalanced datasets because a model may classify most transactions as legitimate and still achieve high accuracy.

For this reason, accuracy should not be interpreted alone.

### Precision — 87.23%

Precision measures how many transactions predicted as fraudulent were actually fraudulent.

A precision of approximately 87.23% means that, among the transactions classified as fraud in the evaluated test set, approximately 87.23% were truly fraudulent.

Higher precision generally means fewer false fraud alerts.

### Recall — 83.67%

Recall measures how many actual fraudulent transactions the model successfully identifies.

A recall of approximately 83.67% means the model detected approximately 83.67% of the fraudulent transactions in the evaluated test set.

Higher recall is important when missing a fraudulent transaction is costly.

### F1-score — 85.42%

The F1-score is the harmonic mean of precision and recall.

It provides a single measure that balances these two metrics.

A higher F1-score indicates a better balance between detecting fraud and limiting false alerts, although the best balance depends on the application.

### Average Precision — 0.8603

Average precision summarizes the precision-recall trade-off across different classification thresholds.

It is useful for fraud detection because the dataset contains very few fraudulent transactions compared with legitimate ones.

### ROC-AUC — 0.9623

ROC-AUC measures how well the model distinguishes fraudulent transactions from legitimate transactions across different thresholds.

A value of 0.9623 indicates strong ranking performance on the evaluated test data.

ROC-AUC is not the same as accuracy and does not mean the model is 96.23% accurate.

## 6. Confusion Matrix

The recorded confusion matrix is:

```text
[[56852,    12],
 [   16,    82]]
```

Rows represent actual classes, and columns represent predicted classes.

| | Predicted legitimate | Predicted fraudulent |
|---|---:|---:|
| Actual legitimate | 56,852 | 12 |
| Actual fraudulent | 16 | 82 |

Interpretation:

- True negatives: 56,852 legitimate transactions correctly identified.
- False positives: 12 legitimate transactions incorrectly flagged as fraud.
- False negatives: 16 fraudulent transactions incorrectly classified as legitimate.
- True positives: 82 fraudulent transactions correctly identified.

The test set contained 56,962 transactions, including 98 fraudulent transactions.

## 7. Decision Threshold

The prediction API uses a decision threshold of 0.40.

The rule is:

- Fraud probability greater than or equal to 0.40: classify as fraudulent.
- Fraud probability below 0.40: classify as legitimate.

A lower threshold may increase fraud recall but can also increase false positives.

A higher threshold may reduce false positives but can allow more fraudulent transactions to go undetected.

The threshold should be selected based on evaluation results and the relative costs of missed fraud and false alerts.

## 8. Limitations and Considerations

- The dataset is highly imbalanced.
- The evaluation results apply to the held-out test data and may not generalize to future transactions.
- Predicted probabilities are not guaranteed to be calibrated.
- A threshold of 0.40 is a project decision and should be validated for the intended use case.
- The model has not been validated for deployment in a real financial system.
- Further testing is needed to evaluate robustness, calibration, and performance under changing fraud patterns.

## 9. Conclusion

The recorded evaluation results suggest that the Random Forest model distinguishes fraudulent transactions from legitimate transactions effectively on the held-out test data.

Precision, recall, F1-score, average precision, and ROC-AUC provide more useful insight than accuracy alone for this imbalanced classification problem.

The saved model and decision threshold are now integrated into the FraudGuard AI FastAPI application, allowing transaction predictions through an API endpoint.