# FraudGuard AI — Exploratory Data Analysis

## 1. Objective

Explore the credit card transaction dataset to understand its structure, data quality, class distribution, and transaction amount patterns before developing a fraud detection model.

## 2. Dataset Overview

- **Dataset:** Credit Card Fraud Detection
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total transactions:** 284,807
- **Total columns:** 31
- **Input features:** 30
- **Target column:** `Class`

The target column uses `0` for legitimate transactions and `1` for fraudulent transactions.

The dataset contains:
- `Time`: Elapsed time in seconds within the dataset.
- `V1`–`V28`: Anonymized, PCA-transformed features.
- `Amount`: Transaction amount.
- `Class`: Transaction label.

## 3. Data Quality Assessment

The initial inspection identified the following:

| Check | Finding |
|---|---|
| Total records | 284,807 |
| Total columns | 31 |
| Missing values | 0 |
| Input features | 30 |
| Target variable | `Class` |
| Feature data types | Primarily numerical |
| Target data type | Integer |

No missing values were detected during the initial inspection.

Although the dataset is numerically structured and contains no detected missing values, additional checks and appropriate preprocessing are still important before model training.

## 4. Class Distribution

The target variable is highly imbalanced.

| Transaction class | Count | Percentage |
|---|---:|---:|
| Legitimate (0) | 284,315 | 99.8273% |
| Fraudulent (1) | 492 | 0.1727% |
| **Total** | **284,807** | **100%** |

Approximately 578 legitimate transactions occur for every fraudulent transaction.

![Transaction class distribution](class_distribution.png)

### Key Findings

- Fraudulent transactions represent approximately 0.17% of the dataset.
- Legitimate transactions dominate the observations.
- A classifier that predicts every transaction as legitimate would achieve approximately 99.83% accuracy while detecting zero fraudulent transactions.

### Implications for Modeling

Accuracy alone is insufficient for evaluating this problem.

The project therefore considers fraud-class precision, recall, F1-score, average precision, and false-positive and false-negative counts.

Stratified data splitting is also used to help preserve the class distribution across training, validation, and test sets.

## 5. Transaction Amount Analysis

The transaction amount distributions differ between legitimate and fraudulent transactions.

| Statistic | Legitimate | Fraudulent |
|---|---:|---:|
| Count | 284,315 | 492 |
| Mean | 88.29 | 122.21 |
| Median | 22.00 | 9.25 |
| 25th percentile | 5.65 | 1.00 |
| 75th percentile | 77.05 | 105.89 |
| Minimum | 0.00 | 0.00 |
| Maximum | 25,691.16 | 2,125.87 |

![Transaction amount box plot](amount_boxplot.png)

### Key Findings

- The median fraudulent transaction amount is lower than the median legitimate transaction amount.
- The mean fraudulent transaction amount is higher than the mean legitimate transaction amount.
- The difference between the mean and median indicates that the distributions are influenced by larger transaction amounts.
- The middle 50% of transaction amounts spans 5.65–77.05 for legitimate transactions and 1.00–105.89 for fraudulent transactions.
- The distributions overlap, so transaction amount alone cannot reliably distinguish fraud from legitimate activity.

These observations are descriptive. They do not establish causation or prove how useful transaction amount will be for prediction.

## 6. Feature Considerations

### Anonymized Features

The `V1`–`V28` features are PCA-transformed and anonymized. Their original business meanings are not directly available in the dataset.

### Time and Amount

`Time` and `Amount` provide additional transaction-level information. Their predictive value must be assessed through appropriate model evaluation rather than assumed from descriptive statistics alone.

### Target Leakage Prevention

The `Class` column is the target variable and must not be included among the model's input features.

The input matrix should contain the 30 features, while the target vector should contain the corresponding `Class` labels.

## 7. Overall Conclusions

The initial exploratory analysis identified three important characteristics:

1. **Severe class imbalance:** Fraud accounts for approximately 0.17% of transactions.
2. **Differences in transaction amounts:** Legitimate and fraudulent transactions have different descriptive statistics, but their amount distributions overlap.
3. **Limited feature interpretability:** The anonymized PCA features cannot be directly interpreted as their original transaction characteristics.

These findings informed the project's approach to model development and evaluation.

## 8. Next Steps

Based on the EDA findings, the next stages of the project are:

1. Separate the input features from the target variable.
2. Create stratified training, validation, and test sets.
3. Train baseline and tree-based classification models.
4. Compare models using metrics appropriate for imbalanced classification.
5. Investigate classification thresholds using validation data.
6. Evaluate the selected model on a held-out test set.
7. Integrate the selected model into a prediction API.