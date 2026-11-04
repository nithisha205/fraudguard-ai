from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def print_threshold_table(y_true, probabilities, thresholds=THRESHOLDS):
    """Print precision, recall, F1, false positives and missed fraud for each threshold."""
    print("Threshold | Precision | Recall | F1-score | False positives | Missed fraud")

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)

        print(f"{threshold:9.2f} | {precision:9.4f} | {recall:6.4f} | {f1:8.4f} | {fp:15,d} | {fn:12,d}")
