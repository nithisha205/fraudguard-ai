from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# Locate the project folder and dataset.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Load the dataset and separate features (X) from the target (y)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    return df, df.drop(columns=["Class"]), df["Class"]


def split_data(X, y):
    """Create the reproducible, stratified 60% train / 20% validation / 20% test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_random_forest():
    """The original Random Forest configuration (baseline)."""
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_xgboost():
    """The final model. No class weighting, so its probabilities stay realistic."""
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cross_val_folds():
    """5 stratified folds used for model comparison and threshold selection."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def out_of_fold_probabilities(model, X, y):
    """Fraud probability for every row, each predicted by a model that never saw that row."""
    probabilities = np.zeros(len(y))
    for train_idx, predict_idx in cross_val_folds().split(X, y):
        fold_model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[predict_idx] = fold_model.predict_proba(X.iloc[predict_idx])[:, 1]
    return probabilities


# Business costs: a missed fraud costs the transaction amount; a false alarm costs a manual review.
REVIEW_COST = 5.0


def total_cost(y_true, flagged, amounts, review_cost=REVIEW_COST):
    """Missed fraud amount plus the cost of reviewing false alarms."""
    y_true, flagged, amounts = np.asarray(y_true), np.asarray(flagged), np.asarray(amounts)
    missed_fraud = (y_true == 1) & ~flagged
    false_alarms = (y_true == 0) & flagged
    return float(amounts[missed_fraud].sum() + review_cost * false_alarms.sum())
