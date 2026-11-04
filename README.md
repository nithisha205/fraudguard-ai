# FraudGuard AI

![Tests](https://github.com/nithisha205/fraudguard-ai/actions/workflows/tests.yml/badge.svg)

An end-to-end machine learning project that detects potentially fraudulent credit card transactions and **explains every prediction**.

- XGBoost classifier trained on 284,807 real (anonymised) card transactions, of which only 0.17% are fraud.
- Decision threshold chosen by **business cost** (missed fraud amount vs. manual review cost) with cross-validation.
- FastAPI service that returns the fraud probability *and* the features that pushed it up or down (TreeSHAP).
- Streamlit dashboard to inspect real test transactions, try "what if" changes and see the cost trade-off live.
- Honest evaluation: a time-based split, error analysis of missed fraud, and the ideas that did not work.
- Drift monitoring and a retraining simulation showing when and how to update the model.
- Runs with one command in Docker; tests and a Docker build run on every push with GitHub Actions.

## Results

Held-out test set: 56,962 transactions, 98 fraud worth $10,644.93 in total. The test set was used once, after every choice was made.

| Model | Threshold | Avg. precision | ROC-AUC | Fraud caught | False alarms | Total cost* |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (baseline) | 0.44 | 0.8603 | 0.9623 | 82 / 98 | 11 | $2,077.77 |
| **XGBoost (final)** | **0.02** | **0.8722** | **0.9757** | **85 / 98** | 30 | **$1,980.72** |

\* Total cost = amount of fraud missed + $5 review cost for every false alarm.

The final model catches 3 more frauds at the price of 19 more manual reviews, lowering total cost by about 5%.
Fraud precision is 73.9% and recall 86.7%. Accuracy is not used: predicting "legitimate" for everything already scores 99.83%.

![Cross-validated cost by threshold](reports/figures/threshold_cost.png)

The threshold is low because XGBoost (trained without class weighting) gives realistic probabilities: the base fraud rate is about 0.15%,
so a 2% chance of fraud is already 13 times the average. The cost curve is nearly flat from 0.02 to 0.24, so the exact choice in that range matters little.

### Model comparison

5-fold cross-validation on the training + validation data (227,845 transactions, 394 fraud), produced by `scripts/compare_models.py`:

| Model | Avg. precision | Brier score | Best threshold | Cost | Missed fraud | False alarms | Cost with expected-loss rule |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7478 | 0.02303 | 0.99 | $13,571 | 70 | 213 | $193,939 |
| Random Forest | 0.8352 | 0.00045 | 0.20 | $12,789 | 64 | 79 | $12,929 |
| LightGBM | 0.8402 | 0.00046 | 0.03 | $13,224 | 76 | 31 | $13,390 |
| **XGBoost** | **0.8525** | **0.00040** | 0.02 | $12,723 | 62 | 100 | $12,862 |
| XGBoost + isotonic calibration | 0.8488 | 0.00040 | 0.04 | $12,658 | 60 | 136 | $12,527 |

Total fraud in these transactions: $49,483. Best thresholds are picked on the same predictions they are scored on, so these costs are slightly optimistic.
XGBoost has the best average precision and the most accurate probabilities (lowest Brier score), so it was chosen.
Logistic Regression's huge expected-loss cost shows why that rule needs realistic probabilities: `class_weight="balanced"` inflates them, so it flags almost everything.

### Does it work on future transactions?

The random split mixes transactions from the whole 48 hours, so the model is tested on the same period it learned from.
`experiments/time_split_evaluation.py` instead trains on the first 60% of transactions in time, picks the threshold on the next 20%, and tests on the last 20% (hours 40–48):

| Split | Avg. precision | ROC-AUC | Fraud caught | False alarms | Cost / total fraud |
|---|---:|---:|---:|---:|---:|
| Random (stratified) | 0.8710 | 0.9752 | 82 / 98 (84%) | 7 | $1,968 / $10,645 (18%) |
| Time-based (past → future) | 0.7993 | 0.9793 | 57 / 75 (76%) | 11 | $2,693 / $7,729 (35%) |

Average precision drops from 0.87 to 0.80 and the share of fraud money lost doubles, so **the random-split numbers are optimistic**.
Fraud patterns shift even within two days; a real deployment would need regular retraining and monitoring.
(Both rows use a single 60/20/20 split without cross-validation, so they differ slightly from the final model's results above.)

### Monitoring drift and retraining

If the model gets worse over time, when should it be retrained? `scripts/monitoring.py` replays the 48 hours in 8-hour blocks.
A model trained on the first 16 hours and never updated (**static**) is compared with retraining before every block on all data so far (**expanding**)
or on the latest 16 hours only (**sliding**). All use the saved threshold of 0.02.

![Monitoring simulation](reports/figures/monitoring.png)

| Hours | Fraud | Drift status | Score PSI | Static avg precision | Expanding avg precision |
|---|---:|---|---:|---:|---:|
| 16–24 | 90 | no earlier day | – | 0.794 | 0.794 |
| 24–32 | 75 | retrain | 1.07 | 0.826 | 0.853 |
| 32–40 | 59 | retrain | 0.42 | 0.723 | 0.776 |
| 40–48 | 77 | warning | 0.16 | 0.777 | 0.804 |

| Strategy | Total cost | Fraud caught |
|---|---:|---:|
| Static (never retrained) | $12,846 | 82.7% |
| **Expanding (all data so far)** | **$11,834** | 82.4% |
| Sliding (latest 16 hours) | $12,013 | 81.4% |

- **Retraining pays off:** expanding retraining wins in every block after the first day and cuts total cost by about 8% ($1,012).
  Keeping all past data beats a short sliding window, because fraud is rare and every example counts.
- **Drift can be seen before labels arrive.** The Population Stability Index (PSI) compares each block with the same hours one day earlier,
  without needing to know which transactions were fraud. The fraud-score PSI flagged exactly the blocks where the static model fell behind.
- **Compare like with like.** Comparing night-time transactions with a mix of day and night raised false alarms on almost every feature
  (V12: PSI 0.81 against the first 16 hours, but 0.003 against the previous night). With a same-time-of-day reference, only a real shift on day 2 remains,
  led by `V1`, `V3` and `V28`.

PSI rule of thumb: below 0.10 OK, 0.10–0.25 warning, above 0.25 retrain. With only two days of data this is a simulation, not proof;
a real system would watch weeks of data and retrain on a schedule as well as on alerts.

### What the model still misses

The 13 missed test frauds ($1,830.72) have a median predicted fraud probability below 0.01%: to the model they look completely normal.
In cross-validation, 53 frauds (about $12,000) were missed by *every* model, even at a 5% threshold.
Since features `V1`–`V28` are anonymised PCA components, catching these would need new information such as cardholder history,
merchant or transaction velocity, not a different model or threshold.

Three ideas that did **not** help, kept here because negative results are results too:

- **Hour-of-day features** (`experiments/feature_experiment.py`): fraud is up to 10× more common in some hours, but adding the hour (also as a 24-hour circle) left cross-validated average precision unchanged (0.8529 vs 0.8525) and did not lower cost. The model already gets this from `Time` and the V features.
- **Flagging by expected loss** (`probability × amount > $5`) instead of one threshold lets many small frauds through and was not cheaper, even with isotonic calibration (about 2% better, within noise).
- **Up-weighting fraud** (`scale_pos_weight=10`) did not improve average precision on the validation set for XGBoost, and hurt LightGBM badly (0.83 → 0.56).

## How it works

1. **Split** — stratified 60% train / 20% validation / 20% test (`src/fraudguard/data.py`). The test set is used once, at the end.
2. **Compare models** — Logistic Regression, Random Forest, LightGBM, XGBoost and calibrated XGBoost with 5-fold cross-validation (`scripts/compare_models.py`).
3. **Choose the threshold by cost** — `scripts/evaluate_final_model.py` gets out-of-fold XGBoost probabilities for the training + validation data,
   keeps the threshold with the lowest `missed fraud amount + $5 × false alarms`, trains the final model on all of that data,
   evaluates it once on the test set against the Random Forest baseline, and lists the largest missed frauds with their explanations.
4. **Explain** — `src/fraudguard/explain.py` uses XGBoost's built-in TreeSHAP. Contributions are in log-odds: positive pushes towards fraud, negative towards legitimate.
   The base value plus all contributions reproduces the model's prediction exactly (checked in the tests).

## Quick start

The dataset is not needed to run the app: the trained model and a sample of 298 test transactions are included.

### With Docker

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### With Python

```bash
pip install -r requirements.txt
pip install -e .
uvicorn fraudguard.api:app --reload
streamlit run src/fraudguard/dashboard.py
```

## Dashboard

`src/fraudguard/dashboard.py` has four tabs:

- **Check a transaction** — pick any of the 298 held-out test transactions (all 98 test frauds plus 200 legitimate ones),
  see the fraud probability, the decision, the true label and a chart of the features behind the decision.
  "What if?" lets you change the most influential features and watch the probability move.
- **Sample overview** — fraud caught, money missed, false alarms and total cost for the sample; move the threshold and review cost in the sidebar to see the trade-off.
- **Monitoring** — drift status per 8-hour block and the cost of never retraining vs retraining.
- **Model** — test metrics, how the threshold was chosen, and the cost curve.

## API

### Example

A real fraudulent transaction from the test set:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @transaction.json
```

```json
{
  "fraud_probability": 0.900464,
  "threshold": 0.02,
  "prediction": "Fraudulent",
  "base_probability": 0.001533,
  "top_factors": [
    {"feature": "V10", "value": -9.2228, "contribution": 5.1232},
    {"feature": "V14", "value": -9.8251, "contribution": 2.6112},
    {"feature": "V4",  "value": 6.0173,  "contribution": 1.5446}
  ]
}
```

The response is cut to three factors here; the API returns five.
The request body must contain exactly the 30 model features (`Time`, `V1`–`V28`, `Amount`) as numbers; anything else returns `422`.

| Endpoint | Purpose |
|---|---|
| `GET /` | API status |
| `GET /health` | Model type, threshold, how it was chosen, and test metrics |
| `POST /predict` | Fraud probability, prediction and top 5 contributing features |

## Retraining

1. Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group) into `data/`.
2. Install the project once with `pip install -r requirements.txt` and `pip install -e .`, then run the scripts from the project root:

```bash
python scripts/inspect_data.py               # dataset overview
python scripts/explore_data.py               # charts in reports/figures/
python scripts/compare_models.py             # cross-validated model comparison (about 5 minutes)
python experiments/feature_experiment.py     # hour-of-day feature experiment
python experiments/time_split_evaluation.py  # random vs time-based split
python scripts/monitoring.py                 # drift report and retraining simulation
python scripts/evaluate_final_model.py       # cost-based threshold, final test, saves the model
python scripts/make_samples.py               # dashboard sample transactions
```

## Tests and linting

```bash
pytest
ruff check .
ruff format --check .
```

## Project structure

```text
src/fraudguard/              the installable package used by the API and dashboard
  data.py                    loading, split, models, cross-validation and cost
  explain.py                 per-prediction feature contributions (TreeSHAP)
  predictor.py               loads the model; scores and explains one transaction
  drift.py                   Population Stability Index (PSI) drift checks
  api.py                     FastAPI service
  dashboard.py               Streamlit dashboard
scripts/                     the main pipeline: explore, compare, train and evaluate, monitor, make samples
experiments/                 one-off studies (time split, hour features) and the earlier baseline scripts
tests/                       API, explanation, cost, drift and dashboard tests
samples/                     298 held-out test transactions for the dashboard
reports/                     EDA notes, baseline evaluation, monitoring results, figures
models/fraud_model.joblib    trained model, threshold, feature list and test metrics
Dockerfile, docker-compose.yml, requirements-app.txt   container for the API and dashboard
pyproject.toml               package, pytest and ruff settings
```

## Limitations

- Features `V1`–`V28` are PCA-anonymised, so explanations name components rather than real-world attributes.
- Results drop on later transactions (see the time-based split); the reported test results come from a random split.
- The $5 review cost is an assumption; a real team would plug in its own costs.
- Not validated for use in a real payment system.

## Data

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group of Université Libre de Bruxelles (ULB),
released under the Open Database License. `samples/sample_transactions.csv` is a 298-row extract of it.
