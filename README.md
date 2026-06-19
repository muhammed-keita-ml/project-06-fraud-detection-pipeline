# Project 06 - Credit Card Fraud Detection Pipeline

## Overview
Production-style ML pipeline for detecting fraudulent credit card
transactions, built around a single central challenge: extreme class
imbalance (~0.17% positive class). This project extends the MLOps
patterns established in Project 04 (Heart Disease Pipeline) and
Project 05 (Production ML Monitoring API) into a FinTech context, and
is the first entry in a three-project FinTech track (06 -> 07 -> 08).

## The Core Challenge: Class Imbalance
Fraud datasets are inherently imbalanced - in this dataset, fraudulent
transactions make up only 0.173% of all transactions (492 of 284,807).
A model that predicts "legitimate" for every transaction would score
99.8% accuracy while catching zero fraud, which makes accuracy useless
as an evaluation metric and makes ROC-AUC misleadingly optimistic.

This project treats imbalance handling as the primary technical
narrative and compares five strategies, each trained across up to
three model types, for thirteen total experiments:

- **Baseline** - no correction, to make the problem concrete
- **Class weighting** - `class_weight="balanced"` / `scale_pos_weight`
- **Random undersampling** - downsample the majority class
- **SMOTE** - synthetic minority oversampling
- **Isolation Forest** - unsupervised anomaly detection, no fraud labels used

Every run is tracked in MLflow on DagsHub with strategy and model type
logged as params, evaluated primarily on **PR-AUC** and **recall at
>=90% precision (recall@P90)** rather than accuracy or ROC-AUC.

## Results

| Rank | Strategy | Model | PR-AUC | Recall@P90 | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | class_weighted | xgboost | 0.8800 | 0.8367 | 0.8817 | 0.8367 | 0.8586 | 0.9682 |
| 2 | smote | xgboost | 0.8774 | 0.7959 | 0.7311 | 0.8878 | 0.8018 | 0.9792 |
| 3 | smote | random_forest | 0.8741-0.8747 | 0.8061 | 0.8265-0.8351 | 0.8265 | 0.8265-0.8308 | 0.9644 |
| 4 | baseline | random_forest | 0.8734 | 0.8571 | 0.9412 | 0.8163 | 0.8743 | 0.9630 |
| 5 | class_weighted | random_forest | 0.8542-0.8629 | 0.7857-0.8469 | 0.9059-0.9605 | 0.7449-0.7857 | 0.8391-0.8415 | 0.9529-0.9573 |
| 6 | baseline | xgboost | 0.7973 | 0.7857 | 0.8667 | 0.7959 | 0.8298 | 0.9390 |
| 7 | baseline | logistic_regression | 0.7432 | 0.2143 | 0.8289 | 0.6429 | 0.7241 | 0.9559 |
| 8 | smote | logistic_regression | 0.7249 | 0.0000 | 0.0580 | 0.9184 | 0.1092 | 0.9699 |
| 9 | class_weighted | logistic_regression | 0.7159 | 0.0000 | 0.0610 | 0.9184 | 0.1144 | 0.9722 |
| 10 | random_undersampling | random_forest | 0.6953 | 0.0000 | 0.0423 | 0.9184 | 0.0809 | 0.9777 |
| 11 | random_undersampling | logistic_regression | 0.6778 | 0.0000 | 0.0384 | 0.9184 | 0.0738 | 0.9759 |
| 12 | random_undersampling | xgboost | 0.3750 | 0.0000 | 0.0333 | 0.9184 | 0.0644 | 0.9749 |
| 13 | unsupervised | isolation_forest | 0.2180 | 0.0000 | 0.3084 | 0.3367 | 0.3220 | 0.9543 |

*Random Forest rows show a small range across two runs (Kaggle notebook
vs. local `src/train.py`) due to run-to-run variance under `n_jobs=-1`;
rankings and conclusions are unaffected.*

### Key Findings

**Winner: XGBoost + class weighting.** PR-AUC = 0.880, recall@P90 =
0.837 - it catches 83.7% of fraud while keeping precision above 90%.
It is also the cheapest strategy tested: no resampling, no synthetic
data, no shrinking the training set, just a single `scale_pos_weight`
parameter.

**The imbalance-correction "win" is smaller than expected - and that's
the more interesting finding.** The best uncorrected baseline (Random
Forest, no correction at all) scored PR-AUC = 0.873 with recall@P90 =
0.857, which is actually *higher* recall@P90 than the overall winner.
The gap between the best engineered strategy and the best uncorrected
baseline is under 0.01 PR-AUC points. Tree-based ensembles already
handle imbalance reasonably well by learning complex boundaries from
whatever minority-class signal exists; the corrective strategies here
mainly help weaker learners (Logistic Regression) catch up, rather
than pushing strong learners much further.

**Random undersampling collapses recall@P90 to zero for every model
tested, and so does SMOTE / class weighting for Logistic Regression.**
Six of the thirteen runs show `recall_at_p90 = 0.0`, despite reasonable
raw recall (~0.92) and even acceptable F1 scores. This means no
probability threshold on these models' output can hit >=90% precision
with any recall at all - undersampling discards ~99% of legitimate
transactions during training, leaving models so trigger-happy that any
threshold strict enough for 90% precision also rejects every true
fraud case. **F1 alone hides this completely**; several of the
zero-recall@P90 runs have F1 scores that look like a reasonable
mid-table strategy on that metric alone. This is the central argument
for using PR-AUC and recall-at-precision instead of F1 or accuracy in
this domain.

**Isolation Forest** (no fraud labels used at all) scored the weakest
PR-AUC (0.218), quantifying the cost of going fully unsupervised -
relevant context for production systems where confirmed fraud labels
lag behind transactions by days or weeks.

**Production implication:** the simplest fix (class weighting) on a
strong base learner (XGBoost) outperforms more elaborate resampling
pipelines here, while resampling techniques that look fine on F1 can
be silently unusable once a precision constraint is applied - a
constraint every real fraud-ops team has, since false positives carry
direct customer-friction and investigation cost.

## Dataset
Credit Card Fraud Detection (ULB) - 284,807 transactions, 492 fraud
cases (0.173%), PCA-anonymized features V1-V28 plus Time and Amount.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Reproducibility
This pipeline was run in two independent environments and produced
matching conclusions:
1. **Kaggle notebooks** - `01_eda_imbalance.ipynb` and
   `02_modeling_imbalance_strategies.ipynb`, with MLflow tracking
   authenticated via Kaggle Secrets
2. **Local execution** - `python -m src.train`, with MLflow tracking
   authenticated via a local `.env` file

Both log to the same DagsHub-hosted MLflow tracking server:
https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline

## Project Structure
```
project-06-fraud-detection-pipeline/
|-- data/
|   |-- raw/              # creditcard.csv (not committed, see .gitignore)
|   `-- processed/
|-- notebooks/
|   |-- 01_eda_imbalance.ipynb
|   `-- 02_modeling_imbalance_strategies.ipynb
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- imbalance_strategies.py
|   |-- train.py
|   `-- evaluate.py
|-- tests/
|   `-- test_pipeline.py
|-- models/
|-- reports/
|   |-- figures/
|   `-- results_comparison.csv
|-- requirements.txt
`-- README.md
```

## Running Locally
```powershell
# Activate the project's isolated venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Place the dataset
# Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and save as data/raw/creditcard.csv

# Configure DagsHub credentials
copy .env.example .env
# then edit .env and add your DAGSHUB_TOKEN

# Run tests
pytest tests/ -v

# Run the full training pipeline (13 model/strategy combinations)
python -m src.train
```

## Roadmap
- **Project 06** (this repo): Model training + MLflow tracking + imbalance strategy comparison
- **Project 07**: Cloud deployment + Kubernetes orchestration + monitoring at scale
- **Project 08 (FRIP)**: Capstone - fraud detection + credit scoring + monitoring as a unified platform

## Tech Stack
Python, scikit-learn, imbalanced-learn, XGBoost, MLflow, DagsHub, pandas, matplotlib, seaborn, pytest

## Status
Complete. Live demo: https://huggingface.co/spaces/muhammed-keita-ml/credit-card-fraud-detector

## Links
- **GitHub**: https://github.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
- **Hugging Face Space**: https://huggingface.co/spaces/muhammed-keita-ml/credit-card-fraud-detector
- **DagsHub / MLflow**: https://dagshub.com/muhammed-keita-ml/project-06-fraud-detection-pipeline
- **Medium**: https://medium.com/@mkeitaone/when-good-enough-metrics-hide-a-broken-model-building-a-credit-card-fraud-detection-pipeline-2114f3af0b55
