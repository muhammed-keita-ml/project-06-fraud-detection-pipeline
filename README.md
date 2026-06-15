# Project 06 - Credit Card Fraud Detection Pipeline

## Overview
Production-style ML pipeline for detecting fraudulent credit card
transactions, with a central focus on extreme class imbalance
(~0.17% positive class). This project extends the MLOps patterns
established in Project 04 (Heart Disease Pipeline) and Project 05
(Production ML Monitoring API) into a FinTech context.

## The Core Challenge: Class Imbalance
Fraud datasets are inherently imbalanced - fraudulent transactions
typically represent less than 1% of all transactions. This project
treats imbalance handling as the primary technical narrative:

- Baseline models trained without imbalance correction (to demonstrate the problem)
- Class weighting
- SMOTE / ADASYN oversampling
- Random undersampling
- Anomaly-detection baseline (Isolation Forest)

Each strategy is tracked as a separate MLflow run and evaluated
primarily on PR-AUC and recall-at-fixed-precision rather than
accuracy or ROC-AUC, which are misleading under extreme imbalance.

## Dataset
Credit Card Fraud Detection (ULB) - 284,807 transactions, 492 fraud
cases (0.172%), PCA-anonymized features V1-V28 plus Time and Amount.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Project Structure
```
project-06-fraud-detection-pipeline/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- imbalance_strategies.py
|   |-- train.py
|   `-- evaluate.py
|-- tests/
|-- models/
|-- reports/figures/
|-- requirements.txt
`-- README.md
```

## Roadmap
- **Project 06** (this repo): Model training + MLflow tracking + imbalance strategy comparison
- **Project 07**: Cloud deployment + Kubernetes orchestration + monitoring at scale
- **Project 08 (FRIP)**: Capstone - fraud detection + credit scoring + monitoring as a unified platform

## Tech Stack
Python, scikit-learn, imbalanced-learn, XGBoost/LightGBM, MLflow, DagsHub, pandas, matplotlib, seaborn

## Status
In progress - EDA phase