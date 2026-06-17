"""Training entrypoint for Project 06: Credit Card Fraud Detection.

Trains models across five class-imbalance strategies, logging every run
to MLflow on DagsHub:

    1. baseline               - no imbalance correction
    2. class_weighted         - class_weight="balanced" / scale_pos_weight
    3. random_undersampling   - downsample majority class
    4. smote                  - synthetic minority oversampling
    5. unsupervised           - Isolation Forest, no fraud labels used

Usage:
    python -m src.train
    python -m src.train --data-path data/raw/creditcard.csv --experiment fraud-detection-imbalance-comparison

Requires DAGSHUB_TOKEN (and optionally DAGSHUB_USERNAME / DAGSHUB_REPO)
set in the environment or a local .env file -- see .env.example.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import dagshub
import dagshub.auth
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.evaluate import confusion_matrix_dict, evaluate_predictions
from src.imbalance_strategies import (
    apply_random_undersampling,
    apply_smote,
    compute_scale_pos_weight,
)
from src.preprocessing import scale_amount_time, split_features_target, stratified_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "fraud-detection-imbalance-comparison"
RANDOM_STATE = 42


def setup_dagshub_tracking(repo_owner: str, repo_name: str) -> None:
    """Authenticate with DagsHub via token (non-interactive, no browser
    prompt) and point MLflow's tracking URI at the DagsHub-hosted server."""
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "DAGSHUB_TOKEN not set. Copy .env.example to .env and fill in "
            "your token, or export it in your shell session."
        )

    dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    logger.info("MLflow tracking configured for %s/%s", repo_owner, repo_name)


def get_model_zoo(scale_pos_weight: float | None = None, class_weighted: bool = False) -> dict:
    """Return a fresh dict of model_name -> estimator for one strategy pass.

    Models are re-instantiated per strategy (rather than reused) so that
    fitted state never leaks between runs.
    """
    if class_weighted:
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "xgboost": xgb.XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
            ),
        }
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="aucpr"),
    }


def run_and_log(model_name, model, strategy_name, Xtr, ytr, Xte, yte, extra_params=None):
    """Fit one model, evaluate it, and log params/metrics to an MLflow run."""
    with mlflow.start_run(run_name=f"{strategy_name}_{model_name}"):
        mlflow.log_param("strategy", strategy_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("train_rows", len(Xtr))
        if extra_params:
            for k, v in extra_params.items():
                mlflow.log_param(k, v)

        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        metrics = evaluate_predictions(yte, y_pred, y_proba)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cm = confusion_matrix_dict(yte, y_pred)
        for k, v in cm.items():
            mlflow.log_metric(k, v)

        logger.info(
            "%s / %s -> pr_auc=%.4f recall@p90=%.4f f1=%.4f",
            strategy_name, model_name, metrics["pr_auc"], metrics["recall_at_p90"], metrics["f1"],
        )

        return {"strategy": strategy_name, "model": model_name, **metrics}


def train_all_strategies(data_path: Path, experiment_name: str) -> pd.DataFrame:
    df = load_raw_data(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train_scaled, X_test_scaled, _scaler = scale_amount_time(X_train, X_test)

    mlflow.set_experiment(experiment_name)
    results = []

    # 1. Baseline: no correction
    for name, model in get_model_zoo().items():
        Xtr = X_train if name == "random_forest" else X_train_scaled
        Xte = X_test if name == "random_forest" else X_test_scaled
        results.append(run_and_log(name, model, "baseline", Xtr, y_train, Xte, y_test))

    # 2. Class weighting
    spw = compute_scale_pos_weight(y_train)
    for name, model in get_model_zoo(scale_pos_weight=spw, class_weighted=True).items():
        Xtr = X_train if name == "random_forest" else X_train_scaled
        Xte = X_test if name == "random_forest" else X_test_scaled
        results.append(run_and_log(name, model, "class_weighted", Xtr, y_train, Xte, y_test))

    # 3. Random undersampling
    X_train_rus_scaled, y_train_rus = apply_random_undersampling(X_train_scaled, y_train)
    X_train_rus_raw, y_train_rus_raw = apply_random_undersampling(X_train, y_train)
    for name, model in get_model_zoo().items():
        if name == "random_forest":
            Xtr, ytr, Xte = X_train_rus_raw, y_train_rus_raw, X_test
        else:
            Xtr, ytr, Xte = X_train_rus_scaled, y_train_rus, X_test_scaled
        results.append(run_and_log(
            name, model, "random_undersampling", Xtr, ytr, Xte, y_test,
            extra_params={"resampled_train_size": len(Xtr)},
        ))

    # 4. SMOTE
    X_train_smote_scaled, y_train_smote = apply_smote(X_train_scaled, y_train)
    X_train_smote_raw, y_train_smote_raw = apply_smote(X_train, y_train)
    for name, model in get_model_zoo().items():
        if name == "random_forest":
            Xtr, ytr, Xte = X_train_smote_raw, y_train_smote_raw, X_test
        else:
            Xtr, ytr, Xte = X_train_smote_scaled, y_train_smote, X_test_scaled
        results.append(run_and_log(
            name, model, "smote", Xtr, ytr, Xte, y_test,
            extra_params={"resampled_train_size": len(Xtr)},
        ))

    # 5. Isolation Forest (unsupervised, no fraud labels used)
    contamination = y_train.mean()
    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_param("strategy", "unsupervised")
        mlflow.log_param("model", "isolation_forest")
        mlflow.log_param("contamination", contamination)

        iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
        iso.fit(X_train_scaled)

        raw_pred = iso.predict(X_test_scaled)
        y_pred_iso = np.where(raw_pred == -1, 1, 0)
        y_score_iso = -iso.decision_function(X_test_scaled)

        metrics = evaluate_predictions(y_test, y_pred_iso, y_score_iso)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        logger.info(
            "unsupervised / isolation_forest -> pr_auc=%.4f recall@p90=%.4f",
            metrics["pr_auc"], metrics["recall_at_p90"],
        )
        results.append({"strategy": "unsupervised", "model": "isolation_forest", **metrics})

    return pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Project 06 fraud detection models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--repo-owner", type=str, default=os.environ.get("DAGSHUB_USERNAME", "muhammed-keita-ml"))
    parser.add_argument("--repo-name", type=str, default=os.environ.get("DAGSHUB_REPO", "project-06-fraud-detection-pipeline"))
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    setup_dagshub_tracking(args.repo_owner, args.repo_name)
    results_df = train_all_strategies(args.data_path, args.experiment)

    print("\n=== Results (sorted by PR-AUC) ===")
    print(results_df.to_string(index=False))

    output_path = Path("reports") / "results_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Saved comparison table to %s", output_path)

    return results_df


if __name__ == "__main__":
    main()
