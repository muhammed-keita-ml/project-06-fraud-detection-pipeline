"""Train and save the final production model for Project 06.

This script trains the winning configuration identified in the
imbalance-strategy comparison (see notebooks/02_modeling_imbalance_strategies.ipynb
and reports/results_comparison.csv): XGBoost with class weighting via
scale_pos_weight, PR-AUC = 0.880, recall@P90 = 0.837.

Unlike src/train.py (which trains all 13 strategy/model combinations for
comparison), this script trains only the single winning configuration,
on the full training split, and saves everything needed for inference:

    models/fraud_xgboost_class_weighted.json   - the XGBoost model
    models/scaler.joblib                       - fitted StandardScaler for Time/Amount
    models/feature_columns.json                - exact column order expected at inference
    models/model_metadata.json                 - metrics, params, training info

The same run is also logged to MLflow (DagsHub) as an artifact, consistent
with the rest of the pipeline's tracking.

Usage:
    python -m src.train_final_model
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import dagshub
import dagshub.auth
import joblib
import mlflow
import xgboost as xgb
from dotenv import load_dotenv

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.evaluate import confusion_matrix_dict, evaluate_predictions
from src.imbalance_strategies import compute_scale_pos_weight
from src.preprocessing import scale_amount_time, split_features_target, stratified_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
RANDOM_STATE = 42
EXPERIMENT_NAME = "fraud-detection-final-model"


def setup_dagshub_tracking(repo_owner: str, repo_name: str) -> None:
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN not set. Copy .env.example to .env and fill it in.")
    dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def compute_preset_patterns(X_train, y_train) -> dict:
    """Compute representative V-feature patterns from real class-conditional
    means in the training data, for use as 'Simple mode' presets in the
    Gradio demo. V1-V28 are PCA-anonymized and have no inherent real-world
    meaning, so these presets give the demo realistic feature combinations
    to react to, rather than requiring the user to understand 28 anonymized
    components directly."""
    v_cols = [c for c in X_train.columns if c.startswith("V")]
    legit_means = X_train.loc[y_train == 0, v_cols].mean().round(3)
    fraud_means = X_train.loc[y_train == 1, v_cols].mean().round(3)
    return {
        "Typical legitimate pattern": legit_means.to_dict(),
        "Fraud-like pattern": fraud_means.to_dict(),
    }


def train_final_model(data_path: Path = DEFAULT_DATA_PATH) -> dict:
    df = load_raw_data(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_amount_time(X_train, X_test)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info("Training final model: XGBoost + class_weighted (scale_pos_weight=%.2f)", scale_pos_weight)

    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    cm = confusion_matrix_dict(y_test, y_pred)

    logger.info("Final model metrics: %s", metrics)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "fraud_xgboost_class_weighted.json"
    model.save_model(model_path)

    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    feature_columns = list(X_train.columns)
    feature_columns_path = MODELS_DIR / "feature_columns.json"
    feature_columns_path.write_text(json.dumps(feature_columns, indent=2))

    preset_patterns = compute_preset_patterns(X_train, y_train)
    preset_patterns_path = MODELS_DIR / "preset_patterns.json"
    preset_patterns_path.write_text(json.dumps(preset_patterns, indent=2))

    metadata = {
        "model_name": "fraud_xgboost_class_weighted",
        "strategy": "class_weighted",
        "algorithm": "xgboost",
        "scale_pos_weight": scale_pos_weight,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "confusion_matrix": cm,
        "scaled_columns": ["Time", "Amount"],
        "feature_columns_file": "feature_columns.json",
        "preset_patterns_file": "preset_patterns.json",
        "scaler_file": "scaler.joblib",
        "model_file": "fraud_xgboost_class_weighted.json",
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model artifacts to %s/", MODELS_DIR)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="final_model_class_weighted_xgboost"):
        mlflow.log_param("strategy", "class_weighted")
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for k, v in cm.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(feature_columns_path))
        mlflow.log_artifact(str(preset_patterns_path))
        mlflow.log_artifact(str(metadata_path))
        logger.info("Logged final model artifacts to MLflow/DagsHub")

    return metadata


def main():
    load_dotenv()
    repo_owner = os.environ.get("DAGSHUB_USERNAME", "muhammed-keita-ml")
    repo_name = os.environ.get("DAGSHUB_REPO", "project-06-fraud-detection-pipeline")

    setup_dagshub_tracking(repo_owner, repo_name)
    metadata = train_final_model()

    print("\n=== Final Model Metrics ===")
    for k, v in metadata["metrics"].items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"\nArtifacts saved to: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
