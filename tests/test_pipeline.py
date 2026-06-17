"""Tests for Project 06 src modules.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import confusion_matrix_dict, evaluate_predictions, recall_at_precision
from src.imbalance_strategies import apply_random_undersampling, apply_smote, compute_scale_pos_weight
from src.preprocessing import scale_amount_time, split_features_target, stratified_split


@pytest.fixture
def imbalanced_df():
    """A small synthetic imbalanced dataset mimicking the real schema."""
    rng = np.random.RandomState(42)
    n_legit, n_fraud = 980, 20

    legit = pd.DataFrame(rng.normal(0, 1, size=(n_legit, 28)), columns=[f"V{i}" for i in range(1, 29)])
    legit["Time"] = rng.uniform(0, 172800, n_legit)
    legit["Amount"] = rng.exponential(50, n_legit)
    legit["Class"] = 0

    fraud = pd.DataFrame(rng.normal(2, 1, size=(n_fraud, 28)), columns=[f"V{i}" for i in range(1, 29)])
    fraud["Time"] = rng.uniform(0, 172800, n_fraud)
    fraud["Amount"] = rng.exponential(20, n_fraud)
    fraud["Class"] = 1

    return pd.concat([legit, fraud], ignore_index=True)


class TestPreprocessing:
    def test_split_features_target(self, imbalanced_df):
        X, y = split_features_target(imbalanced_df)
        assert "Class" not in X.columns
        assert y.name == "Class"
        assert len(X) == len(y) == len(imbalanced_df)

    def test_stratified_split_preserves_ratio(self, imbalanced_df):
        X, y = split_features_target(imbalanced_df)
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02

    def test_scale_amount_time_only_scales_target_columns(self, imbalanced_df):
        X, y = split_features_target(imbalanced_df)
        X_train, X_test, _, _ = stratified_split(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_amount_time(X_train, X_test)

        assert abs(X_train_scaled["Amount"].mean()) < 1e-6
        assert abs(X_train_scaled["Time"].mean()) < 1e-6
        pd.testing.assert_series_equal(X_train_scaled["V1"], X_train["V1"])


class TestImbalanceStrategies:
    def test_smote_balances_classes(self, imbalanced_df):
        X, y = split_features_target(imbalanced_df)
        X_res, y_res = apply_smote(X, y)
        counts = y_res.value_counts()
        assert counts[0] == counts[1]
        assert len(X_res) == len(y_res)

    def test_random_undersampling_balances_classes(self, imbalanced_df):
        X, y = split_features_target(imbalanced_df)
        X_res, y_res = apply_random_undersampling(X, y)
        counts = y_res.value_counts()
        assert counts[0] == counts[1]
        assert counts[1] == (y == 1).sum()

    def test_compute_scale_pos_weight(self, imbalanced_df):
        _, y = split_features_target(imbalanced_df)
        weight = compute_scale_pos_weight(y)
        expected = (y == 0).sum() / (y == 1).sum()
        assert weight == pytest.approx(expected)


class TestEvaluate:
    def test_recall_at_precision_perfect_separation(self):
        y_true = [0, 0, 0, 1, 1]
        y_proba = [0.1, 0.2, 0.3, 0.9, 0.95]
        assert recall_at_precision(y_true, y_proba, min_precision=0.90) == pytest.approx(1.0)

    def test_recall_at_precision_returns_zero_when_unreachable(self):
        # Heavy class overlap: no threshold reaches 90% precision.
        y_true = [0] * 9 + [1]
        y_proba = [0.5] * 10
        result = recall_at_precision(y_true, y_proba, min_precision=0.90)
        assert result == 0.0

    def test_evaluate_predictions_returns_expected_keys(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]
        y_proba = [0.1, 0.2, 0.8, 0.4]
        metrics = evaluate_predictions(y_true, y_pred, y_proba)
        expected_keys = {"pr_auc", "recall_at_p90", "precision", "recall", "f1", "roc_auc"}
        assert set(metrics.keys()) == expected_keys

    def test_confusion_matrix_dict(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        cm = confusion_matrix_dict(y_true, y_pred)
        assert cm == {
            "true_negatives": 1,
            "false_positives": 1,
            "false_negatives": 1,
            "true_positives": 1,
        }
