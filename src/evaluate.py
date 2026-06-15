"""Evaluation utilities focused on imbalanced classification metrics."""

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
)


def evaluate_predictions(y_true, y_pred, y_proba):
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }