"""Evaluation utilities focused on imbalanced classification metrics.

PR-AUC and recall-at-fixed-precision are the primary metrics for this
project. Under ~0.17% class imbalance, accuracy and even ROC-AUC can
look deceptively strong while the model is operationally unusable;
see the Project 06 modeling notebook (Section 10) for the empirical
case, especially the recall@P90 = 0 results for several strategies
that looked reasonable on F1 alone.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_precision(y_true, y_proba, min_precision: float = 0.90) -> float:
    """Highest recall achievable while keeping precision >= min_precision.

    Returns 0.0 if no threshold achieves the minimum precision -- this is
    not a bug, it's the correct (and important) signal that a strategy is
    operationally unusable at that precision bar.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    valid = precisions >= min_precision
    return float(recalls[valid].max()) if valid.any() else 0.0


def evaluate_predictions(y_true, y_pred, y_proba, min_precision: float = 0.90) -> dict:
    """Compute the full metric set for one (model, strategy) run."""
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "recall_at_p90": recall_at_precision(y_true, y_proba, min_precision),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def confusion_matrix_dict(y_true, y_pred) -> dict:
    """Confusion matrix as a flat dict, convenient for MLflow logging."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"true_negatives": int(tn), "false_positives": int(fp),
            "false_negatives": int(fn), "true_positives": int(tp)}
