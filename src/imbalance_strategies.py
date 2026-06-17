"""Class imbalance handling strategies for fraud detection.

This module is the heart of Project 06: each function returns a
resampled (X, y) pair, or a fitting parameter (class weights /
scale_pos_weight), so each strategy can be trained and logged as its
own MLflow run and compared on equal footing.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Synthetic Minority Oversampling: generates synthetic fraud examples
    by interpolating between existing minority-class neighbors."""
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)


def apply_random_undersampling(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Downsample the majority class to match the minority class count."""
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X_train, y_train)


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio used by XGBoost's scale_pos_weight param for class weighting."""
    return (y_train == 0).sum() / (y_train == 1).sum()


STRATEGIES = (
    "baseline",
    "class_weighted",
    "random_undersampling",
    "smote",
    "unsupervised",
)
