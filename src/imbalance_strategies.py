"""Class imbalance handling strategies for fraud detection.

This module is the heart of Project 06: each function returns a
resampled (X, y) pair or a model wrapper configured for a specific
imbalance-handling approach, so each can be logged as its own MLflow run.
"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_smote(X_train, y_train, random_state: int = 42):
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)


def apply_random_undersampling(X_train, y_train, random_state: int = 42):
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X_train, y_train)


def class_weight_dict(y_train):
    """Compute class weights for use with class_weight='balanced'-style params."""
    counts = y_train.value_counts()
    total = counts.sum()
    return {cls: total / (len(counts) * count) for cls, count in counts.items()}