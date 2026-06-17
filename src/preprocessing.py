"""Preprocessing utilities: train/test split and feature scaling.

V1-V28 arrive pre-scaled from the original PCA transformation. Only Time
and Amount are on raw scales and benefit from standardization, which
matters for Logistic Regression and Isolation Forest in particular.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Class"
SCALE_COLS = ["Time", "Amount"]


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified split is essential here: a plain random split risks an
    unrepresentative (or even zero) fraud count in the test set given the
    ~0.17% positive rate.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def scale_amount_time(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit a StandardScaler on train, apply to both train and test.

    Returns (X_train_scaled, X_test_scaled, fitted_scaler). The scaler is
    returned so it can be persisted alongside the model for inference-time
    consistency.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test_scaled[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

    return X_train_scaled, X_test_scaled, scaler
