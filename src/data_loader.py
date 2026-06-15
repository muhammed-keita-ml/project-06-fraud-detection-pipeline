"""Data loading utilities for the Credit Card Fraud Detection pipeline."""

import pandas as pd


def load_raw_data(path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load the raw Kaggle Credit Card Fraud dataset."""
    return pd.read_csv(path)