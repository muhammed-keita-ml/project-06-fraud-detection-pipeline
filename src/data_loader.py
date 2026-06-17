"""Data loading utilities for the Credit Card Fraud Detection pipeline."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data/raw/creditcard.csv")


def load_raw_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle Credit Card Fraud dataset.

    Args:
        path: Path to creditcard.csv. Defaults to data/raw/creditcard.csv,
            relative to the project root.

    Returns:
        DataFrame with columns Time, V1-V28, Amount, Class.

    Raises:
        FileNotFoundError: if the file isn't present, with guidance on how
            to obtain it from Kaggle.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place creditcard.csv at: {path.resolve()}"
        )

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info(
        "Loaded %s rows, %s fraud cases (%.4f%%)",
        f"{len(df):,}",
        df["Class"].sum(),
        df["Class"].mean() * 100,
    )
    return df
