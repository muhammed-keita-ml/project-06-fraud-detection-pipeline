"""Preprocessing utilities: scaling, train/test split, feature prep."""

from sklearn.model_selection import train_test_split


def split_features_target(df, target_col: str = "Class"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def stratified_split(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )