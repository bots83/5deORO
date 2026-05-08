"""Dataset utilities con split temporal y walk-forward cross-validation."""
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import get_feature_cols, get_target_cols


def temporal_split(
    features_df: pd.DataFrame,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_cols = get_feature_cols(features_df)
    target_cols = get_target_cols(features_df)

    X = features_df[feat_cols].values.astype(np.float32)
    y = features_df[target_cols].values.astype(np.int32)

    split_idx = int(len(X) * (1 - test_fraction))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def walk_forward_splits(
    features_df: pd.DataFrame,
    initial_train: int = 500,
    step: int = 50,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Walk-forward cross-validation: en cada iteración el train crece e incorpora
    el bloque de test anterior, y se evalúa en un nuevo bloque siguiente.

    Yields tuplas (X_train, X_test, y_train, y_test) por cada fold.
    """
    feat_cols = get_feature_cols(features_df)
    target_cols = get_target_cols(features_df)
    X = features_df[feat_cols].values.astype(np.float32)
    y = features_df[target_cols].values.astype(np.int32)

    n = len(X)
    train_end = initial_train
    while train_end + step <= n:
        test_start = train_end
        test_end = min(train_end + step, n)
        yield X[:train_end], X[test_start:test_end], y[:train_end], y[test_start:test_end]
        train_end = test_end


def load_features(csv_path: str = "data/processed/features.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col="fecha", parse_dates=True)
