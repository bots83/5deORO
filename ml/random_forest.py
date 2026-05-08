"""Modelo Random Forest multilabel."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


class RandomForestMultilabel:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        random_state: int = 42,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestMultilabel":
        base = RandomForestClassifier(**self.params)
        self.model = MultiOutputClassifier(base, n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.model.predict_proba(X)
        cols = []
        for p in probas:
            if p.shape[1] == 1:
                cols.append(np.zeros(X.shape[0], dtype=np.float32))
            else:
                cols.append(p[:, 1])
        return np.column_stack(cols)
