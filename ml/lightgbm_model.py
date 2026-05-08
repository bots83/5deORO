"""LightGBM multilabel — alternativa rápida a XGBoost."""
import numpy as np
from sklearn.multioutput import MultiOutputClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class LightGBMMultilabel:
    def __init__(
        self,
        max_depth: int = 5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm no instalado")
        self.params = dict(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMMultilabel":
        base = LGBMClassifier(**self.params)
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
