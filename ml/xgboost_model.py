"""Modelo XGBoost multilabel — modelo primario."""
import numpy as np
from sklearn.multioutput import MultiOutputClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class XGBoostMultilabel:
    def __init__(
        self,
        max_depth: int = 4,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost no instalado")
        self.params = dict(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostMultilabel":
        base = XGBClassifier(**self.params)
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

    def feature_importances(self, feature_names: list[str] | None = None) -> dict:
        importances = np.mean(
            [est.feature_importances_ for est in self.model.estimators_], axis=0
        )
        if feature_names:
            return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1]))
        return {f"f{i}": v for i, v in enumerate(importances)}
