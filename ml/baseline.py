"""Baseline: predice los números más frecuentes históricamente."""
import numpy as np


class FrequencyBaseline:
    """
    Predice probabilidades basadas en la frecuencia histórica acumulada.
    La probabilidad de cada número es su frecuencia relativa en el histórico.
    """

    def __init__(self):
        self.freq_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FrequencyBaseline":
        # y: (n_sorteos, 48) — conteo acumulado de apariciones por número
        self.freq_ = y.mean(axis=0)  # frecuencia promedio por número
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.tile(self.freq_, (n, 1))

    def get_params(self, deep=True):
        return {}


class RandomBaseline:
    """Baseline puramente aleatorio para validar la métrica."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.random.rand(X.shape[0], 48)
        return probs / probs.sum(axis=1, keepdims=True)
