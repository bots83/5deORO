"""Modelo Bayesiano: Beta-Binomial con prior uniforme.

Para cada número, modelamos su probabilidad de aparición como
Beta(α + k, β + n - k) donde k = apariciones, n = sorteos vistos.

Bajo H0 IID el prior es Beta(1,1) = uniforme.
"""
import numpy as np

POOL = 48
DRAW_SIZE = 5


class BetaBinomialModel:
    """
    Posterior Beta para cada número. Los parámetros se actualizan con la historia.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, decay: float = 1.0):
        """
        decay: factor exponencial para dar más peso a observaciones recientes.
            decay=1.0 → todas las observaciones pesan igual
            decay=0.99 → observaciones antiguas decaen
        """
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.posteriors_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BetaBinomialModel":
        # y: (n_sorteos, POOL)
        n_sorteos, pool = y.shape
        if self.decay == 1.0:
            k = y.sum(axis=0)  # apariciones por número
            n = n_sorteos
            alpha_post = self.alpha + k
            beta_post = self.beta + n - k
        else:
            # Pesos exponenciales
            weights = self.decay ** np.arange(n_sorteos)[::-1]
            k = (y * weights[:, None]).sum(axis=0)
            n = weights.sum()
            alpha_post = self.alpha + k
            beta_post = self.beta + n - k

        # Media posterior: alpha / (alpha + beta)
        self.posteriors_ = alpha_post / (alpha_post + beta_post)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.tile(self.posteriors_, (n, 1))


class MarkovChainModel:
    """
    Modelo de Markov de orden 1: P(num_t aparece | num_t-1 apareció).
    Si las apariciones son IID, no debería haber dependencia significativa.
    """
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.p_given_prev_ = None  # (POOL, 2) — prob de aparecer dado prev
        self.base_rate_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarkovChainModel":
        # P(num aparece | num apareció en t-1)
        n_sorteos, pool = y.shape
        self.p_given_prev_ = np.zeros((pool, 2))
        for num in range(pool):
            seq = y[:, num]
            transitions = np.zeros((2, 2))  # [prev, curr]
            for t in range(1, n_sorteos):
                transitions[seq[t-1], seq[t]] += 1
            for prev in [0, 1]:
                total = transitions[prev].sum() + 2 * self.smoothing
                self.p_given_prev_[num, prev] = (
                    (transitions[prev, 1] + self.smoothing) / total
                )

        self.base_rate_ = y.mean(axis=0)
        return self

    def predict_proba(self, X: np.ndarray, y_context: np.ndarray | None = None) -> np.ndarray:
        n = X.shape[0]
        pool = self.p_given_prev_.shape[0]
        out = np.zeros((n, pool), dtype=np.float32)
        if y_context is None:
            return np.tile(self.base_rate_, (n, 1))
        for i in range(n):
            if i == 0:
                out[i] = self.base_rate_
            else:
                prev_state = y_context[i - 1]
                for num in range(pool):
                    out[i, num] = self.p_given_prev_[num, int(prev_state[num])]
        return out
