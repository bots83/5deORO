"""Métricas de evaluación rigurosas con simulaciones Monte Carlo.

Sistema: 6 números del 1-45 sin reemplazo.
Baseline aleatorio teórico para top-K recall:
  E[hits] = 6 * K / 45
  Var[hits] = 6 * K * (45-K) * (45-6) / (45^2 * (45-1)) (hipergeométrica)
"""
import numpy as np
import pandas as pd

POOL = 48
DRAW_SIZE = 5


def top_k_recall(y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> float:
    """
    Para cada sorteo, selecciona los top-k números con mayor probabilidad predicha.
    Cuenta cuántos de los DRAW_SIZE números reales están en ese top-k.
    Retorna promedio (no normalizado).
    """
    assert y_true.shape == y_prob.shape
    n = y_true.shape[0]
    total = 0.0
    for i in range(n):
        top_k_idx = np.argsort(y_prob[i])[::-1][:k]
        total += y_true[i][top_k_idx].sum()
    return total / n


def random_baseline_expected(k: int, draw_size: int = DRAW_SIZE, pool: int = POOL) -> tuple[float, float]:
    """
    Distribución hipergeométrica para top-K recall bajo aleatoriedad.
    Retorna (esperanza, std).
    """
    mean = draw_size * k / pool
    # Var hipergeométrica: K(N-K)*n*(N-n)/(N^2*(N-1)) donde N=pool, K=draw_size, n=k
    if pool > 1:
        var = draw_size * (pool - draw_size) * k * (pool - k) / (pool ** 2 * (pool - 1))
        std = np.sqrt(max(var, 0))
    else:
        std = 0.0
    return mean, std


def monte_carlo_baseline(k: int, n_sorteos: int, n_sims: int = 10000,
                         draw_size: int = DRAW_SIZE, pool: int = POOL,
                         seed: int = 42) -> dict:
    """
    Simula `n_sims` corridas de un predictor aleatorio sobre `n_sorteos` sorteos
    para calcular la distribución empírica del top-K recall.
    """
    rng = np.random.default_rng(seed)
    recalls = []
    for _ in range(n_sims):
        total = 0.0
        for _ in range(n_sorteos):
            true_set = set(rng.choice(pool, draw_size, replace=False))
            pred_top_k = set(rng.choice(pool, k, replace=False))
            total += len(true_set & pred_top_k)
        recalls.append(total / n_sorteos)
    recalls = np.array(recalls)
    return {
        "media": float(recalls.mean()),
        "std": float(recalls.std()),
        "p2.5": float(np.percentile(recalls, 2.5)),
        "p97.5": float(np.percentile(recalls, 97.5)),
        "p5": float(np.percentile(recalls, 5)),
        "p95": float(np.percentile(recalls, 95)),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    k_values: list[int] | None = None,
) -> dict:
    if k_values is None:
        k_values = [5, 10, 15, 20]

    results = {"modelo": label}
    for k in k_values:
        recall = top_k_recall(y_true, y_prob, k=k)
        exp_random, std_random = random_baseline_expected(k)
        results[f"top{k}_recall"] = round(recall, 4)
        results[f"top{k}_esperado_aleatorio"] = round(exp_random, 4)
        results[f"top{k}_z_score"] = round(
            (recall - exp_random) / (std_random / np.sqrt(y_true.shape[0])) if std_random > 0 else 0.0, 4
        )
    return results


def compare_models(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).set_index("modelo")


def hit_rate_distribution(y_true: np.ndarray, y_prob: np.ndarray, k: int = 10) -> dict:
    """Distribución del número de hits por sorteo (no solo el promedio)."""
    n = y_true.shape[0]
    hits = []
    for i in range(n):
        top_k_idx = np.argsort(y_prob[i])[::-1][:k]
        hits.append(int(y_true[i][top_k_idx].sum()))
    hits = np.array(hits)
    counts = np.bincount(hits, minlength=DRAW_SIZE + 1)
    return {
        "media": float(hits.mean()),
        "std": float(hits.std()),
        "max": int(hits.max()),
        "min": int(hits.min()),
        "distribucion": {int(i): int(c) for i, c in enumerate(counts)},
    }
