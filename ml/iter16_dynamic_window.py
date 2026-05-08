"""Iter 16: Window dinámico - encontrar la ventana óptima para cada sorteo.

En lugar de usar una ventana fija, usamos la ventana que mejor predice
los últimos N sorteos antes del sorteo a predecir.

Esto se basa en la hipótesis de que la "memoria" del sistema cambia con el tiempo.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter9_cdm import cdm_predictor
from ml.iter5_novel_algos import cluster_predictor, pair_boost_predictor


def find_best_window(history, validation_n=10, candidate_windows=[20, 30, 50, 80, 120, 200]):
    """
    Para cada ventana candidata, evalúa qué tan bien predice los últimos validation_n sorteos.
    Retorna la mejor ventana.
    """
    if len(history) < validation_n + 30:
        return 100  # default

    train_history = history.iloc[:-validation_n]
    val_sorteos = history.iloc[-validation_n:]

    best_window = 100
    best_score = -1
    for w in candidate_windows:
        if len(train_history) < w:
            continue
        score = 0
        for i in range(validation_n):
            sub_history = history.iloc[:len(train_history) + i]
            try:
                p = cdm_predictor(sub_history, last_n=w)
                if p.sum() > 0:
                    p = p / p.sum()
                    real = {int(val_sorteos.iloc[i][c]) for c in NUM_COLS}
                    sorted_idx = np.argsort(p)[::-1][:15]
                    top_set = set((sorted_idx + 1).tolist())
                    score += len(real & top_set)
            except Exception:
                pass
        if score > best_score:
            best_score = score
            best_window = w

    return best_window


def adaptive_predictor(history, validation_n=10):
    """Usa la window óptima para este sorteo."""
    best_w = find_best_window(history, validation_n=validation_n)
    cdm_p = cdm_predictor(history, last_n=best_w)
    cluster_p = cluster_predictor(history, decay=0.85)
    pair_p = pair_boost_predictor(history)

    if cdm_p.sum() > 0: cdm_p = cdm_p / cdm_p.sum()
    if cluster_p.sum() > 0: cluster_p = cluster_p / cluster_p.sum()
    if pair_p.sum() > 0: pair_p = pair_p / pair_p.sum()

    # Combinar con pesos similares a iter 11
    final = 0.74 * cluster_p + 0.22 * cdm_p + 0.04 * pair_p
    return final / final.sum()


def evaluate(df, predictor_fn, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 20, 25, 30, 35, 40, 45]
    n_5of5 = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        try:
            probs = predictor_fn(history)
        except Exception as e:
            print(f"  err: {e}")
            probs = np.ones(POOL) / POOL
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    return {**{f"top{k}_5of5": n_5of5[k] for k in top_ks},
            **{f"top{k}_4plus": n_4plus[k] for k in top_ks},
            **{f"top{k}_3plus": n_3plus[k] for k in top_ks}}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Adaptive window predictor (find best window per sorteo)...")
    r = evaluate(df, adaptive_predictor)
    print("\nResultados:")
    for k in [10, 12, 15, 20, 25, 30, 35, 40, 45]:
        print(f"  Top-{k}: 5/5={r[f'top{k}_5of5']}, 4+={r[f'top{k}_4plus']}, 3+={r[f'top{k}_3plus']}")


if __name__ == "__main__":
    main()
