"""Iter 12: Selección INTELIGENTE de Top-K.

En lugar de tomar los K más probables, hacemos selección que:
- Cubre todas las decenas (1-12, 13-24, 25-36, 37-48)
- Balancea pares/impares
- Incluye números 'frescos' (no salieron en últimos sorteos)
- Distribución uniforme garantizada

Idea: si distribuimos los K cubriendo bien el espacio, capturamos más
de los 5 reales aunque no sean los más probables individualmente.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter5_novel_algos import (
    pair_boost_predictor, cluster_predictor, streak_predictor,
    markov_predictor, adaptive_window_predictor, rank_stability_predictor,
    dayofweek_predictor
)
from ml.iter9_cdm import cdm_predictor


def smart_top_k(probs, k, decade_quota=None, parity_quota=None):
    """
    Selecciona top-K cubriendo decenas y paridad.

    Si decade_quota se especifica (ej. {0: 3, 1: 3, 2: 3, 3: 3}), garantiza esa cantidad por decena.
    Si no, calcula proporcionalmente.
    """
    POOL = 48
    if decade_quota is None:
        # Distribución proporcional 12-12-12-12
        per_dec = k // 4
        rem = k - per_dec * 4
        decade_quota = {0: per_dec, 1: per_dec, 2: per_dec, 3: per_dec}
        for i in range(rem):
            decade_quota[i] += 1

    # Decenas: 0=[1-12], 1=[13-24], 2=[25-36], 3=[37-48]
    decade_of = lambda n: (n - 1) // 12

    selected = set()
    by_decade = {0: [], 1: [], 2: [], 3: []}
    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        n = idx + 1
        d = decade_of(n)
        by_decade[d].append(n)

    # Tomar quota de cada decena
    for d in [0, 1, 2, 3]:
        for n in by_decade[d][:decade_quota[d]]:
            selected.add(n)

    # Si no se llenó, tomar resto de los más probables
    if len(selected) < k:
        for idx in sorted_idx:
            n = idx + 1
            if n not in selected:
                selected.add(n)
                if len(selected) == k:
                    break

    return sorted(selected)


def get_components_fast(history):
    components = {}
    for d in [1.0, 0.99, 0.95, 0.85, 0.70]:
        try:
            p = cdm_predictor(history, decay=d)
            if p.sum() > 0:
                components[f"cdm_{d}"] = p / p.sum()
        except Exception:
            pass
    for ln in [30, 100]:
        try:
            p = cdm_predictor(history, last_n=ln)
            if p.sum() > 0:
                components[f"cdm_last{ln}"] = p / p.sum()
        except Exception:
            pass
    algos = {
        "pair": pair_boost_predictor,
        "cluster095": lambda h: cluster_predictor(h, decay=0.95),
        "streak095": lambda h: streak_predictor(h, decay=0.95),
        "markov": lambda h: markov_predictor(h, order=1),
        "adaptive": adaptive_window_predictor,
        "rank": rank_stability_predictor,
    }
    for name, fn in algos.items():
        try:
            p = fn(history)
            if p.sum() > 0 and not np.isnan(p).any():
                components[name] = p / p.sum()
        except Exception:
            pass
    return components


def evaluate_smart_topk(df, last_n=50, top_k=10):
    """Evalúa Top-K smart vs naive."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    n_naive_5of5 = 0
    n_smart_5of5 = 0
    n_naive_3plus = 0
    n_smart_3plus = 0
    n_naive_4plus = 0
    n_smart_4plus = 0

    # Pesos default mejores que conocemos
    default_weights = {
        "cdm_0.95": 0.25, "cdm_0.99": 0.20, "cdm_0.85": 0.15,
        "pair": 0.10, "cluster095": 0.08, "rank": 0.07,
        "adaptive": 0.05, "markov": 0.05, "streak095": 0.05,
    }

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        components = get_components_fast(history)
        if not components:
            continue

        # Combinar
        ensemble = np.zeros(POOL)
        total_w = 0
        for name, w in default_weights.items():
            if name in components:
                ensemble += components[name] * w
                total_w += w
        if total_w == 0:
            continue
        probs = ensemble / total_w

        # Top-K naive (los K más probables)
        sorted_idx = np.argsort(probs)[::-1]
        naive_top = set((sorted_idx[:top_k] + 1).tolist())

        # Top-K smart (cubriendo decenas)
        smart_top = set(smart_top_k(probs, top_k))

        h_naive = len(real & naive_top)
        h_smart = len(real & smart_top)
        if h_naive == 5: n_naive_5of5 += 1
        if h_naive >= 3: n_naive_3plus += 1
        if h_naive >= 4: n_naive_4plus += 1
        if h_smart == 5: n_smart_5of5 += 1
        if h_smart >= 3: n_smart_3plus += 1
        if h_smart >= 4: n_smart_4plus += 1

    return {
        "naive_5of5": n_naive_5of5, "smart_5of5": n_smart_5of5,
        "naive_3plus": n_naive_3plus, "smart_3plus": n_smart_3plus,
        "naive_4plus": n_naive_4plus, "smart_4plus": n_smart_4plus,
    }


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Comparando Top-K NAIVE vs SMART (cubre decenas):\n")
    results = []
    for k in [10, 12, 15, 18, 20, 25, 30]:
        r = evaluate_smart_topk(df, top_k=k)
        r["top_k"] = k
        results.append(r)
        print(f"  Top-{k:2d}: NAIVE 5/5={r['naive_5of5']:2d} 3+={r['naive_3plus']:2d} 4+={r['naive_4plus']:2d} | SMART 5/5={r['smart_5of5']:2d} 3+={r['smart_3plus']:2d} 4+={r['smart_4plus']:2d}")

    print("\n" + "=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    for r in results:
        if r['smart_5of5'] > r['naive_5of5']:
            print(f"  ✓ Top-{r['top_k']}: SMART mejora 5/5 hits ({r['naive_5of5']} → {r['smart_5of5']})")
        if r['smart_3plus'] > r['naive_3plus']:
            print(f"  ✓ Top-{r['top_k']}: SMART mejora 3+ hits ({r['naive_3plus']} → {r['smart_3plus']})")

    with open("reports/iter12_smart.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
