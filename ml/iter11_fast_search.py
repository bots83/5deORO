"""Iter 11: Fast random search SIN BB (que es el cuello de botella).

Solo CDM y algoritmos novedosos (rápidos). 500 trials.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL, NUM_COLS
from ml.iter5_novel_algos import (
    pair_boost_predictor, cluster_predictor, streak_predictor,
    markov_predictor, adaptive_window_predictor, rank_stability_predictor,
    dayofweek_predictor
)
from ml.iter9_cdm import cdm_predictor


def get_components_fast(history):
    """Solo componentes rápidos (sin BB que requiere build_features)."""
    components = {}

    # CDM con varios decays - rápido
    for d in [1.0, 0.99, 0.97, 0.95, 0.92, 0.85, 0.70]:
        try:
            p = cdm_predictor(history, decay=d)
            if p.sum() > 0:
                components[f"cdm_{d}"] = p / p.sum()
        except Exception:
            pass

    # CDM con last_n
    for ln in [30, 50, 100, 150]:
        try:
            p = cdm_predictor(history, last_n=ln)
            if p.sum() > 0:
                components[f"cdm_last{ln}"] = p / p.sum()
        except Exception:
            pass

    # Algoritmos novedosos
    algos = {
        "pair": pair_boost_predictor,
        "pair_50": lambda h: pair_boost_predictor(h, last_n_for_pairs=50),
        "cluster095": lambda h: cluster_predictor(h, decay=0.95),
        "cluster085": lambda h: cluster_predictor(h, decay=0.85),
        "cluster075": lambda h: cluster_predictor(h, decay=0.75),
        "streak095": lambda h: streak_predictor(h, decay=0.95),
        "streak085": lambda h: streak_predictor(h, decay=0.85),
        "markov": lambda h: markov_predictor(h, order=1),
        "adaptive": adaptive_window_predictor,
        "rank": rank_stability_predictor,
        "day": dayofweek_predictor,
    }
    for name, fn in algos.items():
        try:
            p = fn(history)
            if p.sum() > 0 and not np.isnan(p).any():
                components[name] = p / p.sum()
        except Exception:
            pass

    return components


def precompute_components(df, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    cache = {}
    print(f"Precomputing components for {n_total - start} sorteos...", flush=True)
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        cache[idx] = get_components_fast(history)
    print("done", flush=True)
    return cache


def evaluate(df, comp_cache, weights, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        components = comp_cache[idx]
        if not components:
            continue
        ensemble = np.zeros(POOL)
        total_w = 0
        for name, w in weights.items():
            if name in components and w > 0:
                ensemble += components[name] * w
                total_w += w
        if total_w == 0:
            continue
        probs = ensemble / total_w
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    return {
        **{f"top{k}_5of5": n_5of5[k] for k in top_ks},
        **{f"top{k}_4plus": n_4plus[k] for k in top_ks},
        **{f"top{k}_3plus": n_3plus[k] for k in top_ks},
    }


def random_search(df, comp_cache, n_trials=500, seed=42):
    rng = np.random.default_rng(seed)
    component_names = list(next(iter(comp_cache.values())).keys())
    print(f"Componentes: {len(component_names)}: {component_names}\n")

    best = None
    history = []
    for i in range(n_trials):
        # Mezcla de estrategias de muestreo
        strategy = rng.integers(0, 3)
        if strategy == 0:
            raw = rng.exponential(1.0, size=len(component_names))
            sparsity = rng.uniform(0.3, 0.95)
            mask = rng.random(len(component_names)) < sparsity
            raw = raw * mask
        elif strategy == 1:
            # Pocos componentes muy pesados
            raw = np.zeros(len(component_names))
            n_active = rng.integers(2, 5)
            indices = rng.choice(len(component_names), n_active, replace=False)
            raw[indices] = rng.exponential(1.0, size=n_active)
        else:
            # Todos pero con magnitudes muy diferentes
            raw = rng.exponential(0.3, size=len(component_names))

        if raw.sum() == 0:
            continue
        weights_arr = raw / raw.sum()
        weights = dict(zip(component_names, weights_arr))

        r = evaluate(df, comp_cache, weights, last_n=50)
        r["weights"] = {k: round(float(v), 3) for k, v in weights.items() if v > 0}

        # Score: priorizar 5/5 hits en top-Ks medianos
        score = (r["top10_5of5"] * 100 + r["top12_5of5"] * 80 +
                 r["top15_5of5"] * 60 + r["top18_5of5"] * 45 +
                 r["top20_5of5"] * 35 + r["top25_5of5"] * 25 +
                 r["top30_5of5"] * 15 + r["top35_5of5"] * 8 +
                 r["top40_5of5"] * 4 + r["top45_5of5"] * 2 +
                 r["top10_3plus"] * 12 + r["top15_3plus"] * 6 +
                 r["top10_4plus"] * 30 + r["top15_4plus"] * 15)
        r["score"] = score
        history.append(r)

        if best is None or score > best["score"]:
            best = r
            print(f"  [{i+1}/{n_trials}] NEW BEST score={score:.0f}: t10_5/5={r['top10_5of5']} t15={r['top15_5of5']} t20={r['top20_5of5']} t25={r['top25_5of5']} t30={r['top30_5of5']} | t10_3+={r['top10_3plus']} t15_3+={r['top15_3plus']} t10_4+={r['top10_4plus']}")
        elif (i+1) % 50 == 0:
            print(f"  [{i+1}/{n_trials}] cur best score={best['score']:.0f}")

    return best, history


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    comp_cache = precompute_components(df, last_n=50)

    print("=== RANDOM SEARCH 500 TRIALS ===")
    best, hist = random_search(df, comp_cache, n_trials=500)

    print("\n" + "=" * 80)
    print("MEJOR ENSEMBLE")
    print("=" * 80)
    print(f"Score: {best['score']:.0f}")
    print(f"\n5/5 hits:")
    for k in [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]:
        n = best[f"top{k}_5of5"]
        check = "✅" if n >= 35 else "❌"
        print(f"  Top-{k:2d}: {n:2d}/50 {check}")
    print(f"\n3+ hits:")
    for k in [10, 12, 15, 20, 25]:
        print(f"  Top-{k:2d}: {best[f'top{k}_3plus']:2d}/50")
    print(f"\n4+ hits:")
    for k in [10, 15, 20, 25, 30]:
        print(f"  Top-{k:2d}: {best[f'top{k}_4plus']:2d}/50")

    print("\nPesos (>0.01):")
    for k, v in sorted(best["weights"].items(), key=lambda x: -x[1]):
        if v > 0.01:
            print(f"  {k:25s}: {v:.3f}")

    with open("reports/iter11_best.json", "w") as f:
        out = {
            "score": best["score"],
            "weights": best["weights"],
            "metrics_5of5": {f"top{k}": best[f"top{k}_5of5"] for k in [10, 15, 20, 25, 30, 35, 40, 45]},
            "metrics_3plus": {f"top{k}": best[f"top{k}_3plus"] for k in [10, 15, 20, 25]},
            "metrics_4plus": {f"top{k}": best[f"top{k}_4plus"] for k in [10, 15, 20, 25, 30]},
        }
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
