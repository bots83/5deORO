"""Iter 6: Encontrar el TOP-K mínimo donde alcanzamos 35/50 con 5/5 hits.

Busca exhaustivamente combinando:
- BB ensembles
- Pair boost
- Cluster
- Streak
- Markov
- Pesos optimizados por grid search
"""
import sys
import time
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL, NUM_COLS
from ml.bayesian import BetaBinomialModel
from ml.iter5_novel_algos import (
    pair_boost_predictor, cluster_predictor, streak_predictor,
    markov_predictor, adaptive_window_predictor, rank_stability_predictor,
    dayofweek_predictor
)


def precompute(df, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    cache = {}
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = build_features(history, min_history=30)
        cache[idx] = train_df
    return cache


def predict_super(history, train_df, weights):
    """Combina BB + algoritmos novedosos con pesos."""
    if len(history) < 30 or train_df is None or len(train_df) < 20:
        return np.ones(POOL) / POOL

    feat_cols = [c for c in train_df.columns if not c.startswith("target_")]
    target_cols = [c for c in train_df.columns if c.startswith("target_")]
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.int32)
    feats_pred = _build_features_for_row(history)
    X_pred = np.array([[feats_pred.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

    components = {}
    # BB con varios decays
    for d in [0.99, 0.95, 0.85, 0.70]:
        try:
            m = BetaBinomialModel(decay=d)
            m.fit(X_train, y_train)
            p = m.predict_proba(X_pred)[0]
            components[f"bb_{d}"] = p / (p.sum() + 1e-9)
        except Exception:
            pass

    # Algoritmos novedosos
    algos = {
        "pair": pair_boost_predictor,
        "cluster095": lambda h: cluster_predictor(h, decay=0.95),
        "cluster085": lambda h: cluster_predictor(h, decay=0.85),
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

    # Combinar con pesos
    ensemble = np.zeros(POOL)
    total_w = 0
    for name, w in weights.items():
        if name in components and w > 0:
            ensemble += components[name] * w
            total_w += w

    if total_w == 0:
        return np.ones(POOL) / POOL
    return ensemble / total_w


def evaluate(df, train_cache, weights, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 15, 20, 25, 30, 35, 40]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        probs = predict_super(history, train_cache.get(idx), weights)
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


def random_search_weights(df, train_cache, n_trials=100, seed=42):
    """Random search sobre el espacio de pesos."""
    rng = np.random.default_rng(seed)
    component_names = ["bb_0.99", "bb_0.95", "bb_0.85", "bb_0.7",
                       "pair", "cluster095", "cluster085",
                       "streak095", "streak085", "markov",
                       "adaptive", "rank", "day"]

    best = None
    history = []
    print(f"Random search con {n_trials} trials...")
    for i in range(n_trials):
        # Sample weights de Dirichlet (suman a 1)
        # Sparsify: con prob 0.3, peso = 0
        raw = rng.exponential(1.0, size=len(component_names))
        mask = rng.random(len(component_names)) < 0.7  # 70% activos
        raw = raw * mask
        if raw.sum() == 0:
            continue
        weights_arr = raw / raw.sum()
        weights = dict(zip(component_names, weights_arr))

        r = evaluate(df, train_cache, weights, last_n=50)
        r["weights"] = {k: round(v, 3) for k, v in weights.items() if v > 0}
        history.append(r)

        # Score: maximizar sorteos donde top-10 tiene 5 hits + bonus por k pequeños
        score = (r["top10_5of5"] * 100 + r["top15_5of5"] * 50 +
                 r["top20_5of5"] * 30 + r["top25_5of5"] * 15 +
                 r["top30_5of5"] * 8 + r["top35_5of5"] * 4 +
                 r["top40_5of5"] * 2 +
                 r["top10_3plus"] * 5 + r["top15_3plus"] * 2)
        r["score"] = score

        if best is None or score > best["score"]:
            best = r
            print(f"  [{i+1}/{n_trials}] NEW BEST score={score:.0f}: t10_5/5={r['top10_5of5']} t20_5/5={r['top20_5of5']} t30_5/5={r['top30_5of5']} t40_5/5={r['top40_5of5']}")
        elif (i+1) % 10 == 0:
            print(f"  [{i+1}/{n_trials}] cur score={score:.0f} (best={best['score']:.0f})")

    return best, history


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Pre-computing train features...")
    cache = precompute(df, last_n=50)
    print("Done\n")

    best, hist = random_search_weights(df, cache, n_trials=80)

    print("\n" + "=" * 80)
    print("MEJOR CONFIGURACIÓN ENCONTRADA")
    print("=" * 80)
    print(f"Score: {best['score']}")
    print(f"\nResultados por top-K (sorteos con 5/5 hits):")
    for k in [10, 15, 20, 25, 30, 35, 40]:
        n = best[f"top{k}_5of5"]
        check = "✅" if n >= 35 else f"❌ falta {35-n}"
        print(f"  Top-{k:2d}: {n:2d}/50 {check}")
    print(f"\nResultados por top-K (sorteos con 3+ hits):")
    for k in [10, 15, 20, 25, 30]:
        n = best[f"top{k}_3plus"]
        print(f"  Top-{k:2d}: {n:2d}/50 con 3+ aciertos")

    print("\nPesos óptimos:")
    for k, v in sorted(best["weights"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    with open("reports/iter6_best.json", "w") as f:
        out = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in best.items()}
        json.dump(out, f, indent=2, default=str)

    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in hist]).to_csv("reports/iter6_search.csv", index=False)


if __name__ == "__main__":
    main()
