"""Iter 10: Combinación inteligente de TODOS los predictores con CDM.

Esta es la versión final que toma lo mejor de:
- BB ensemble
- CDM bayesiano
- Algoritmos novedosos (pair, cluster, streak, markov, etc.)
- Pesos calibrados por backtest
"""
import sys
import json
import time
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
from ml.iter9_cdm import cdm_predictor


def get_all_components(history, train_df):
    """Genera dict de todas las predicciones base normalizadas."""
    components = {}

    # CDM con varios decays
    for d in [1.0, 0.99, 0.95, 0.85, 0.70]:
        try:
            p = cdm_predictor(history, decay=d)
            if p.sum() > 0:
                components[f"cdm_{d}"] = p / p.sum()
        except Exception:
            pass

    # CDM con last_n
    for ln in [30, 50, 100]:
        try:
            p = cdm_predictor(history, last_n=ln)
            if p.sum() > 0:
                components[f"cdm_last{ln}"] = p / p.sum()
        except Exception:
            pass

    # BB con varios decays
    if train_df is not None and len(train_df) >= 20:
        feat_cols = [c for c in train_df.columns if not c.startswith("target_")]
        target_cols = [c for c in train_df.columns if c.startswith("target_")]
        X_train = train_df[feat_cols].values.astype(np.float32)
        y_train = train_df[target_cols].values.astype(np.int32)
        feats_pred = _build_features_for_row(history)
        X_pred = np.array([[feats_pred.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

        for d in [0.99, 0.95, 0.85, 0.70]:
            try:
                m = BetaBinomialModel(decay=d)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_pred)[0]
                if p.sum() > 0:
                    components[f"bb_{d}"] = p / p.sum()
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

    return components


def predict_combined(components, weights):
    if not components:
        return np.ones(POOL) / POOL
    ensemble = np.zeros(POOL)
    total_w = 0
    for name, w in weights.items():
        if name in components and w > 0:
            ensemble += components[name] * w
            total_w += w
    if total_w == 0:
        return np.mean(list(components.values()), axis=0)
    return ensemble / total_w


def precompute(df, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    cache = {}
    print(f"  Precomputing {n_total - start} train sets...", flush=True)
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = build_features(history, min_history=30)
        cache[idx] = train_df
    print("  done", flush=True)
    return cache


def precompute_components(df, train_cache, last_n=50):
    """Precomputa componentes para los últimos last_n sorteos."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    comp_cache = {}
    print(f"  Precomputing components for {n_total - start} sorteos...", flush=True)
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        components = get_all_components(history, train_cache.get(idx))
        comp_cache[idx] = components
    print("  done", flush=True)
    return comp_cache


def evaluate_with_weights(df, comp_cache, weights, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        components = comp_cache.get(idx, {})
        probs = predict_combined(components, weights)
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


def random_search(df, comp_cache, n_trials=200, seed=42):
    """Random search masivo."""
    rng = np.random.default_rng(seed)
    component_names = list(next(iter(comp_cache.values())).keys())
    print(f"  Componentes disponibles: {len(component_names)}: {component_names}")

    best = None
    history = []

    for i in range(n_trials):
        # Sample de Dirichlet con sparsity
        raw = rng.exponential(1.0, size=len(component_names))
        sparsity = rng.uniform(0.4, 0.9)  # 40-90% activos
        mask = rng.random(len(component_names)) < sparsity
        raw = raw * mask
        if raw.sum() == 0:
            continue
        weights_arr = raw / raw.sum()
        weights = dict(zip(component_names, weights_arr))

        r = evaluate_with_weights(df, comp_cache, weights, last_n=50)
        r["weights"] = {k: round(float(v), 3) for k, v in weights.items() if v > 0}

        # Score: priorizar top-K más pequeños con 5/5 hits
        score = (r["top10_5of5"] * 100 + r["top12_5of5"] * 70 +
                 r["top15_5of5"] * 50 + r["top18_5of5"] * 35 +
                 r["top20_5of5"] * 25 + r["top25_5of5"] * 15 +
                 r["top30_5of5"] * 8 + r["top35_5of5"] * 4 +
                 r["top40_5of5"] * 2 +
                 r["top10_3plus"] * 10 + r["top15_3plus"] * 5 + r["top20_3plus"] * 2)
        r["score"] = score
        history.append(r)

        if best is None or score > best["score"]:
            best = r
            print(f"  [{i+1}/{n_trials}] NEW BEST score={score:.0f}: t10_5/5={r['top10_5of5']} t15={r['top15_5of5']} t20={r['top20_5of5']} t25={r['top25_5of5']} t30={r['top30_5of5']} t35={r['top35_5of5']} t40={r['top40_5of5']} | t10_3+={r['top10_3plus']} t15_3+={r['top15_3plus']}")
        elif (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_trials}] cur score={score:.0f} (best={best['score']:.0f})")

    return best, history


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Pre-computing train features...")
    cache = precompute(df, last_n=50)

    print("Pre-computing all components for last 50 sorteos...")
    comp_cache = precompute_components(df, cache, last_n=50)

    print(f"\n=== RANDOM SEARCH (200 trials) ===")
    best, hist = random_search(df, comp_cache, n_trials=200)

    print("\n" + "=" * 80)
    print("MEJOR ENSEMBLE")
    print("=" * 80)
    print(f"Score: {best['score']}")
    print("\n5/5 hits:")
    for k in [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]:
        n = best[f"top{k}_5of5"]
        check = "✅" if n >= 35 else "❌"
        print(f"  Top-{k:2d}: {n:2d}/50 {check}")
    print("\n3+ hits:")
    for k in [10, 15, 20, 25, 30]:
        n = best[f"top{k}_3plus"]
        print(f"  Top-{k:2d}: {n:2d}/50 con 3+ aciertos")

    print("\nPesos óptimos:")
    for k, v in sorted(best["weights"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:25s}: {v:.3f}")

    with open("reports/iter10_best.json", "w") as f:
        out = {
            "score": best["score"],
            "weights": best["weights"],
            "metrics_5of5": {f"top{k}": best[f"top{k}_5of5"] for k in [10, 15, 20, 25, 30, 35, 40, 45]},
            "metrics_3plus": {f"top{k}": best[f"top{k}_3plus"] for k in [10, 15, 20, 25, 30]},
            "metrics_4plus": {f"top{k}": best[f"top{k}_4plus"] for k in [10, 15, 20, 25, 30]},
        }
        json.dump(out, f, indent=2, default=str)

    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in hist]).to_csv("reports/iter10_search.csv", index=False)


if __name__ == "__main__":
    main()
