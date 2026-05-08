"""Iter 7: Stacking con meta-learner.

Idea: para cada sorteo, generamos predicciones con N modelos base, y entrenamos
un meta-modelo (LightGBM) que combina las probabilidades base con features
adicionales para producir la predicción final.

Esto da más flexibilidad que el random search de pesos lineales.
"""
import sys
import time
import json
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

try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except ImportError:
    LGBM_OK = False


def get_base_probs(history, train_df):
    """Genera matriz (n_models, POOL) de probabilidades."""
    if len(history) < 30 or train_df is None or len(train_df) < 20:
        return None

    feat_cols = [c for c in train_df.columns if not c.startswith("target_")]
    target_cols = [c for c in train_df.columns if c.startswith("target_")]
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.int32)
    feats_pred = _build_features_for_row(history)
    X_pred = np.array([[feats_pred.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

    probs_list = []
    names = []
    for d in [0.99, 0.95, 0.85, 0.70]:
        try:
            m = BetaBinomialModel(decay=d)
            m.fit(X_train, y_train)
            p = m.predict_proba(X_pred)[0]
            probs_list.append(p / (p.sum() + 1e-9))
            names.append(f"bb_{d}")
        except Exception:
            pass

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
                probs_list.append(p / p.sum())
                names.append(name)
        except Exception:
            pass

    if not probs_list:
        return None, None
    return np.stack(probs_list), names


def precompute(df, last_n=80):
    """Precompute training features y base probs."""
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


def evaluate_stacking(df, cache, last_n_train=30, last_n_test=50):
    """
    Stacking:
    - Para los últimos `last_n_train + last_n_test` sorteos, genera base probs
    - Entrena meta-learner con primeros `last_n_train` (X = probs_base, y = real)
    - Evalúa en últimos `last_n_test`
    """
    n_total = len(df)
    total = last_n_train + last_n_test
    start = max(60, n_total - total)

    # Recolectar features (probs base) y targets
    print(f"  Recolectando datos para stacking ({total} sorteos)...", flush=True)
    X_meta = []  # cada fila será las probs concatenadas (n_models * POOL)
    y_meta = []  # cada fila será el target binario (POOL,)

    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = cache.get(idx)
        result = get_base_probs(history, train_df)
        if result[0] is None:
            continue
        probs_matrix, names = result

        # X: (n_models, POOL) → flatten o usar como input para meta
        # Para meta por número: X_per_num = probs_matrix.T (POOL, n_models)
        X_per_num = probs_matrix.T  # (POOL, n_models)

        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        y_per_num = np.array([1 if (n+1) in real else 0 for n in range(POOL)])

        X_meta.append(X_per_num)
        y_meta.append(y_per_num)

    X_meta_arr = np.concatenate(X_meta, axis=0)  # (n_sorteos * POOL, n_models)
    y_meta_arr = np.concatenate(y_meta, axis=0)  # (n_sorteos * POOL,)
    print(f"  X_meta shape: {X_meta_arr.shape}, y_meta shape: {y_meta_arr.shape}", flush=True)

    # Split: primeros last_n_train para train, últimos last_n_test para test
    train_size = last_n_train * POOL
    X_train_meta = X_meta_arr[:train_size]
    y_train_meta = y_meta_arr[:train_size]
    X_test_meta = X_meta_arr[train_size:]
    y_test_meta = y_meta_arr[train_size:]

    print(f"  Train: {X_train_meta.shape[0]}, Test: {X_test_meta.shape[0]}", flush=True)

    if not LGBM_OK:
        print("  LightGBM no disponible, usando modelo simple")
        # Modelo simple: regresión sobre la suma ponderada
        from sklearn.linear_model import LogisticRegression
        meta = LogisticRegression(max_iter=200)
        meta.fit(X_train_meta, y_train_meta)
        # Para predict, devolver prob de clase 1
        test_probs = meta.predict_proba(X_test_meta)[:, 1]
    else:
        meta = LGBMRegressor(
            num_leaves=15, max_depth=4, learning_rate=0.05,
            n_estimators=100, verbose=-1
        )
        meta.fit(X_train_meta, y_train_meta)
        test_probs = meta.predict(X_test_meta)

    # Reshape: (n_test_sorteos, POOL)
    test_probs_per_sorteo = test_probs.reshape(last_n_test, POOL)

    # Evaluar
    top_ks = [10, 15, 20, 25, 30, 35, 40]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for i in range(last_n_test):
        idx = n_total - last_n_test + i
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{j}"] for j in range(1, 6)}
        probs = test_probs_per_sorteo[i]
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


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    last_n_test = 50
    last_n_train = 50  # entrenamos meta con los 50 anteriores

    cache = precompute(df, last_n=last_n_test + last_n_train)

    print("\n=== STACKING META-LEARNER ===")
    r = evaluate_stacking(df, cache, last_n_train=last_n_train, last_n_test=last_n_test)
    print("\nResultados (5/5 hits dentro del top-K):")
    for k in [10, 15, 20, 25, 30, 35, 40]:
        n = r[f"top{k}_5of5"]
        check = "✅" if n >= 35 else f"❌ falta {35-n}"
        print(f"  Top-{k:2d}: {n:2d}/50 {check}")
    print("\nResultados (3+ hits dentro del top-K):")
    for k in [10, 15, 20, 25, 30]:
        print(f"  Top-{k:2d}: {r[f'top{k}_3plus']:2d}/50 con 3+ hits")

    with open("reports/iter7_stacking.json", "w") as f:
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in r.items()},
                  f, indent=2, default=str)
    print(f"\n✓ Guardado en reports/iter7_stacking.json")


if __name__ == "__main__":
    main()
