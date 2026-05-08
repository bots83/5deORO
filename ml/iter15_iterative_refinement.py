"""Iter 15: Refinamiento iterativo basado en resultados pasados.

Idea: usar TODOS los resultados de iters anteriores y hacer ensemble FINAL.
Stack los mejores predictores con LightGBM como meta-learner.
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

try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except ImportError:
    LGBM_OK = False


def get_predictor_outputs(history):
    """Para un history, genera las predicciones de TODOS los predictores."""
    outputs = {}

    cdm_configs = [
        ("cdm_0.99", lambda h: cdm_predictor(h, decay=0.99)),
        ("cdm_0.95", lambda h: cdm_predictor(h, decay=0.95)),
        ("cdm_0.85", lambda h: cdm_predictor(h, decay=0.85)),
        ("cdm_0.70", lambda h: cdm_predictor(h, decay=0.70)),
        ("cdm_last30", lambda h: cdm_predictor(h, last_n=30)),
        ("cdm_last100", lambda h: cdm_predictor(h, last_n=100)),
        ("cdm_last150", lambda h: cdm_predictor(h, last_n=150)),
    ]
    for name, fn in cdm_configs:
        try:
            p = fn(history)
            if p.sum() > 0 and not np.isnan(p).any():
                outputs[name] = p / p.sum()
        except Exception:
            pass

    algos = {
        "pair": pair_boost_predictor,
        "cluster095": lambda h: cluster_predictor(h, decay=0.95),
        "cluster085": lambda h: cluster_predictor(h, decay=0.85),
        "streak095": lambda h: streak_predictor(h, decay=0.95),
        "markov": lambda h: markov_predictor(h, order=1),
        "adaptive": adaptive_window_predictor,
        "rank": rank_stability_predictor,
    }
    for name, fn in algos.items():
        try:
            p = fn(history)
            if p.sum() > 0 and not np.isnan(p).any():
                outputs[name] = p / p.sum()
        except Exception:
            pass

    return outputs


def build_meta_dataset(df, train_start, train_end):
    """Para cada sorteo en [train_start, train_end), genera (features, target)
    donde features son las probs de cada predictor por número, target es 1 si salió.
    """
    X_list = []
    y_list = []
    feature_names = None

    for idx in range(train_start, train_end):
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue
        if feature_names is None:
            feature_names = sorted(outputs.keys())

        # Para cada número, su feature vector son las probs de cada predictor
        for num in range(POOL):
            features = [outputs.get(name, np.ones(POOL)/POOL)[num] for name in feature_names]
            X_list.append(features)

            sorteo = df.iloc[idx]
            real = {sorteo[f"n{i}"] for i in range(1, 6)}
            y_list.append(1 if (num + 1) in real else 0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), feature_names


def evaluate_meta(df, model, feature_names, last_n=50):
    n_total = len(df)
    start = n_total - last_n
    top_ks = [10, 12, 15, 18, 20, 25, 30, 35, 40]
    n_5of5 = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue

        # Features para cada número
        X = []
        for num in range(POOL):
            features = [outputs.get(name, np.ones(POOL)/POOL)[num] for name in feature_names]
            X.append(features)
        X = np.array(X, dtype=np.float32)

        probs = model.predict(X)  # (POOL,)
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    return {f"top{k}_5of5": n_5of5[k] for k in top_ks} | \
           {f"top{k}_4plus": n_4plus[k] for k in top_ks} | \
           {f"top{k}_3plus": n_3plus[k] for k in top_ks}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    if not LGBM_OK:
        print("LightGBM no disponible")
        return

    n_total = len(df)
    test_size = 50

    # Train: [60, n_total - 50)
    print("Building meta-training data...")
    X_train, y_train, feature_names = build_meta_dataset(df, 60, n_total - test_size)
    print(f"Train: {X_train.shape}, features: {feature_names}")

    print("\nTraining LightGBM meta-learner...")
    model = LGBMRegressor(num_leaves=31, max_depth=5, learning_rate=0.03,
                          n_estimators=200, verbose=-1)
    model.fit(X_train, y_train)

    print("\nEvaluating on last 50 sorteos...")
    r = evaluate_meta(df, model, feature_names, last_n=test_size)

    print("\nResultados:")
    print(f"{'Top-K':<10} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 12, 15, 18, 20, 25, 30, 35, 40]:
        print(f"top-{k:<6} {r[f'top{k}_5of5']:>5} {r[f'top{k}_4plus']:>5} {r[f'top{k}_3plus']:>5}")

    print(f"\n🎯 META 35/50:")
    metas = [("5/5", "5of5"), ("4+", "4plus"), ("3+", "3plus")]
    for label, suffix in metas:
        for k in [10, 12, 15, 18, 20, 25, 30, 35, 40]:
            if r[f"top{k}_{suffix}"] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({r[f'top{k}_{suffix}']}/50) ✅")
                break
        else:
            print(f"  {label} ≥ 35/50: NO en rangos probados")

    # Feature importance
    print("\nFeature importance:")
    fi = model.feature_importances_
    for name, imp in sorted(zip(feature_names, fi), key=lambda x: -x[1]):
        print(f"  {name:25s}: {imp}")

    with open("reports/iter15_meta.json", "w") as f:
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in r.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
