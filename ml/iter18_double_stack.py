"""Iter 18: Doble stacking - meta-learner del meta-learner.

Usar las predicciones de iter 15 (meta-learner) como features
junto con las features originales para un meta-meta-learner.

También incluir features secundarias agregadas:
- Suma probabilidades del top-10 (mide concentración)
- Std de probabilidades (mide dispersión)
- Rank vs frecuencia histórica
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs

from lightgbm import LGBMRegressor


def get_extended_features(history, num):
    """Features avanzadas para el número `num`."""
    outputs = get_predictor_outputs(history)
    if not outputs:
        return None

    base_feats = []
    for name in sorted(outputs.keys()):
        base_feats.append(outputs[name][num])

    # Features secundarias agregadas
    all_probs = np.array([outputs.get(n, np.ones(POOL)/POOL)[num] for n in sorted(outputs.keys())])
    extra_feats = [
        float(all_probs.mean()),
        float(all_probs.std()),
        float(all_probs.min()),
        float(all_probs.max()),
        float(all_probs.max() - all_probs.min()),
        float(np.median(all_probs)),
    ]

    # Frecuencia histórica del número
    n_total = len(history)
    count = sum(1 for _, row in history.iterrows() if num + 1 in [int(row[c]) for c in NUM_COLS])
    freq_hist = count / n_total
    extra_feats.append(freq_hist)

    # Last 30 sorteos
    last_30 = history.tail(30)
    count_30 = sum(1 for _, row in last_30.iterrows() if num + 1 in [int(row[c]) for c in NUM_COLS])
    freq_30 = count_30 / max(len(last_30), 1)
    extra_feats.append(freq_30)

    # Diferencia freq reciente vs hist
    extra_feats.append(freq_30 - freq_hist)

    # ¿Salió en el último sorteo?
    if len(history) > 0:
        last_set = set([int(history.iloc[-1][c]) for c in NUM_COLS])
        extra_feats.append(1.0 if (num + 1) in last_set else 0.0)
    else:
        extra_feats.append(0.0)

    # Gap (cuántos sorteos lleva sin salir)
    gap = 0
    for j in range(len(history) - 1, -1, -1):
        if num + 1 in [int(history.iloc[j][c]) for c in NUM_COLS]:
            break
        gap += 1
    extra_feats.append(float(gap))

    return base_feats + extra_feats


def build_extended_dataset(df, train_start, train_end):
    X_list = []
    y_list = []
    feature_names = None

    for idx in range(train_start, train_end):
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue
        if feature_names is None:
            base_names = sorted(outputs.keys())
            extra_names = ["mean_p", "std_p", "min_p", "max_p", "range_p", "median_p",
                          "freq_hist", "freq_30", "freq_diff", "in_last", "gap"]
            feature_names = base_names + extra_names

        for num in range(POOL):
            feats = get_extended_features(history, num)
            if feats is None:
                continue
            X_list.append(feats)

            sorteo = df.iloc[idx]
            real = {sorteo[f"n{i}"] for i in range(1, 6)}
            y_list.append(1 if (num + 1) in real else 0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), feature_names


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos")

    n_total = len(df)
    print("\nBuilding extended meta-dataset...")
    X_train, y_train, feature_names = build_extended_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}, features: {len(feature_names)}")

    print("\nTraining extended meta-learner...")
    model = LGBMRegressor(num_leaves=63, max_depth=6, learning_rate=0.02,
                          n_estimators=400, reg_alpha=0.1, reg_lambda=0.1, verbose=-1)
    model.fit(X_train, y_train)

    print("\nEvaluating on last 50 sorteos...")
    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(n_total - 50, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        X = np.array([get_extended_features(history, num) for num in range(POOL)], dtype=np.float32)
        if any(x is None for x in X):
            continue
        probs = model.predict(X)
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 35, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in top_ks:
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    fi = model.feature_importances_
    print(f"\nTop 10 features importance:")
    for name, imp in sorted(zip(feature_names, fi), key=lambda x: -x[1])[:10]:
        print(f"  {name:25s}: {imp}")

    with open("reports/iter18_extended.json", "w") as f:
        out = {f"top{k}_5of5": n_5of5[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_4plus": n_4plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_3plus": n_3plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]}
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in out.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
