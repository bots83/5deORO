"""Iter 22: Boosting el dataset con sorteos sintéticos cercanos.

Idea: bootstrap con jitter - generar variantes de sorteos históricos cambiando
1-2 números cerca para tener más muestras de entrenamiento.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset
from ml.iter20_deep_nn import get_features_for_sorteo

from lightgbm import LGBMRegressor


def synthetic_jitter(real_set, n_samples=10, max_change=2):
    """Genera variantes de un sorteo cambiando hasta max_change números."""
    rng = np.random.default_rng(42)
    variants = []
    for _ in range(n_samples):
        new_set = set(real_set)
        n_changes = rng.integers(1, max_change + 1)
        for _ in range(n_changes):
            # Quitar uno y poner otro al azar
            to_remove = rng.choice(list(new_set))
            new_set.remove(to_remove)
            choices = list(set(range(1, POOL + 1)) - new_set)
            to_add = rng.choice(choices)
            new_set.add(to_add)
        variants.append(new_set)
    return variants


def build_augmented_meta_dataset(df, train_start, train_end, augment_factor=2):
    """
    Construye dataset aumentado:
    - Para cada sorteo histórico, generamos `augment_factor` variantes sintéticas.
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

        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}

        # Original
        for num in range(POOL):
            features = [outputs.get(name, np.ones(POOL)/POOL)[num] for name in feature_names]
            X_list.append(features)
            y_list.append(1 if (num + 1) in real else 0)

        # Augmented: pequeñas variaciones del target
        for variant in synthetic_jitter(real, n_samples=augment_factor):
            for num in range(POOL):
                features = [outputs.get(name, np.ones(POOL)/POOL)[num] for name in feature_names]
                X_list.append(features)
                y_list.append(1 if (num + 1) in variant else 0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), feature_names


def evaluate(df, model, feature_names, last_n=50):
    n_total = len(df)
    start = n_total - last_n
    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue
        X = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in feature_names]
                      for num in range(POOL)], dtype=np.float32)
        probs = model.predict(X)
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    return n_5of5, n_4plus, n_3plus


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    n_total = len(df)

    # Probar diferentes factores de augmentación
    factors = [0, 1, 2, 3, 5]
    results = {}
    for factor in factors:
        print(f"\n=== Factor {factor} ===")
        if factor == 0:
            X, y, feature_names = build_meta_dataset(df, 60, n_total - 50)
            sw = np.where(y == 1, 3.0, 1.0)
        else:
            X, y, feature_names = build_augmented_meta_dataset(df, 60, n_total - 50, augment_factor=factor)
            sw = np.where(y == 1, 3.0, 1.0)

        print(f"  Train shape: {X.shape}")

        model = LGBMRegressor(num_leaves=50, max_depth=5, learning_rate=0.03, n_estimators=250,
                              reg_alpha=0.1, reg_lambda=0.1, verbose=-1)
        model.fit(X, y, sample_weight=sw)

        n_5of5, n_4plus, n_3plus = evaluate(df, model, feature_names)
        results[factor] = {"5of5": dict(n_5of5), "4plus": dict(n_4plus), "3plus": dict(n_3plus)}
        print(f"  Top-31_3+={n_3plus[31]} Top-35_3+={n_3plus[35]} Top-40_4+={n_4plus[40]}")

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"{'Factor':<8} {'t31_3+':>8} {'t35_3+':>8} {'t40_4+':>8} {'t45_5/5':>8}")
    for f, r in results.items():
        print(f"{f:<8} {r['3plus'][31]:>8} {r['3plus'][35]:>8} {r['4plus'][40]:>8} {r['5of5'][45]:>8}")

    with open("reports/iter22_synthetic.json", "w") as f:
        json.dump({str(k): {kk: int(v) if isinstance(v, np.integer) else v for kk, v in r["3plus"].items()}
                   for k, r in results.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
