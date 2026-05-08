"""Iter 26: Bagging masivo - 50 modelos LightGBM con diferentes seeds y subsets.

Para cada modelo:
- Subsample del 80% del training set
- Random feature subset
- Random hyperparams
Promediamos las predicciones.

Esto reduce varianza y mejora estabilidad.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset

from lightgbm import LGBMRegressor


def train_bagged_models(X, y, n_models=50, seed=42):
    """Entrena n_models con bootstrap."""
    rng = np.random.default_rng(seed)
    models = []
    for i in range(n_models):
        # Bootstrap sample
        n = len(X)
        idx = rng.choice(n, size=int(n * 0.8), replace=True)
        X_sub = X[idx]
        y_sub = y[idx]
        sw_sub = np.where(y_sub == 1, 3.0, 1.0)

        # Random hyperparams
        cfg = {
            "num_leaves": int(rng.choice([31, 50, 63, 100])),
            "max_depth": int(rng.choice([-1, 4, 5, 6, 7])),
            "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
            "n_estimators": int(rng.choice([100, 200, 250, 300])),
            "reg_alpha": float(rng.choice([0.01, 0.1, 0.3])),
            "reg_lambda": float(rng.choice([0.01, 0.1, 0.3])),
            "feature_fraction": 0.85,
            "random_state": int(rng.integers(0, 100000)),
        }
        model = LGBMRegressor(**cfg, verbose=-1)
        model.fit(X_sub, y_sub, sample_weight=sw_sub)
        models.append(model)
        if (i+1) % 10 == 0:
            print(f"  Trained {i+1}/{n_models}", flush=True)
    return models


def predict_bagged(models, X):
    """Promedio de predicciones."""
    preds = np.zeros(X.shape[0])
    for m in models:
        preds += m.predict(X)
    return preds / len(models)


def evaluate(df, models, fnames, last_n=50):
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
        X = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in fnames]
                      for num in range(POOL)], dtype=np.float32)
        probs = predict_bagged(models, X)
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
    print(f"Dataset: {len(df)} sorteos", flush=True)

    n_total = len(df)
    print("\nBuilding meta dataset...", flush=True)
    X_train, y_train, fnames = build_meta_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}", flush=True)

    print("\nTraining 50 bagged models...", flush=True)
    models = train_bagged_models(X_train, y_train, n_models=50)

    print("\nEvaluating bagged ensemble...", flush=True)
    n_5of5, n_4plus, n_3plus = evaluate(df, models, fnames)

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 31, 35, 38, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}", flush=True)

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in range(5, 46):
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    print(f"\n🎯 K mínimo para 40/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in range(5, 46):
            if dic[k] >= 40:
                print(f"  {label} ≥ 40/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    with open("reports/iter26_bagging.json", "w") as f:
        out = {f"top{k}_5of5": n_5of5[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_4plus": n_4plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_3plus": n_3plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]}
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in out.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
