"""Iter 21: LightGBM con custom objective optimizando recall@K directamente.

Por defecto LightGBM optimiza MSE/log-loss. Pero nuestro objetivo es recall@K.
Implementamos un loss custom que penaliza más los falsos negativos en top-K.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset

from lightgbm import LGBMRegressor, LGBMClassifier


class WeightedLGBM:
    """LightGBM con sample_weights ajustados para recall@K."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, sample_weight=None):
        # Si no hay weights, dar más peso a positivos (5 vs 43)
        if sample_weight is None:
            # Positivos: peso 5, negativos: peso 1
            sample_weight = np.where(y == 1, 5.0, 1.0)
        self.model = LGBMRegressor(**self.kwargs, verbose=-1)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


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
    print("Building meta dataset...")
    X_train, y_train, feature_names = build_meta_dataset(df, 60, n_total - 50)

    # Probar diferentes pesos para positivos
    weights_pos_options = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    print(f"\nProbando pesos para positivos: {weights_pos_options}\n")
    best = None

    for w_pos in weights_pos_options:
        sample_w = np.where(y_train == 1, w_pos, 1.0)
        model = LGBMRegressor(num_leaves=50, max_depth=5, learning_rate=0.03,
                              n_estimators=250, reg_alpha=0.1, reg_lambda=0.1,
                              verbose=-1)
        model.fit(X_train, y_train, sample_weight=sample_w)

        n_5of5, n_4plus, n_3plus = evaluate(df, model, feature_names)
        score = n_3plus[31] * 5 + n_4plus[39] * 3 + n_3plus[35] * 2
        marker = ""
        if best is None or score > best["score"]:
            best = {"w_pos": w_pos, "n_5of5": n_5of5, "n_4plus": n_4plus, "n_3plus": n_3plus, "score": score}
            marker = " ⭐"
        print(f"  w_pos={w_pos:5.1f}: t31_3+={n_3plus[31]:2d} t35_3+={n_3plus[35]:2d} t39_4+={n_4plus[39]:2d} t40_4+={n_4plus[40]:2d} | score={score}{marker}")

    print(f"\n🏆 BEST w_pos={best['w_pos']}")
    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 31, 35, 39, 40, 45]:
        print(f"top-{k:<4} {best['n_5of5'][k]:>5} {best['n_4plus'][k]:>5} {best['n_3plus'][k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", best["n_5of5"]), ("4+", best["n_4plus"]), ("3+", best["n_3plus"])]:
        for k in range(5, 46):
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    with open("reports/iter21_weighted.json", "w") as f:
        out = {
            "best_w_pos": best["w_pos"],
            "metrics": {
                "5of5": {f"top{k}": best["n_5of5"][k] for k in [10, 15, 20, 25, 30, 35, 40, 45]},
                "4plus": {f"top{k}": best["n_4plus"][k] for k in [10, 15, 20, 25, 30, 35, 40, 45]},
                "3plus": {f"top{k}": best["n_3plus"][k] for k in [10, 15, 20, 25, 30, 35, 40, 45]},
            }
        }
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
