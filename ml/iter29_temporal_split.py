"""Iter 29: Probar diferentes splits temporales del entrenamiento.

Tal vez entrenar con menos datos pero más recientes da mejor predicción.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset
from ml.iter27b_optuna_fast import precompute_test_features, evaluate_cached

from lightgbm import LGBMRegressor


BEST_CFG = {
    "num_leaves": 50, "max_depth": 5, "learning_rate": 0.03,
    "n_estimators": 300, "reg_alpha": 0.2, "reg_lambda": 0.2,
}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n", flush=True)

    n_total = len(df)

    # Probar diferentes train_starts
    train_starts = [60, 100, 150, 200, 230]
    weight_pos_options = [2.0, 3.0, 4.0, 5.0, 6.0]

    print(f"Probando {len(train_starts)} train_starts × {len(weight_pos_options)} pesos\n", flush=True)

    results = []
    best = None

    for train_start in train_starts:
        for w_pos in weight_pos_options:
            X_train, y_train, fnames = build_meta_dataset(df, train_start, n_total - 50)
            sw = np.where(y_train == 1, w_pos, 1.0)
            model = LGBMRegressor(**BEST_CFG, verbose=-1)
            model.fit(X_train, y_train, sample_weight=sw)

            test_X, test_real = precompute_test_features(df, fnames)
            n_3plus, n_4plus, n_5of5 = evaluate_cached(test_X, test_real, model)

            score = (n_3plus[30] * 5 + n_3plus[31] * 4 + n_3plus[35] * 2 +
                     n_4plus[38] * 3 + n_4plus[40] * 2 + n_5of5[45])
            r = {
                "train_start": train_start, "w_pos": w_pos,
                "30_3+": n_3plus[30], "31_3+": n_3plus[31], "35_3+": n_3plus[35],
                "38_4+": n_4plus[38], "40_4+": n_4plus[40], "45_5/5": n_5of5[45],
                "score": score,
            }
            results.append(r)
            mark = ""
            if best is None or score > best["score"]:
                best = r
                mark = " ⭐"
            print(f"  ts={train_start} w={w_pos}: t30_3+={n_3plus[30]} t35_3+={n_3plus[35]} t40_4+={n_4plus[40]} t45_5/5={n_5of5[45]} | score={score}{mark}", flush=True)

    print(f"\n🏆 BEST: train_start={best['train_start']}, w_pos={best['w_pos']}")
    print(f"  Metrics: t30_3+={best['30_3+']} t31_3+={best['31_3+']} t35_3+={best['35_3+']} t38_4+={best['38_4+']} t40_4+={best['40_4+']} t45_5/5={best['45_5/5']}")

    with open("reports/iter29_temporal.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
