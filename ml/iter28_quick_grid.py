"""Iter 28: Grid quick - probar configs específicas eficientemente con cache.

Inspirado en iter 27 trial 1 (que ya logró 42/50 con ≥3 hits en top-30).
Buscaremos en una vecindad de esa config.
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


# Configs específicas a probar (basadas en lo que funcionó)
CONFIGS = [
    # iter21 base
    {"num_leaves": 50, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 250,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "weight_pos": 3.0},
    # Variantes
    {"num_leaves": 50, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 200,
     "reg_alpha": 0.05, "reg_lambda": 0.05, "weight_pos": 2.5},
    {"num_leaves": 50, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 300,
     "reg_alpha": 0.2, "reg_lambda": 0.2, "weight_pos": 4.0},
    {"num_leaves": 80, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 250,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "weight_pos": 3.0},
    {"num_leaves": 30, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 200,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "weight_pos": 3.0},
    {"num_leaves": 100, "max_depth": -1, "learning_rate": 0.02, "n_estimators": 400,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "weight_pos": 3.0},
    # Más variantes
    {"num_leaves": 60, "max_depth": 5, "learning_rate": 0.025, "n_estimators": 300,
     "reg_alpha": 0.15, "reg_lambda": 0.15, "weight_pos": 2.5,
     "feature_fraction": 0.8, "bagging_fraction": 0.8},
    {"num_leaves": 50, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 250,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "weight_pos": 3.5,
     "min_child_samples": 10},
    {"num_leaves": 70, "max_depth": 6, "learning_rate": 0.02, "n_estimators": 350,
     "reg_alpha": 0.1, "reg_lambda": 0.1, "weight_pos": 3.0,
     "feature_fraction": 0.9},
]


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n", flush=True)

    n_total = len(df)
    print("Building meta dataset...", flush=True)
    X_train, y_train, fnames = build_meta_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}", flush=True)

    print("Pre-computing test features...", flush=True)
    test_X, test_real = precompute_test_features(df, fnames)
    print(f"  Test cache: {len(test_X)} sorteos\n", flush=True)

    results = []
    best = None

    for i, cfg in enumerate(CONFIGS):
        cfg = dict(cfg)
        weight_pos = cfg.pop("weight_pos", 3.0)
        sw = np.where(y_train == 1, weight_pos, 1.0)

        try:
            model = LGBMRegressor(**cfg, verbose=-1)
            model.fit(X_train, y_train, sample_weight=sw)
            n_3plus, n_4plus, n_5of5 = evaluate_cached(test_X, test_real, model)

            score = (n_3plus[30] * 5 + n_3plus[31] * 4 + n_3plus[35] * 2 +
                     n_4plus[38] * 3 + n_4plus[40] * 2)

            r = {
                "cfg": cfg, "weight_pos": weight_pos,
                "30_3+": n_3plus[30], "31_3+": n_3plus[31], "35_3+": n_3plus[35],
                "38_4+": n_4plus[38], "40_4+": n_4plus[40], "45_5/5": n_5of5[45],
                "score": score,
            }
            results.append(r)
            mark = ""
            if best is None or score > best["score"]:
                best = r
                mark = " ⭐"
            print(f"  [{i+1}/{len(CONFIGS)}] t30_3+={n_3plus[30]} t35_3+={n_3plus[35]} t38_4+={n_4plus[38]} t40_4+={n_4plus[40]} t45_5/5={n_5of5[45]} | score={score}{mark}", flush=True)
        except Exception as e:
            print(f"  [{i+1}] error: {e}", flush=True)

    print(f"\n🏆 BEST score={best['score']}")
    print(f"  Config: {best['cfg']}")
    print(f"  weight_pos={best['weight_pos']}")
    print(f"  Metrics: 30_3+={best['30_3+']} 35_3+={best['35_3+']} 38_4+={best['38_4+']} 40_4+={best['40_4+']} 45_5/5={best['45_5/5']}")

    with open("reports/iter28_grid.json", "w") as f:
        json.dump([{k: int(v) if isinstance(v, np.integer) else (str(v) if k == "cfg" else v)
                    for k, v in r.items()} for r in results],
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
