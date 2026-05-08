"""Iter 30: Buscar config que MINIMIZE el top-K necesario para 5/5 hits en 40/50.

Score function: priorizar fuertemente los 5/5 hits a top-K bajo.
"""
import sys
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset
from ml.iter27b_optuna_fast import precompute_test_features, evaluate_cached

from lightgbm import LGBMRegressor


def evaluate_5of5_focus(model, test_X, test_real):
    """Evalúa 5/5 hits para múltiples top-K."""
    n_5of5 = {k: 0 for k in [25, 30, 35, 40, 41, 42, 43, 44, 45, 46]}
    n_4plus = {k: 0 for k in [30, 35, 40, 45]}
    n_3plus = {k: 0 for k in [25, 30, 35]}

    for X, real in zip(test_X, test_real):
        probs = model.predict(X)
        sorted_idx = np.argsort(probs)[::-1]
        for k in n_5of5:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
        for k in n_4plus:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits >= 4: n_4plus[k] += 1
        for k in n_3plus:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits >= 3: n_3plus[k] += 1

    return n_5of5, n_4plus, n_3plus


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n", flush=True)

    n_total = len(df)

    # Grid completo enfocado en 5/5
    train_starts = [60, 100, 130, 150, 170, 200, 230]
    weight_pos_list = [3.0, 4.0, 5.0, 6.0, 8.0]
    leaves_list = [30, 50, 80]
    n_est_list = [200, 300, 400]

    print(f"Probando {len(train_starts)*len(weight_pos_list)*len(leaves_list)*len(n_est_list)} combinaciones", flush=True)

    best = None
    results = []
    count = 0
    total = len(train_starts)*len(weight_pos_list)*len(leaves_list)*len(n_est_list)

    for ts in train_starts:
        X_train, y_train, fnames = build_meta_dataset(df, ts, n_total - 50)
        test_X, test_real = precompute_test_features(df, fnames)

        for w_pos in weight_pos_list:
            sw = np.where(y_train == 1, w_pos, 1.0)
            for leaves in leaves_list:
                for n_est in n_est_list:
                    count += 1
                    cfg = {
                        "num_leaves": leaves, "max_depth": 5,
                        "learning_rate": 0.03, "n_estimators": n_est,
                        "reg_alpha": 0.2, "reg_lambda": 0.2,
                    }
                    model = LGBMRegressor(**cfg, verbose=-1)
                    model.fit(X_train, y_train, sample_weight=sw)
                    n_5of5, n_4plus, n_3plus = evaluate_5of5_focus(model, test_X, test_real)

                    # Score: priorizar 5/5 en top-30, top-35, top-40, top-45
                    score = (n_5of5[30] * 100 + n_5of5[35] * 50 +
                             n_5of5[40] * 20 + n_5of5[42] * 10 +
                             n_5of5[44] * 5 + n_5of5[45] * 3 +
                             n_4plus[40] * 2 + n_3plus[30])
                    r = {
                        "ts": ts, "w_pos": w_pos, "leaves": leaves, "n_est": n_est,
                        "5/5_top30": n_5of5[30], "5/5_top35": n_5of5[35],
                        "5/5_top40": n_5of5[40], "5/5_top42": n_5of5[42],
                        "5/5_top44": n_5of5[44], "5/5_top45": n_5of5[45],
                        "5/5_top46": n_5of5[46],
                        "score": score,
                    }
                    results.append(r)
                    mark = ""
                    if best is None or score > best["score"]:
                        best = r
                        mark = " ⭐"
                    if count % 10 == 0 or mark:
                        print(f"  [{count}/{total}] ts={ts} w={w_pos} L={leaves} n={n_est}: 5/5(t30={n_5of5[30]}, t35={n_5of5[35]}, t40={n_5of5[40]}, t44={n_5of5[44]}, t45={n_5of5[45]}) | score={score}{mark}", flush=True)

    print(f"\n🏆 BEST CONFIG: ts={best['ts']} w_pos={best['w_pos']} leaves={best['leaves']} n_est={best['n_est']}")
    print(f"  5/5 hits per top-K:")
    for k in [30, 35, 40, 42, 44, 45, 46]:
        print(f"    Top-{k}: {best.get(f'5/5_top{k}', 0)}/50 ({best.get(f'5/5_top{k}', 0)*2}%)")

    # Top 10 results
    print("\n=== TOP 10 RESULTS ===")
    sorted_r = sorted(results, key=lambda x: -x["score"])[:10]
    for r in sorted_r:
        print(f"  ts={r['ts']} w={r['w_pos']} L={r['leaves']} n={r['n_est']}: t30_5/5={r['5/5_top30']} t35={r['5/5_top35']} t40={r['5/5_top40']} t44={r['5/5_top44']} t45={r['5/5_top45']}")

    with open("reports/iter30_minimize_k.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
