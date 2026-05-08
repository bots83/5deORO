"""Iter 17b: Quick hyperparam tuning - solo unas pocas configs prometedoras."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset

from lightgbm import LGBMRegressor


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

    return {f"top{k}_5of5": n_5of5[k] for k in top_ks} | \
           {f"top{k}_4plus": n_4plus[k] for k in top_ks} | \
           {f"top{k}_3plus": n_3plus[k] for k in top_ks}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos", flush=True)

    n_total = len(df)
    print("\nBuilding meta dataset...", flush=True)
    X_train, y_train, feature_names = build_meta_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}", flush=True)

    # Configs prometedoras (no exhaustivo)
    configs = [
        # Default que funcionó bien
        {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 200},
        # Variaciones
        {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.05, "n_estimators": 100},
        {"num_leaves": 63, "max_depth": 6, "learning_rate": 0.02, "n_estimators": 300},
        {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 200},
        {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.03, "n_estimators": 200},
        {"num_leaves": 50, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 250, "reg_alpha": 0.1, "reg_lambda": 0.1},
        {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 200, "min_child_samples": 5},
        {"num_leaves": 127, "max_depth": 7, "learning_rate": 0.01, "n_estimators": 500},
    ]

    print(f"\nProbando {len(configs)} configs...\n", flush=True)
    best = None
    results = []
    for i, cfg in enumerate(configs):
        try:
            model = LGBMRegressor(**cfg, verbose=-1)
            model.fit(X_train, y_train)
            r = evaluate(df, model, feature_names)
            r["cfg"] = cfg
            score = r["top25_3plus"] * 5 + r["top30_3plus"] * 3 + r["top20_3plus"] * 2 + r["top15_3plus"]
            r["score"] = score
            results.append(r)
            mark = ""
            if best is None or score > best["score"]:
                best = r
                mark = " ⭐"
            print(f"  [{i+1}/{len(configs)}] {cfg} | t15_3+={r['top15_3plus']} t20_3+={r['top20_3plus']} t25_3+={r['top25_3plus']} t30_3+={r['top30_3plus']} t35_3+={r['top35_3plus']} | score={score}{mark}", flush=True)
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}", flush=True)

    print(f"\n=== BEST CFG: {best['cfg']} ===", flush=True)
    print(f"Score: {best['score']}")
    for k in [10, 15, 20, 25, 30, 35, 40]:
        print(f"  Top-{k}: 5/5={best[f'top{k}_5of5']}, 4+={best[f'top{k}_4plus']}, 3+={best[f'top{k}_3plus']}")

    # K mín para 35/50
    print(f"\n🎯 K mínimo para 35/50:")
    for label, suffix in [("3+", "3plus"), ("4+", "4plus"), ("5/5", "5of5")]:
        for k in range(5, 49):
            if best.get(f"top{k}_{suffix}", 0) >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({best[f'top{k}_{suffix}']}/50) ✅")
                break

    with open("reports/iter17b_tuned.json", "w") as f:
        json.dump([{k: int(v) if isinstance(v, np.integer) else v for k, v in r.items() if k not in ["cfg"]} for r in results],
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
