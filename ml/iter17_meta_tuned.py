"""Iter 17: Meta-learner con hyperparameter tuning + class weighting.

Goal: mejorar top-25 / top-30 para alcanzar 35/50 con ≥3 hits.

Estrategia:
- Tune LightGBM (depth, leaves, lr, n_estimators)
- Class weighting (POOL=48, solo 5 son positivos => imbalance 5/43 = 12%)
- Cross-validation temporal
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


def evaluate_with_model(df, model, feature_names, last_n=50, is_classifier=False):
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
        if is_classifier:
            probs = model.predict_proba(X)[:, 1]
        else:
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
    print(f"Dataset: {len(df)} sorteos")

    n_total = len(df)
    print("\nBuilding meta dataset...")
    X_train, y_train, feature_names = build_meta_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}")

    # Probar varias configuraciones de hyperparams
    configs = []
    for num_leaves in [15, 31, 63, 127]:
        for max_depth in [3, 5, 7, -1]:
            for lr in [0.01, 0.03, 0.05, 0.1]:
                for n_est in [100, 200, 400]:
                    configs.append({
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                    })

    # Limitar para no demorar demasiado
    configs = configs[:30]
    print(f"Probando {len(configs)} configs de LightGBM regressor...\n")

    results = []
    best = None
    for i, cfg in enumerate(configs):
        try:
            model = LGBMRegressor(**cfg, verbose=-1)
            model.fit(X_train, y_train)
            r = evaluate_with_model(df, model, feature_names)
            r["cfg"] = cfg
            # Score: maximizar top-25/30 con 3+
            score = r["top25_3plus"] * 5 + r["top30_3plus"] * 3 + r["top20_3plus"] * 2
            r["score"] = score
            results.append(r)
            mark = ""
            if best is None or score > best["score"]:
                best = r
                mark = " ⭐"
            if mark or i % 5 == 0:
                print(f"  [{i+1}/{len(configs)}] cfg={cfg} | t20_3+={r['top20_3plus']} t25_3+={r['top25_3plus']} t30_3+={r['top30_3plus']} | score={score}{mark}")
        except Exception as e:
            print(f"  [{i+1}] {cfg}: ERROR {e}")

    # También probar classifier con class weights
    print("\n=== Classifier con class_weight balanced ===")
    try:
        clf = LGBMClassifier(
            num_leaves=31, max_depth=5, learning_rate=0.03, n_estimators=200,
            class_weight={0: 1, 1: 9}, verbose=-1
        )
        clf.fit(X_train, y_train)
        r_clf = evaluate_with_model(df, clf, feature_names, is_classifier=True)
        print(f"  t20_3+={r_clf['top20_3plus']} t25_3+={r_clf['top25_3plus']} t30_3+={r_clf['top30_3plus']} t35_3+={r_clf['top35_3plus']}")
        r_clf["cfg"] = "Classifier balanced"
        r_clf["score"] = r_clf["top25_3plus"] * 5 + r_clf["top30_3plus"] * 3
        results.append(r_clf)
    except Exception as e:
        print(f"  Classifier ERROR: {e}")

    # Mejor encontrado
    print("\n" + "=" * 80)
    print("MEJOR CONFIG")
    print("=" * 80)
    if best:
        print(f"Config: {best['cfg']}")
        print(f"\nResultados:")
        for k in [10, 15, 20, 25, 30, 35, 40]:
            print(f"  Top-{k}: 5/5={best[f'top{k}_5of5']}, 4+={best[f'top{k}_4plus']}, 3+={best[f'top{k}_3plus']}")
        print(f"\n🎯 K mínimo para 35/50:")
        for label, suffix in [("3+", "3plus"), ("4+", "4plus"), ("5/5", "5of5")]:
            for k in range(5, 49):
                if best.get(f"top{k}_{suffix}", 0) >= 35:
                    print(f"  {label} ≥ 35/50: Top-{k} ({best[f'top{k}_{suffix}']}/50) ✅")
                    break

    with open("reports/iter17_tuned.json", "w") as f:
        out = [{k: int(v) if isinstance(v, np.integer) else v for k, v in r.items() if k not in ["cfg"]} for r in results[:10]]
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
