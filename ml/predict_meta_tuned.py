"""Predicción final con meta-learner TUNEADO (mejor config de iter 17b).

Config: num_leaves=50, max_depth=5, learning_rate=0.03, n_estimators=250,
        reg_alpha=0.1, reg_lambda=0.1

Backtest 50 sorteos:
- Top-31 con ≥3 hits: 37/50 (74%) ✅
- Top-39 con ≥4 hits: 41/50 (82%) ✅
- Top-35 con ≥3 hits: 44/50 (88%) ✅
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


# Mejor config de iter 17b
BEST_CFG = {
    "num_leaves": 50,
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 250,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos")

    n_total = len(df)

    # Entrenar con TODO menos los últimos 50 (para backtest)
    print("\nEntrenando meta-learner TUNEADO...")
    X_train, y_train, feature_names = build_meta_dataset(df, 60, n_total - 50)

    model = LGBMRegressor(**BEST_CFG, verbose=-1)
    model.fit(X_train, y_train)
    print(f"  Train: {X_train.shape}")
    print(f"  Config: {BEST_CFG}")

    # Backtest sobre últimos 50
    print("\nBacktest sobre últimos 50...")
    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}
    n_2plus = {k: 0 for k in top_ks}
    n_1plus = {k: 0 for k in top_ks}

    for idx in range(n_total - 50, n_total):
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
            if hits >= 2: n_2plus[k] += 1
            if hits >= 1: n_1plus[k] += 1

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5} {'2+':>5} {'1+':>5}")
    for k in [10, 12, 15, 18, 20, 25, 30, 31, 35, 38, 39, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5} {n_2plus[k]:>5} {n_1plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus), ("2+", n_2plus), ("1+", n_1plus)]:
        for k in top_ks:
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    # Predicción para próximo sorteo
    print("\n" + "=" * 80)
    print("🎯 PREDICCIÓN PARA EL PRÓXIMO SORTEO (config tuneada)")
    print("=" * 80)
    outputs = get_predictor_outputs(df)
    X_pred = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in feature_names]
                       for num in range(POOL)], dtype=np.float32)
    probs = model.predict(X_pred)
    sorted_idx = np.argsort(probs)[::-1]

    output = {
        "fecha_dataset": str(df["fecha"].max().date()),
        "n_sorteos_dataset": len(df),
        "model": "LightGBM meta-learner TUNEADO (iter 17b)",
        "model_config": BEST_CFG,
        "features": feature_names,
        "predictions": {},
        "backtest_50": {
            f"top{k}": {"5_hits": n_5of5[k], "4plus_hits": n_4plus[k], "3plus_hits": n_3plus[k],
                       "2plus_hits": n_2plus[k], "1plus_hits": n_1plus[k]}
            for k in top_ks
        },
        "all_probs": {int(i+1): float(p) for i, p in enumerate(probs)},
    }

    levels = [
        (5, "puntual", "alta varianza"),
        (10, "estándar", f"backtest {n_3plus[10]}/50 con ≥3 hits"),
        (15, "amplio", f"backtest {n_3plus[15]}/50 con ≥3 hits"),
        (20, "seguro", f"backtest {n_3plus[20]}/50 con ≥3 hits"),
        (25, "muy seguro", f"backtest {n_3plus[25]}/50 con ≥3 hits"),
        (30, "alta cobertura", f"backtest {n_3plus[30]}/50 con ≥3 hits"),
        (31, "🏆 META 3+", f"backtest {n_3plus[31]}/50 con ≥3 hits ✅"),
        (35, "ultra confianza", f"backtest {n_3plus[35]}/50 con ≥3 hits"),
        (39, "🏆 META 4+", f"backtest {n_4plus[39]}/50 con ≥4 hits ✅"),
        (40, "máxima precisión", f"backtest {n_4plus[40]}/50 con ≥4 hits"),
    ]

    for k, label, note in levels:
        nums = sorted((sorted_idx[:k] + 1).tolist())
        cov = float(probs[sorted_idx[:k]].sum() / probs.sum())
        output["predictions"][f"top{k}"] = {
            "numbers": nums, "label": label, "note": note,
            "prob_total": round(cov, 4),
            "backtest": output["backtest_50"][f"top{k}"],
        }
        print(f"\n📊 TOP-{k:2d} ({label}): {note}")
        print(f"   {' '.join(f'{n:02d}' for n in nums)}")

    # Top-10 detalle
    baseline = 5/48
    print("\n" + "=" * 80)
    print("TOP-10 RANKING DETALLADO")
    print("=" * 80)
    for i in range(10):
        n = sorted_idx[i] + 1
        p = float(probs[n-1])
        # Normalizar para razón
        avg_prob = float(probs.mean())
        ratio = p / avg_prob
        bar = "█" * int(min(ratio, 5) * 8)
        print(f"  {i+1:2d}. Núm {n:2d}: prob={p:.4f} ({ratio:.2f}x avg) {bar}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/prediction_meta_tuned.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Guardado en reports/prediction_meta_tuned.json")


if __name__ == "__main__":
    main()
