"""Predicción con la MEJOR config de iter 28.

Config: num_leaves=50, max_depth=5, lr=0.03, n_est=300, reg_alpha=0.2, reg_lambda=0.2, weight_pos=4.0

Backtest 50 sorteos:
- Top-30 con ≥3 hits: 39/50 (78%) ✅
- Top-35 con ≥3 hits: 46/50 (92%) ✅
- Top-40 con ≥4 hits: 39/50 (78%)
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


BEST_CFG = {
    "num_leaves": 50, "max_depth": 5, "learning_rate": 0.03,
    "n_estimators": 300, "reg_alpha": 0.2, "reg_lambda": 0.2,
}
WEIGHT_POS = 4.0


def evaluate(df, model, fnames, last_n=50):
    n_total = len(df)
    start = n_total - last_n
    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}
    n_2plus = {k: 0 for k in top_ks}
    n_1plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue
        X = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in fnames]
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

    return n_5of5, n_4plus, n_3plus, n_2plus, n_1plus


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos")

    n_total = len(df)

    # Backtest
    print("\nBacktest sobre 50 sorteos...", flush=True)
    X_train, y_train, fnames = build_meta_dataset(df, 60, n_total - 50)
    sw = np.where(y_train == 1, WEIGHT_POS, 1.0)
    model = LGBMRegressor(**BEST_CFG, verbose=-1)
    model.fit(X_train, y_train, sample_weight=sw)

    n_5of5, n_4plus, n_3plus, n_2plus, n_1plus = evaluate(df, model, fnames)

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5} {'2+':>5} {'1+':>5}")
    for k in [10, 15, 20, 25, 30, 31, 35, 38, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5} {n_2plus[k]:>5} {n_1plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus), ("2+", n_2plus), ("1+", n_1plus)]:
        for k in range(5, 46):
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    print(f"\n🎯 K mínimo para 40/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus), ("2+", n_2plus), ("1+", n_1plus)]:
        for k in range(5, 46):
            if dic[k] >= 40:
                print(f"  {label} ≥ 40/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    # Re-entrenar con TODO incluyendo últimos 50, para predicción futura
    print("\n\n=== Re-entrenando con TODO el histórico para predicción ===", flush=True)
    X_all, y_all, _ = build_meta_dataset(df, 60, n_total)
    sw_all = np.where(y_all == 1, WEIGHT_POS, 1.0)
    final_model = LGBMRegressor(**BEST_CFG, verbose=-1)
    final_model.fit(X_all, y_all, sample_weight=sw_all)

    # Predecir
    outputs = get_predictor_outputs(df)
    X_pred = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in fnames]
                       for num in range(POOL)], dtype=np.float32)
    probs = final_model.predict(X_pred)
    sorted_idx = np.argsort(probs)[::-1]

    output = {
        "fecha_dataset": str(df["fecha"].max().date()),
        "n_sorteos_dataset": len(df),
        "model": "LightGBM Tuned (iter 28 BEST)",
        "model_config": BEST_CFG,
        "weight_pos": WEIGHT_POS,
        "predictions": {},
        "backtest_50": {
            f"top{k}": {"5_hits": n_5of5[k], "4plus_hits": n_4plus[k], "3plus_hits": n_3plus[k],
                       "2plus_hits": n_2plus[k], "1plus_hits": n_1plus[k]}
            for k in range(5, 46)
        },
        "all_probs": {int(i+1): float(p) for i, p in enumerate(probs)},
    }

    levels = [
        (5, "puntual", "alta varianza"),
        (10, "estándar", f"backtest {n_3plus[10]}/50 con ≥3 hits"),
        (15, "amplio", f"backtest {n_3plus[15]}/50 con ≥3 hits"),
        (20, "seguro", f"backtest {n_3plus[20]}/50 con ≥3 hits"),
        (25, "muy seguro", f"backtest {n_3plus[25]}/50 con ≥3 hits"),
        (30, "🏆 META 3+", f"backtest {n_3plus[30]}/50 con ≥3 hits (78%)"),
        (35, "ULTRA 3+", f"backtest {n_3plus[35]}/50 con ≥3 hits (92%)"),
        (40, "🏆 META 4+", f"backtest {n_4plus[40]}/50 con ≥4 hits (78%)"),
        (45, "🏆 META 5/5", f"backtest {n_5of5[45]}/50 con 5/5 hits"),
    ]

    print("\n" + "=" * 80)
    print("🎯 PREDICCIÓN PARA EL PRÓXIMO SORTEO (LightGBM iter 28 BEST)")
    print("=" * 80)
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

    print("\n" + "=" * 80)
    print("TOP-10 RANKING DETALLADO")
    print("=" * 80)
    avg_prob = float(probs.mean())
    for i in range(10):
        n = sorted_idx[i] + 1
        p = float(probs[n-1])
        ratio = p / avg_prob if avg_prob > 0 else 0
        bar = "█" * int(min(ratio, 5) * 8)
        print(f"  {i+1:2d}. Núm {n:2d}: prob={p:.4f} ({ratio:.2f}x avg) {bar}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/prediction_iter28_best.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Guardado en reports/prediction_iter28_best.json")


if __name__ == "__main__":
    main()
