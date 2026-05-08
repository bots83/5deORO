"""Iter 25: Análisis diagnóstico - ¿qué hace que un sorteo sea "predecible"?

Para cada uno de los 50 sorteos:
1. Generamos predicciones con varios modelos
2. Contamos hits en top-10
3. Analizamos: cuándo aciertan más, cuándo menos

Insight: si encontramos que SORTEOS DESPUÉS DE CIERTOS PATRONES son más predecibles,
podemos construir un meta-modelo "confidence" que dice cuándo fiarse.
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


WEIGHTED_CFG = {
    "num_leaves": 50, "max_depth": 5, "learning_rate": 0.03,
    "n_estimators": 250, "reg_alpha": 0.1, "reg_lambda": 0.1,
}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    n_total = len(df)

    # Entrenar
    print("Entrenando LGBM Weighted...")
    X_train, y_train, fnames = build_meta_dataset(df, 60, n_total - 50)
    sw = np.where(y_train == 1, 3.0, 1.0)
    model = LGBMRegressor(**WEIGHTED_CFG, verbose=-1)
    model.fit(X_train, y_train, sample_weight=sw)

    # Backtest detallado
    results = []
    print("\nBacktest con análisis...")
    for idx in range(n_total - 50, n_total):
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

        # Stats del sorteo predicho
        prev = df.iloc[idx - 1]
        prev_nums = sorted([int(prev[c]) for c in NUM_COLS])
        prev_sum = sum(prev_nums)
        prev_par = sum(1 for n in prev_nums if n % 2 == 0)
        prev_spread = prev_nums[-1] - prev_nums[0]
        # Confidence: spread of probabilities (varianza)
        prob_std = float(probs.std())
        prob_max = float(probs.max())
        prob_top10_sum = float(np.sort(probs)[::-1][:10].sum())

        # Hits
        top_10 = set((sorted_idx[:10] + 1).tolist())
        top_15 = set((sorted_idx[:15] + 1).tolist())
        top_20 = set((sorted_idx[:20] + 1).tolist())
        top_30 = set((sorted_idx[:30] + 1).tolist())
        h10 = len(real & top_10)
        h15 = len(real & top_15)
        h20 = len(real & top_20)
        h30 = len(real & top_30)

        # Real stats
        real_sum = sum(real)
        real_par = sum(1 for n in real if n % 2 == 0)

        results.append({
            "idx": idx,
            "fecha": str(sorteo["fecha"].date()),
            "real": sorted(real),
            "h10": h10, "h15": h15, "h20": h20, "h30": h30,
            "prob_std": prob_std,
            "prob_max": prob_max,
            "prob_top10_sum": prob_top10_sum,
            "prev_sum": prev_sum,
            "prev_par": prev_par,
            "prev_spread": prev_spread,
            "real_sum": real_sum,
            "real_par": real_par,
        })

    # Análisis
    df_r = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"Hits top-10 (avg): {df_r['h10'].mean():.2f}")
    print(f"  ≥3: {(df_r['h10'] >= 3).sum()}/50")
    print(f"  ≥2: {(df_r['h10'] >= 2).sum()}/50")
    print(f"  ≥1: {(df_r['h10'] >= 1).sum()}/50")
    print(f"  =0: {(df_r['h10'] == 0).sum()}/50")
    print(f"\nHits top-15 (avg): {df_r['h15'].mean():.2f}")
    print(f"  ≥3: {(df_r['h15'] >= 3).sum()}/50")
    print(f"\nHits top-20 (avg): {df_r['h20'].mean():.2f}")
    print(f"  ≥3: {(df_r['h20'] >= 3).sum()}/50")

    print("\nCorrelaciones:")
    print(f"  prob_std vs h10: {df_r['prob_std'].corr(df_r['h10']):.3f}")
    print(f"  prob_max vs h10: {df_r['prob_max'].corr(df_r['h10']):.3f}")
    print(f"  prob_top10_sum vs h10: {df_r['prob_top10_sum'].corr(df_r['h10']):.3f}")
    print(f"  prev_sum vs h10: {df_r['prev_sum'].corr(df_r['h10']):.3f}")

    # Sorteos donde acertamos vs no
    print("\nSorteos con h10 >= 2:")
    high = df_r[df_r['h10'] >= 2].sort_values('h10', ascending=False).head(15)
    for _, r in high.iterrows():
        print(f"  {r['fecha']}: real={r['real']}, h10={r['h10']}, prob_std={r['prob_std']:.4f}")

    print("\nSorteos con h10 = 0:")
    zero = df_r[df_r['h10'] == 0].head(10)
    for _, r in zero.iterrows():
        print(f"  {r['fecha']}: real={r['real']}, prob_std={r['prob_std']:.4f}")

    df_r.to_csv("reports/iter25_diagnostic.csv", index=False)


if __name__ == "__main__":
    main()
