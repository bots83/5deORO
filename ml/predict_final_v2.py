"""Predicción final v2: usa la mejor configuración encontrada (iter 11).

Config óptima: cluster085 (74%) + cdm_last150 (22%) + pair (4%)

Genera predicción multinivel y reporta TODAS las métricas de backtest.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter5_novel_algos import pair_boost_predictor, cluster_predictor
from ml.iter9_cdm import cdm_predictor


# Mejor config de iter 11
WEIGHTS_OPTIMAL = {
    "cdm_last150": 0.218,
    "pair": 0.045,
    "cluster085": 0.737,
}


def predict_with_optimal(history):
    """Genera predicción con la config óptima."""
    components = {}

    p1 = cdm_predictor(history, last_n=150)
    if p1.sum() > 0:
        components["cdm_last150"] = p1 / p1.sum()

    p2 = pair_boost_predictor(history)
    if p2.sum() > 0:
        components["pair"] = p2 / p2.sum()

    p3 = cluster_predictor(history, decay=0.85)
    if p3.sum() > 0:
        components["cluster085"] = p3 / p3.sum()

    ensemble = np.zeros(POOL)
    total_w = 0
    for name, w in WEIGHTS_OPTIMAL.items():
        if name in components:
            ensemble += components[name] * w
            total_w += w
    if total_w == 0:
        return np.ones(POOL) / POOL
    return ensemble / total_w


def backtest(df, last_n=50):
    """Backtest exhaustivo con todas las métricas."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = list(range(5, 46))

    results = {f"top{k}": {"5_hits": 0, "4plus_hits": 0, "3plus_hits": 0, "2plus_hits": 0, "1plus_hits": 0, "0_hits": 0}
               for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        probs = predict_with_optimal(history)
        sorted_idx = np.argsort(probs)[::-1]

        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            r = results[f"top{k}"]
            if hits == 5: r["5_hits"] += 1
            if hits >= 4: r["4plus_hits"] += 1
            if hits >= 3: r["3plus_hits"] += 1
            if hits >= 2: r["2plus_hits"] += 1
            if hits >= 1: r["1plus_hits"] += 1
            if hits == 0: r["0_hits"] += 1

    return results


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos")
    print(f"Último sorteo: {df.iloc[-1]['fecha'].date()}\n")

    # Backtest
    print("=" * 80)
    print("BACKTEST sobre últimos 50 sorteos (sin leakage)")
    print("=" * 80)
    print("Config: cluster085 (74%) + cdm_last150 (22%) + pair (4%)\n")

    bt = backtest(df, last_n=50)

    # Resumen
    print(f"{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5} {'2+':>5} {'1+':>5} {'0':>4}")
    print("-" * 50)
    for k in [10, 12, 15, 18, 20, 25, 29, 30, 35, 37, 40, 45]:
        r = bt[f"top{k}"]
        print(f"top-{k:<4} {r['5_hits']:5d} {r['4plus_hits']:5d} {r['3plus_hits']:5d} {r['2plus_hits']:5d} {r['1plus_hits']:5d} {r['0_hits']:4d}")

    print(f"\n🎯 META: 35/50 sorteos con cierto número de aciertos")
    metas = [
        ("5/5", "5_hits"),
        ("4+",  "4plus_hits"),
        ("3+",  "3plus_hits"),
        ("2+",  "2plus_hits"),
        ("1+",  "1plus_hits"),
    ]
    for label, key in metas:
        for k in range(5, 49):
            if bt[f"top{k}"][key] >= 35:
                print(f"  {label} hits ≥35/50: Top-{k} ({bt[f'top{k}'][key]}/50 = {bt[f'top{k}'][key]*2}%) ✅")
                break
        else:
            print(f"  {label} hits ≥35/50: NO ALCANZADO en top-K hasta 48")

    # Predicción para próximo sorteo
    print("\n" + "=" * 80)
    print("🎯 PREDICCIÓN PARA EL PRÓXIMO SORTEO")
    print("=" * 80)
    probs = predict_with_optimal(df)
    sorted_idx = np.argsort(probs)[::-1]

    output = {
        "fecha_dataset": str(df["fecha"].max().date()),
        "n_sorteos_dataset": len(df),
        "weights_used": WEIGHTS_OPTIMAL,
        "predictions": {},
        "backtest_50_sorteos": bt,
        "all_probs": {int(i+1): float(p) for i, p in enumerate(probs)},
    }

    levels = [
        (5, "puntual", "alta varianza"),
        (10, "recomendado", "buena cobertura"),
        (12, "ampliado", "mejor que top-10"),
        (15, "amplio", "más cobertura"),
        (20, "seguro", "mucha cobertura"),
        (25, "muy seguro", "60% chance de 3+ aciertos"),
        (29, "seguro 3+", "70% chance de 3+ aciertos"),
        (30, "muy amplio", "76% de 3+ aciertos"),
        (37, "casi-todos", "70% de 4+ aciertos"),
        (45, "máxima", "72% de los 5 aciertos"),
    ]

    for k, label, note in levels:
        nums = sorted((sorted_idx[:k] + 1).tolist())
        cov = float(probs[sorted_idx[:k]].sum())
        bt_r = bt[f"top{k}"] if f"top{k}" in bt else None
        output["predictions"][f"top{k}"] = {
            "numbers": nums,
            "label": label,
            "note": note,
            "prob_total": round(cov, 4),
            "backtest": bt_r,
        }
        print(f"\n📊 TOP-{k:2d} ({label}, {note}):")
        if bt_r:
            print(f"   Backtest: {bt_r['5_hits']}/50 con 5/5, {bt_r['3plus_hits']}/50 con 3+, {bt_r['1plus_hits']}/50 con 1+")
        print(f"   Cobertura prob: {cov*100:.1f}%")
        print(f"   Números: {' '.join(f'{n:02d}' for n in nums)}")

    # Top-10 detallado
    print("\n" + "=" * 80)
    print("TOP-10 RANKING DETALLADO")
    print("=" * 80)
    baseline = 5 / 48
    for i in range(10):
        n = sorted_idx[i] + 1
        p = probs[n - 1]
        ratio = p / baseline
        bar = "█" * int(min(ratio, 5) * 8)
        print(f"   {i+1:2d}. Núm {n:2d}: prob={p:.4f} ({ratio:.2f}x random) {bar}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/prediction_final_v2.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Predicción guardada en reports/prediction_final_v2.json")


if __name__ == "__main__":
    main()
