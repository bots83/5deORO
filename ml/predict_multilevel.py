"""Predicción multinivel final.

Genera 5 niveles de predicción para el próximo sorteo:
- Top-5: apuesta puntual (alta varianza)
- Top-10: cobertura estándar (recomendada)
- Top-15: cobertura amplia
- Top-25: cobertura segura
- Top-35: muy segura (alto recall, baja precisión)

Usa la mejor configuración encontrada (de iter 10).
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL
from ml.iter10_meta_blend import get_all_components, predict_combined, precompute, precompute_components


def load_best_weights():
    try:
        with open("reports/iter10_best.json") as f:
            data = json.load(f)
            return data.get("weights", {})
    except Exception:
        # Default weights basado en lo que sabemos
        return {
            "cdm_0.99": 0.20,
            "cdm_0.95": 0.15,
            "cdm_0.85": 0.10,
            "bb_0.99": 0.15,
            "bb_0.7": 0.10,
            "pair": 0.10,
            "cluster095": 0.05,
            "streak095": 0.05,
            "adaptive": 0.05,
            "rank": 0.05,
        }


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos")
    print(f"Último sorteo: {df.iloc[-1]['fecha'].date()} → {[df.iloc[-1][f'n{i}'] for i in range(1,6)]} + B{df.iloc[-1]['bolilla_extra']}")

    weights = load_best_weights()
    print(f"\nUsando pesos óptimos ({len(weights)} componentes activos)")

    # Generar predicción para el siguiente sorteo
    train_df = build_features(df, min_history=30)
    components = get_all_components(df, train_df)
    print(f"Componentes generados: {len(components)}")
    probs = predict_combined(components, weights)

    sorted_idx = np.argsort(probs)[::-1]

    print("\n" + "=" * 80)
    print("🎯 PREDICCIÓN MULTINIVEL PARA EL PRÓXIMO SORTEO")
    print("=" * 80)

    # Top niveles
    levels = [(5, "puntual"), (10, "recomendado"), (15, "amplio"),
              (20, "seguro"), (25, "muy seguro"), (30, "máxima cobertura")]

    output = {
        "fecha_dataset": str(df["fecha"].max().date()),
        "n_sorteos_dataset": len(df),
        "predictions": {},
        "weights_used": weights,
        "all_probs": {int(i+1): float(p) for i, p in enumerate(probs)},
    }

    for k, label in levels:
        nums = sorted((sorted_idx[:k] + 1).tolist())
        prob_sum = float(probs[sorted_idx[:k]].sum())
        output["predictions"][f"top{k}"] = {
            "numbers": nums,
            "label": label,
            "prob_total": round(prob_sum, 4),
        }
        print(f"\n📊 TOP-{k:2d} ({label}): coverage de {prob_sum*100:.1f}%")
        nums_str = " ".join(f"{n:02d}" for n in nums)
        print(f"   {nums_str}")

    # Top 10 con probabilidades individuales
    print("\n" + "=" * 80)
    print("TOP-10 RANKING DETALLADO")
    print("=" * 80)
    baseline_prob = 5 / 48
    for i in range(10):
        n = sorted_idx[i] + 1
        p = probs[n-1]
        ratio = p / baseline_prob
        bar = "█" * int(ratio * 5)
        print(f"   {i+1:2d}. Núm {n:2d}: prob={p:.4f} ({ratio:.2f}x random) {bar}")

    # Guardar
    Path("reports").mkdir(exist_ok=True)
    with open("reports/prediction_multilevel.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Predicción guardada en reports/prediction_multilevel.json")


if __name__ == "__main__":
    main()
