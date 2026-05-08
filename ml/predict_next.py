"""Genera predicción de los 5 números más probables para el próximo sorteo.

Combina múltiples modelos (ensemble) y reporta:
- Top 5 números más probables (la "predicción" requerida)
- Top 10 números (margen de seguridad)
- Probabilidad estimada por cada número
- Comparación con el baseline aleatorio uniforme
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols, NUM_COLS, POOL, DRAW_SIZE, _build_features_for_row
from ml.dataset import temporal_split
from ml.baseline import FrequencyBaseline
from ml.bayesian import BetaBinomialModel
from ml.xgboost_model import XGBoostMultilabel
from ml.random_forest import RandomForestMultilabel
from ml.lightgbm_model import LightGBMMultilabel


def make_models() -> list:
    return [
        ("Frecuencia histórica", FrequencyBaseline(), 1.0),
        ("BetaBinomial decay 0.99", BetaBinomialModel(decay=0.99), 1.5),
        ("BetaBinomial recencia 0.95", BetaBinomialModel(decay=0.95), 2.0),
        ("BetaBinomial recencia 0.90", BetaBinomialModel(decay=0.90), 1.5),
        ("LightGBM", LightGBMMultilabel(max_depth=4, n_estimators=100, learning_rate=0.05), 2.0),
        ("XGBoost", XGBoostMultilabel(max_depth=3, n_estimators=100, learning_rate=0.05), 2.0),
    ]


def predict_next_draw(sorteos_csv: str, top_k: int = 10) -> dict:
    """Predice los próximos números entrenando con TODO el histórico."""
    df = pd.read_csv(sorteos_csv)
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    print(f"\n{'='*70}")
    print(f"PREDICCIÓN PARA EL PRÓXIMO SORTEO DE 5 DE ORO")
    print(f"{'='*70}")
    print(f"Dataset: {len(df)} sorteos ({df['fecha'].min().date()} → {df['fecha'].max().date()})")

    # Features históricas para entrenar
    features_df = build_features(df, min_history=30)
    print(f"Features: {features_df.shape[0]} sorteos × {len(get_feature_cols(features_df))} features")

    feat_cols = get_feature_cols(features_df)
    target_cols = get_target_cols(features_df)
    X_train = features_df[feat_cols].values.astype(np.float32)
    y_train = features_df[target_cols].values.astype(np.int32)

    # Features para el "siguiente" sorteo (usando todos los datos disponibles)
    feats_next = _build_features_for_row(df)
    X_next = np.array([[feats_next[c] for c in feat_cols]], dtype=np.float32)

    # Entrenar y predecir con cada modelo
    models = make_models()
    all_probs = {}
    weights = {}
    print(f"\nEntrenando {len(models)} modelos sobre {len(X_train)} sorteos...")
    for name, model, weight in models:
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_next)[0]
            all_probs[name] = probs
            weights[name] = weight
            top5 = np.argsort(probs)[::-1][:5] + 1
            print(f"  [{name}] top-5: {sorted(top5.tolist())}")
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")

    if not all_probs:
        print("Ningún modelo logró predecir.")
        return {}

    # Ensemble: promedio ponderado de probabilidades
    total_weight = sum(weights[k] for k in all_probs.keys())
    ensemble_probs = np.zeros(POOL)
    for name, probs in all_probs.items():
        ensemble_probs += probs * (weights[name] / total_weight)

    # Top-K
    sorted_idx = np.argsort(ensemble_probs)[::-1]
    top5_nums = (sorted_idx[:5] + 1).tolist()
    top10_nums = (sorted_idx[:top_k] + 1).tolist()

    # Baseline aleatorio
    baseline_prob = DRAW_SIZE / POOL

    # Resultado
    print(f"\n{'='*70}")
    print(f"PREDICCIÓN FINAL (ensemble de {len(all_probs)} modelos)")
    print(f"{'='*70}")
    print(f"\n🎯 LOS 5 NÚMEROS MÁS PROBABLES:")
    print(f"   {sorted(top5_nums)}")
    print(f"\n📊 TOP-{top_k} NÚMEROS (margen de confianza):")
    for i, num in enumerate(top10_nums, 1):
        prob = ensemble_probs[num - 1]
        ratio = prob / baseline_prob
        bar = "█" * int(ratio * 10)
        print(f"   {i:2d}. Número {num:2d}: prob={prob:.4f} ({ratio:+.2f}x random) {bar}")

    # Comparación con baseline aleatorio
    print(f"\n📈 ANÁLISIS:")
    print(f"   Probabilidad aleatoria uniforme: {baseline_prob:.4f} ({baseline_prob*100:.2f}%)")
    print(f"   Mejor número predicho: prob={ensemble_probs.max():.4f} ({ensemble_probs.max()/baseline_prob:.2f}x random)")
    print(f"   Peor número predicho:  prob={ensemble_probs.min():.4f} ({ensemble_probs.min()/baseline_prob:.2f}x random)")

    # Guardar reporte
    return {
        "top5": sorted(top5_nums),
        "top10": top10_nums,
        "ensemble_probs": ensemble_probs.tolist(),
        "models_top5": {name: sorted((np.argsort(p)[::-1][:5] + 1).tolist()) for name, p in all_probs.items()},
        "fecha_dataset_max": str(df["fecha"].max().date()),
        "n_sorteos_train": len(features_df),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--output", default="reports/prediction_next.json")
    args = parser.parse_args()

    result = predict_next_draw(args.input)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Predicción guardada en {args.output}")
