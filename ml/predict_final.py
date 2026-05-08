"""Predicción final con ensemble calibrado por backtest.

Usa los pesos calibrados (de ml/backtest.py) para generar la predicción final.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols, _build_features_for_row, POOL, DRAW_SIZE
from ml.baseline import FrequencyBaseline
from ml.bayesian import BetaBinomialModel
from ml.lightgbm_model import LightGBMMultilabel
from ml.xgboost_model import XGBoostMultilabel


FACTORIES = {
    "Frecuencia": lambda: FrequencyBaseline(),
    "BetaBinomial decay=1.0": lambda: BetaBinomialModel(decay=1.0),
    "BetaBinomial decay=0.99": lambda: BetaBinomialModel(decay=0.99),
    "BetaBinomial decay=0.95": lambda: BetaBinomialModel(decay=0.95),
    "BetaBinomial decay=0.90": lambda: BetaBinomialModel(decay=0.90),
    "LightGBM": lambda: LightGBMMultilabel(max_depth=3, n_estimators=50, learning_rate=0.1),
    "XGBoost": lambda: XGBoostMultilabel(max_depth=3, n_estimators=50, learning_rate=0.1),
}


def predict_final(sorteos_csv: str, weights_json: str = None) -> dict:
    df = pd.read_csv(sorteos_csv)
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    print(f"\n{'='*70}")
    print(f"PREDICCIÓN FINAL CALIBRADA — 5 DE ORO (5 números del 1-48)")
    print(f"{'='*70}")
    print(f"\nDataset: {len(df)} sorteos")
    print(f"Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"Último sorteo: {df.iloc[-1]['fecha'].date()} → "
          f"{[df.iloc[-1][f'n{i}'] for i in range(1,6)]} + Bolilla {df.iloc[-1]['bolilla_extra']}")

    # Cargar pesos
    if weights_json and Path(weights_json).exists():
        with open(weights_json) as f:
            weights = json.load(f)
        print(f"\nUsando pesos calibrados de {weights_json}")
    else:
        weights = {n: 1.0 / len(FACTORIES) for n in FACTORIES}

    # Construir features
    features_df = build_features(df, min_history=30)
    feat_cols = get_feature_cols(features_df)
    target_cols = get_target_cols(features_df)
    X_train = features_df[feat_cols].values.astype(np.float32)
    y_train = features_df[target_cols].values.astype(np.int32)

    # Features para próximo sorteo
    feats_next = _build_features_for_row(df)
    X_next = np.array([[feats_next[c] for c in feat_cols]], dtype=np.float32)

    # Predecir con cada modelo
    print(f"\nEntrenando {len(FACTORIES)} modelos sobre {len(X_train)} sorteos con {len(feat_cols)} features...")
    all_probs = {}
    for name, factory in FACTORIES.items():
        try:
            model = factory()
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_next)[0]
            all_probs[name] = probs
            top5 = sorted((np.argsort(probs)[::-1][:5] + 1).tolist())
            w = weights.get(name, 0)
            print(f"  [{name:30s} w={w:.3f}] top-5: {top5}")
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")

    if not all_probs:
        return {}

    # Ensemble ponderado
    total_w = sum(weights.get(n, 0) for n in all_probs.keys())
    if total_w == 0:
        total_w = sum(1 for _ in all_probs)
        weights = {n: 1.0 / total_w for n in all_probs}
        total_w = 1.0

    ensemble_probs = np.zeros(POOL)
    for name, probs in all_probs.items():
        w = weights.get(name, 0) / total_w
        ensemble_probs += probs * w

    sorted_idx = np.argsort(ensemble_probs)[::-1]
    top5 = sorted((sorted_idx[:5] + 1).tolist())
    top10 = (sorted_idx[:10] + 1).tolist()
    top15 = (sorted_idx[:15] + 1).tolist()

    baseline_prob = DRAW_SIZE / POOL  # 5/48 = 0.1042

    # Reporte
    print(f"\n{'='*70}")
    print(f"🎯 PREDICCIÓN FINAL")
    print(f"{'='*70}")
    print(f"\n>>> LOS 5 NÚMEROS PARA JUGAR EN EL PRÓXIMO SORTEO:")
    print(f"\n      {' - '.join(f'{n:02d}' for n in top5)}")
    print(f"\n📊 TOP-10 RANKING (ordenado por probabilidad):")
    for i, num in enumerate(top10, 1):
        prob = ensemble_probs[num - 1]
        ratio = prob / baseline_prob
        bar = "█" * int(min(ratio, 5) * 10)
        marker = "🎯" if num in top5 else "  "
        print(f"   {marker} {i:2d}. Núm {num:2d}: prob={prob:.4f} ({ratio:.2f}x random) {bar}")

    print(f"\n📊 TOP-15 (margen ampliado):")
    print(f"   {sorted(top15)}")

    print(f"\n📈 ANÁLISIS:")
    print(f"   Probabilidad uniforme aleatoria: {baseline_prob:.4f} ({baseline_prob*100:.2f}%)")
    print(f"   Mejor número: prob={ensemble_probs.max():.4f} ({ensemble_probs.max()/baseline_prob:.2f}x)")
    print(f"   Diferencia max-min: {ensemble_probs.max() - ensemble_probs.min():.4f}")
    print(f"   ⚠ IMPORTANTE: Los tests estadísticos rigurosos confirman que")
    print(f"      el sorteo es aleatorio uniforme. Estos números reflejan")
    print(f"      patrones recientes que NO son estadísticamente significativos.")

    return {
        "fecha_dataset": str(df["fecha"].max().date()),
        "n_sorteos_dataset": len(df),
        "top5": top5,
        "top10": top10,
        "top15": sorted(top15),
        "ensemble_probs": {f"num_{i+1}": float(p) for i, p in enumerate(ensemble_probs)},
        "models_predictions": {n: sorted((np.argsort(p)[::-1][:5] + 1).tolist()) for n, p in all_probs.items()},
        "weights_used": weights,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--weights", default="reports/ensemble_weights.json")
    parser.add_argument("--output", default="reports/prediction_final.json")
    args = parser.parse_args()

    result = predict_final(args.input, args.weights)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Predicción guardada en {args.output}")
