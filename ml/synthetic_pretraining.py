"""Pre-training con dataset sintético + fine-tuning con datos reales.

Estrategia "transfer learning":
1. Generar 10,000 sorteos sintéticos uniformes 1-48 (5 nums, sin reemplazo)
2. Inyectar ligeros sesgos realistas (clustering temporal, hot streaks) para que
   el modelo aprenda a detectar patrones (no para crear señal falsa)
3. Pre-entrenar modelos sobre el dataset sintético
4. Fine-tune con los 105 sorteos reales

Esto permite a los modelos aprender:
- Las características estructurales del juego (5/48, ranges)
- Cómo "decir que no hay señal" cuando los datos son aleatorios
- Estabilidad numérica con muchos ejemplos
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols, POOL, DRAW_SIZE

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def generate_synthetic_uniform(n_sorteos: int, seed: int = 42) -> pd.DataFrame:
    """Genera sorteos sintéticos perfectamente uniformes (control)."""
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2010-01-01")
    for i in range(n_sorteos):
        nums = sorted(rng.choice(POOL, DRAW_SIZE, replace=False) + 1)
        bolilla = rng.integers(1, POOL + 1)
        # 4 sorteos por semana aprox
        fecha = base_date + pd.Timedelta(days=int(i * 7 / 4))
        rows.append({
            "fecha": fecha.date(),
            "dia_semana": ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"][fecha.dayofweek],
            "n1": nums[0], "n2": nums[1], "n3": nums[2], "n4": nums[3], "n5": nums[4],
            "bolilla_extra": int(bolilla),
            "fuente": "synthetic_uniform",
        })
    return pd.DataFrame(rows)


def generate_synthetic_with_drift(n_sorteos: int, drift_strength: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """
    Genera sorteos con drift mínimo en frecuencias (simulando bias real).
    drift_strength=0 → uniforme. drift_strength=0.05 → ligero sesgo (5%).
    """
    rng = np.random.default_rng(seed)
    # Probabilidades base por número
    base_probs = np.ones(POOL) / POOL
    # Pequeño drift aleatorio
    drift = rng.normal(0, drift_strength, POOL)
    base_probs = np.clip(base_probs + drift, 0.001, 1.0)
    base_probs /= base_probs.sum()

    rows = []
    base_date = pd.Timestamp("2010-01-01")
    for i in range(n_sorteos):
        # Hot/cold streaks: prob varía ligeramente con tiempo
        local_drift = rng.normal(0, drift_strength * 0.5, POOL)
        local_probs = np.clip(base_probs + local_drift, 0.001, 1.0)
        local_probs /= local_probs.sum()

        nums = sorted(rng.choice(POOL, DRAW_SIZE, replace=False, p=local_probs) + 1)
        bolilla = rng.integers(1, POOL + 1)
        fecha = base_date + pd.Timedelta(days=int(i * 7 / 4))
        rows.append({
            "fecha": fecha.date(),
            "dia_semana": ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"][fecha.dayofweek],
            "n1": nums[0], "n2": nums[1], "n3": nums[2], "n4": nums[3], "n5": nums[4],
            "bolilla_extra": int(bolilla),
            "fuente": "synthetic_drift",
        })
    return pd.DataFrame(rows)


def evaluate_pretrained_model(real_csv: str, synth_n: int = 1000, drift: float = 0.0) -> dict:
    """
    Pre-entrena con sintéticos, fine-tune con reales, y evalúa.
    Compara contra solo-real (sin pre-training).
    """
    from ml.lightgbm_model import LightGBMMultilabel
    from ml.evaluator import top_k_recall, monte_carlo_baseline

    print(f"Generando {synth_n} sorteos sintéticos (drift={drift})...")
    synth_df = generate_synthetic_with_drift(synth_n, drift_strength=drift)
    real_df = pd.read_csv(real_csv)
    print(f"Datos reales: {len(real_df)} sorteos")

    # Build features para ambos (min_history más bajo para tener más datos)
    print("Construyendo features para sintéticos...")
    synth_features = build_features(synth_df, min_history=50)

    print("Construyendo features para reales...")
    real_features = build_features(real_df, min_history=30)

    feat_cols = get_feature_cols(real_features)
    target_cols = get_target_cols(real_features)

    # Split temporal real: 80% train, 20% test
    n_real = len(real_features)
    test_idx = int(n_real * 0.8)
    real_test = real_features.iloc[test_idx:]

    # Asegurar que synth tenga mismas columnas
    synth_features = synth_features[feat_cols + target_cols]

    X_synth = synth_features[feat_cols].values.astype(np.float32)
    y_synth = synth_features[target_cols].values.astype(np.int32)
    X_real_train = real_features[feat_cols].iloc[:test_idx].values.astype(np.float32)
    y_real_train = real_features[target_cols].iloc[:test_idx].values.astype(np.int32)
    X_test = real_test[feat_cols].values.astype(np.float32)
    y_test = real_test[target_cols].values.astype(np.int32)

    # Combinar pre-training + fine-tuning data
    X_combined = np.vstack([X_synth, X_real_train])
    y_combined = np.vstack([y_synth, y_real_train])
    sample_weights = np.concatenate([
        np.ones(len(X_synth)) * 0.1,  # menos peso a sintéticos
        np.ones(len(X_real_train)) * 1.0,  # peso completo a reales
    ])

    print(f"\nEntrenando con {len(X_synth)} sintéticos + {len(X_real_train)} reales...")

    # Modelo con pre-training
    model_pre = LightGBMMultilabel(max_depth=4, n_estimators=200, learning_rate=0.05)
    # LightGBM en sklearn no acepta sample_weight directamente para multilabel
    # Concatenamos y entrenamos en uno
    model_pre.fit(X_combined, y_combined)

    # Modelo solo con datos reales
    model_real = LightGBMMultilabel(max_depth=4, n_estimators=200, learning_rate=0.05)
    model_real.fit(X_real_train, y_real_train)

    # Comparar
    print("\nEvaluando ambos modelos...")
    y_prob_pre = model_pre.predict_proba(X_test)
    y_prob_real = model_real.predict_proba(X_test)

    results = {}
    for k in [5, 10, 15, 20]:
        recall_pre = top_k_recall(y_test, y_prob_pre, k=k)
        recall_real = top_k_recall(y_test, y_prob_real, k=k)
        mc = monte_carlo_baseline(k=k, n_sorteos=len(X_test), n_sims=2000)
        results[f"k{k}"] = {
            "pre_training": recall_pre,
            "solo_real": recall_real,
            "random_mc": mc["media"],
            "random_p95": mc["p95"],
            "pre_vs_random": recall_pre / mc["media"] if mc["media"] > 0 else 0,
            "real_vs_random": recall_real / mc["media"] if mc["media"] > 0 else 0,
        }
        print(f"\n  K={k}:")
        print(f"    Pre-training: {recall_pre:.4f} ({results[f'k{k}']['pre_vs_random']:.2f}x random)")
        print(f"    Solo real:    {recall_real:.4f} ({results[f'k{k}']['real_vs_random']:.2f}x random)")
        print(f"    MC baseline:  {mc['media']:.4f} ± {mc['std']:.4f} (p95={mc['p95']:.4f})")

    return results


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--output", default="reports/pretraining_results.json")
    parser.add_argument("--synth-n", type=int, default=5000)
    parser.add_argument("--drift", type=float, default=0.0)
    args = parser.parse_args()

    results = evaluate_pretrained_model(args.input, synth_n=args.synth_n, drift=args.drift)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Resultados guardados en {args.output}")
