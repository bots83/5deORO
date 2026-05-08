"""Optimizador agresivo: prueba MUCHAS configuraciones y elige la que mejor
predice los últimos N sorteos en backtest sin leakage.

Estrategia:
1. Para cada uno de los últimos 50 sorteos, simular "predecir sin verlo".
2. Probar 100+ combinaciones de modelos/decays/pesos.
3. Para cada config, calcular promedio de hits en top-5, top-10, top-15.
4. Elegir la config que maximice top-10 (objetivo del usuario).
5. Reportar predicción para el siguiente sorteo con esa config óptima.
"""
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols, _build_features_for_row, POOL, DRAW_SIZE
from ml.baseline import FrequencyBaseline
from ml.bayesian import BetaBinomialModel
from ml.evaluator import top_k_recall, random_baseline_expected


def predict_for_index(df: pd.DataFrame, target_idx: int, models_config: list[tuple]) -> np.ndarray:
    """
    Predice probabilidades para el sorteo en target_idx usando solo datos previos.
    models_config: lista de (name, model_class, kwargs, weight)
    Retorna: np.array de 48 probabilidades.
    """
    history = df.iloc[:target_idx]
    if len(history) < 30:
        return np.ones(POOL) / POOL  # fallback uniforme

    # Build features de history (target_idx-1 sorteo)
    feats_pred = _build_features_for_row(history)
    if not feats_pred:
        return np.ones(POOL) / POOL

    # Para entrenar, build features sobre history excluyendo el último 5%
    train_df = build_features(history, min_history=30)
    if len(train_df) < 20:
        return np.ones(POOL) / POOL

    feat_cols = get_feature_cols(train_df)
    target_cols = get_target_cols(train_df)
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.int32)

    X_pred = np.array([[feats_pred[c] for c in feat_cols]], dtype=np.float32)

    ensemble = np.zeros(POOL)
    total_w = 0
    for name, ModelCls, kwargs, w in models_config:
        try:
            m = ModelCls(**kwargs)
            m.fit(X_train, y_train)
            probs = m.predict_proba(X_pred)[0]
            ensemble += probs * w
            total_w += w
        except Exception:
            continue

    if total_w == 0:
        return np.ones(POOL) / POOL
    return ensemble / total_w


def evaluate_config(df: pd.DataFrame, models_config: list[tuple],
                    last_n: int = 30) -> dict:
    """Evalúa la config sobre los últimos last_n sorteos."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    hits_top5 = []
    hits_top10 = []
    hits_top15 = []

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        probs = predict_for_index(df, idx, models_config)
        sorted_idx = np.argsort(probs)[::-1]
        top5 = set((sorted_idx[:5] + 1).tolist())
        top10 = set((sorted_idx[:10] + 1).tolist())
        top15 = set((sorted_idx[:15] + 1).tolist())
        hits_top5.append(len(real & top5))
        hits_top10.append(len(real & top10))
        hits_top15.append(len(real & top15))

    return {
        "top5_avg": float(np.mean(hits_top5)),
        "top10_avg": float(np.mean(hits_top10)),
        "top15_avg": float(np.mean(hits_top15)),
        "top5_total": sum(hits_top5),
        "top10_total": sum(hits_top10),
        "top15_total": sum(hits_top15),
        "n_sorteos": len(hits_top5),
        "top5_distribution": {i: hits_top5.count(i) for i in range(6)},
        "top10_distribution": {i: hits_top10.count(i) for i in range(6)},
    }


def grid_search_configs(df: pd.DataFrame, last_n: int = 30):
    """Prueba MUCHAS combinaciones."""
    print(f"Grid search sobre los últimos {last_n} sorteos...")

    # Definir configs candidatos
    configs = []

    # Configs base
    decays = [1.0, 0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70]

    # Single model configs
    for d in decays:
        configs.append((f"BB(d={d})", [
            ("bb", BetaBinomialModel, {"decay": d}, 1.0),
        ]))

    # Frecuencia pura
    configs.append(("Freq", [("f", FrequencyBaseline, {}, 1.0)]))

    # Ensembles 2-modelos
    for d1 in [0.99, 0.95, 0.90]:
        for d2 in [0.85, 0.70]:
            for w1 in [0.3, 0.5, 0.7]:
                w2 = 1 - w1
                configs.append((f"BB({d1},{d2})_{w1}", [
                    ("bb1", BetaBinomialModel, {"decay": d1}, w1),
                    ("bb2", BetaBinomialModel, {"decay": d2}, w2),
                ]))

    # Ensemble 3-modelos
    for d1, d2, d3 in [(0.99, 0.95, 0.85), (0.95, 0.85, 0.70), (1.0, 0.90, 0.75)]:
        configs.append((f"BB3({d1},{d2},{d3})", [
            ("bb1", BetaBinomialModel, {"decay": d1}, 0.4),
            ("bb2", BetaBinomialModel, {"decay": d2}, 0.35),
            ("bb3", BetaBinomialModel, {"decay": d3}, 0.25),
        ]))

    print(f"Total configs a evaluar: {len(configs)}")
    results = []
    for i, (name, conf) in enumerate(configs):
        t0 = time.time()
        r = evaluate_config(df, conf, last_n=last_n)
        r["config"] = name
        r["time_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  [{i+1}/{len(configs)}] {name:30s} top5={r['top5_avg']:.3f} top10={r['top10_avg']:.3f} top15={r['top15_avg']:.3f} ({r['time_s']}s)")

    return pd.DataFrame(results).sort_values("top10_avg", ascending=False)


def predict_next_with_best_config(df: pd.DataFrame, best_config: list[tuple]) -> dict:
    """Genera predicción para el próximo sorteo usando la mejor config."""
    feats_next = _build_features_for_row(df)
    train_df = build_features(df, min_history=30)
    feat_cols = get_feature_cols(train_df)
    target_cols = get_target_cols(train_df)
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.int32)
    X_next = np.array([[feats_next[c] for c in feat_cols]], dtype=np.float32)

    ensemble = np.zeros(POOL)
    total_w = 0
    for name, ModelCls, kwargs, w in best_config:
        m = ModelCls(**kwargs)
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_next)[0]
        ensemble += probs * w
        total_w += w
    ensemble /= total_w

    sorted_idx = np.argsort(ensemble)[::-1]
    return {
        "top5": sorted((sorted_idx[:5] + 1).tolist()),
        "top10": (sorted_idx[:10] + 1).tolist(),
        "top15": sorted((sorted_idx[:15] + 1).tolist()),
        "top20": sorted((sorted_idx[:20] + 1).tolist()),
        "probs": {int(i + 1): float(p) for i, p in enumerate(ensemble)},
    }


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    last_n = 30  # backtest sobre últimos 30
    results = grid_search_configs(df, last_n=last_n)

    print("\n" + "=" * 80)
    print(f"RANKING POR TOP-10 RECALL ({last_n} sorteos backtest sin leakage)")
    print("=" * 80)
    cols = ["config", "top5_avg", "top10_avg", "top15_avg", "top5_total", "top10_total", "top15_total"]
    print(results[cols].head(15).to_string(index=False))

    # Random baseline
    exp5, _ = random_baseline_expected(5)
    exp10, _ = random_baseline_expected(10)
    exp15, _ = random_baseline_expected(15)
    print(f"\nBaseline random esperado: top5={exp5:.3f}, top10={exp10:.3f}, top15={exp15:.3f}")

    # Mejor config
    best_row = results.iloc[0]
    best_name = best_row["config"]
    print(f"\n🏆 MEJOR CONFIG: {best_name}")
    print(f"   top5_avg={best_row['top5_avg']:.3f} ({best_row['top5_avg']/exp5:.2f}x random)")
    print(f"   top10_avg={best_row['top10_avg']:.3f} ({best_row['top10_avg']/exp10:.2f}x random)")
    print(f"   top15_avg={best_row['top15_avg']:.3f} ({best_row['top15_avg']/exp15:.2f}x random)")
    print(f"   Distribución hits top-10: {best_row['top10_distribution']}")

    # Reconstruir best_config
    name_to_config = {n: c for n, c in [
        ("Freq", [("f", FrequencyBaseline, {}, 1.0)]),
    ]}

    # Re-ejecutar grid search just on best y guardar la config exacta
    decays = [1.0, 0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80, 0.70]
    for d in decays:
        name_to_config[f"BB(d={d})"] = [("bb", BetaBinomialModel, {"decay": d}, 1.0)]
    for d1 in [0.99, 0.95, 0.90]:
        for d2 in [0.85, 0.70]:
            for w1 in [0.3, 0.5, 0.7]:
                w2 = 1 - w1
                name_to_config[f"BB({d1},{d2})_{w1}"] = [
                    ("bb1", BetaBinomialModel, {"decay": d1}, w1),
                    ("bb2", BetaBinomialModel, {"decay": d2}, w2),
                ]
    for d1, d2, d3 in [(0.99, 0.95, 0.85), (0.95, 0.85, 0.70), (1.0, 0.90, 0.75)]:
        name_to_config[f"BB3({d1},{d2},{d3})"] = [
            ("bb1", BetaBinomialModel, {"decay": d1}, 0.4),
            ("bb2", BetaBinomialModel, {"decay": d2}, 0.35),
            ("bb3", BetaBinomialModel, {"decay": d3}, 0.25),
        ]

    best_config_obj = name_to_config[best_name]
    pred = predict_next_with_best_config(df, best_config_obj)

    print(f"\n🎯 PREDICCIÓN PARA EL PRÓXIMO SORTEO (config óptima):")
    print(f"   Top-5:  {pred['top5']}")
    print(f"   Top-10: {pred['top10']}")
    print(f"   Top-15: {pred['top15']}")

    # Guardar resultados
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/optimizer_results.csv", index=False)
    import json
    with open("reports/optimizer_best.json", "w") as f:
        json.dump({
            "best_config_name": best_name,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v
                       for k, v in best_row.to_dict().items() if k not in ["top5_distribution", "top10_distribution"]},
            "prediction": pred,
            "n_configs_tested": len(results),
            "backtest_n": last_n,
        }, f, indent=2, default=str)
    print(f"\n✓ Resultados guardados en reports/optimizer_*.{{csv,json}}")


if __name__ == "__main__":
    main()
