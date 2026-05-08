"""Optimizador iterativo: prueba múltiples estrategias hasta superar threshold.

Métrica objetivo: # de sorteos con 3+ aciertos en top-10, sobre últimos 50.
"""
import sys
import time
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols, _build_features_for_row, POOL, DRAW_SIZE
from ml.bayesian import BetaBinomialModel
from ml.baseline import FrequencyBaseline


def predict_for_idx_cached(df, target_idx, train_features_cache, models_config):
    """Predice para target_idx usando train_features_cache pre-computado."""
    history = df.iloc[:target_idx]
    if len(history) < 30:
        return np.ones(POOL) / POOL

    feats_pred = _build_features_for_row(history)
    if not feats_pred:
        return np.ones(POOL) / POOL

    train_df = train_features_cache.get(target_idx)
    if train_df is None or len(train_df) < 20:
        return np.ones(POOL) / POOL

    feat_cols = [c for c in train_df.columns if not c.startswith("target_")]
    target_cols = [c for c in train_df.columns if c.startswith("target_")]
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


def precompute_train_features(df: pd.DataFrame, last_n: int):
    """Pre-computa features de training para cada idx del backtest (cacheable)."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    cache = {}
    print(f"  Precomputing {n_total - start} train feature sets...", end=" ", flush=True)
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = build_features(history, min_history=30)
        cache[idx] = train_df
    print("done")
    return cache


def evaluate_config_fast(df, train_cache, models_config, last_n=50):
    """Evalúa rápido con cache."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    hits_top5 = []
    hits_top10 = []
    hits_top15 = []

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        probs = predict_for_idx_cached(df, idx, train_cache, models_config)
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
        "n_3plus_top10": sum(1 for h in hits_top10 if h >= 3),
        "n_3plus_top15": sum(1 for h in hits_top15 if h >= 3),
        "n_2plus_top10": sum(1 for h in hits_top10 if h >= 2),
        "n_zero_top10": sum(1 for h in hits_top10 if h == 0),
        "hits_top5_list": hits_top5,
        "hits_top10_list": hits_top10,
        "hits_top15_list": hits_top15,
    }


def gen_configs():
    """Genera 100+ configs candidatas."""
    configs = []
    # 1. Beta binomial single
    for d in [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]:
        configs.append((f"BB(d={d})", [("bb", BetaBinomialModel, {"decay": d}, 1.0)]))

    # 2. Frecuencia + variantes
    configs.append(("Freq", [("f", FrequencyBaseline, {}, 1.0)]))

    # 3. Ensembles 2-modelos sobre amplio rango de pesos
    decays_2 = [(0.99, 0.70), (0.99, 0.50), (0.99, 0.85), (0.95, 0.70), (0.95, 0.50),
                (0.97, 0.80), (0.95, 0.85), (0.90, 0.70), (1.0, 0.70), (1.0, 0.85),
                (0.92, 0.65), (0.85, 0.55), (0.97, 0.55), (0.99, 0.60), (0.95, 0.55)]
    for d1, d2 in decays_2:
        for w1 in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            w2 = 1 - w1
            configs.append((f"BB({d1}_{d2})_w{w1}", [
                ("bb1", BetaBinomialModel, {"decay": d1}, w1),
                ("bb2", BetaBinomialModel, {"decay": d2}, w2),
            ]))

    # 4. Ensembles 3-modelos
    for d1, d2, d3 in [(0.99, 0.95, 0.85), (0.99, 0.90, 0.70), (0.97, 0.85, 0.70),
                       (1.0, 0.95, 0.85), (0.99, 0.95, 0.70), (1.0, 0.85, 0.65),
                       (0.95, 0.85, 0.65), (1.0, 0.90, 0.60), (0.95, 0.80, 0.55),
                       (0.99, 0.80, 0.50)]:
        for w1, w2, w3 in [(0.5, 0.3, 0.2), (0.4, 0.3, 0.3), (0.6, 0.25, 0.15),
                           (0.33, 0.33, 0.33), (0.5, 0.25, 0.25), (0.7, 0.2, 0.1)]:
            configs.append((f"BB3({d1}_{d2}_{d3})_w{w1}_{w2}", [
                ("bb1", BetaBinomialModel, {"decay": d1}, w1),
                ("bb2", BetaBinomialModel, {"decay": d2}, w2),
                ("bb3", BetaBinomialModel, {"decay": d3}, w3),
            ]))

    # 5. Frecuencia + BB
    for d in [0.99, 0.95, 0.85, 0.70]:
        for wf in [0.3, 0.5, 0.7]:
            configs.append((f"Freq+BB({d})_wf{wf}", [
                ("f", FrequencyBaseline, {}, wf),
                ("bb", BetaBinomialModel, {"decay": d}, 1 - wf),
            ]))

    return configs


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos")
    print(f"Backtest sobre últimos 50 sorteos\n")

    # Pre-compute train features (esto era el cuello de botella)
    train_cache = precompute_train_features(df, last_n=50)

    configs = gen_configs()
    print(f"Total configs a evaluar: {len(configs)}\n")

    best = None
    results = []
    for i, (name, conf) in enumerate(configs):
        t0 = time.time()
        try:
            r = evaluate_config_fast(df, train_cache, conf, last_n=50)
            r["config"] = name
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)
            marker = ""
            if best is None or r["n_3plus_top10"] > best["n_3plus_top10"]:
                best = {**r, "config_obj": conf}
                marker = " ⭐"
            if i % 20 == 0 or marker:
                print(f"  [{i+1}/{len(configs)}] {name:35s} top10_3+={r['n_3plus_top10']:2d}/50 top15_3+={r['n_3plus_top15']:2d}/50 top5_avg={r['top5_avg']:.3f}{marker}")
        except Exception as e:
            print(f"  [{i+1}] {name}: error {e}")

    # Ranking
    df_res = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)}
                           for r in results])
    df_res = df_res.sort_values("n_3plus_top10", ascending=False)

    print("\n" + "=" * 90)
    print(f"TOP 20 CONFIGS POR # DE SORTEOS CON 3+ HITS EN TOP-10 (de 50 sorteos)")
    print("=" * 90)
    cols = ["config", "n_3plus_top10", "n_3plus_top15", "n_2plus_top10",
            "n_zero_top10", "top5_avg", "top10_avg", "top15_avg"]
    print(df_res[cols].head(20).to_string(index=False))

    print(f"\n🏆 MEJOR CONFIG: {best['config']}")
    print(f"   Sorteos con 3+ aciertos en top-10: {best['n_3plus_top10']}/50 ({best['n_3plus_top10']*2}%)")
    print(f"   Sorteos con 3+ aciertos en top-15: {best['n_3plus_top15']}/50 ({best['n_3plus_top15']*2}%)")
    print(f"   Sorteos con 2+ aciertos en top-10: {best['n_2plus_top10']}/50 ({best['n_2plus_top10']*2}%)")
    print(f"   Sorteos con 0 aciertos en top-10:  {best['n_zero_top10']}/50 ({best['n_zero_top10']*2}%)")
    print(f"   top5_avg={best['top5_avg']:.3f} (random {0.521:.3f}, ratio {best['top5_avg']/0.521:.2f}x)")
    print(f"   top10_avg={best['top10_avg']:.3f} (random {1.042:.3f}, ratio {best['top10_avg']/1.042:.2f}x)")

    # Guardar
    Path("reports").mkdir(exist_ok=True)
    df_res.drop(columns=[c for c in ["hits_top5_list", "hits_top10_list", "hits_top15_list"] if c in df_res.columns], errors="ignore").to_csv("reports/iter1_grid_search.csv", index=False)

    with open("reports/iter1_best.json", "w") as f:
        out = {k: v for k, v in best.items() if not isinstance(v, list) and k != "config_obj"}
        out["meta_objetivo"] = {"n_3plus_top10_meta": 35, "logrado": best["n_3plus_top10"], "gap": 35 - best["n_3plus_top10"]}
        json.dump(out, f, indent=2, default=str)

    print(f"\n✓ Resultados guardados en reports/iter1_*.csv y .json")
    print(f"\n🎯 META: 35/50 sorteos con 3+ hits en top-10")
    print(f"📊 LOGRADO: {best['n_3plus_top10']}/50")
    print(f"📉 GAP: {35 - best['n_3plus_top10']} sorteos")


if __name__ == "__main__":
    main()
