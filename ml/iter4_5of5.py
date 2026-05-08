"""Iter 4: Maximizar SORTEOS con 5/5 hits (todos los números reales en top-K).

Métricas:
- 5_in_top_K: # sorteos donde los 5 números están dentro del top-K
- Buscar el K mínimo donde llegamos a 35/50

Probamos top-K en rangos amplios: 10, 15, 20, 25, 30, 35, 40
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL
from ml.bayesian import BetaBinomialModel
from ml.baseline import FrequencyBaseline


def predict_idx(df, target_idx, train_cache, models_config):
    history = df.iloc[:target_idx]
    if len(history) < 30:
        return np.ones(POOL) / POOL

    feats_pred = _build_features_for_row(history)
    if not feats_pred:
        return np.ones(POOL) / POOL

    train_df = train_cache.get(target_idx)
    if train_df is None or len(train_df) < 20:
        return np.ones(POOL) / POOL

    feat_cols = [c for c in train_df.columns if not c.startswith("target_")]
    target_cols = [c for c in train_df.columns if c.startswith("target_")]
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.int32)
    X_pred = np.array([[feats_pred.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

    ensemble = np.zeros(POOL)
    total_w = 0
    for entry in models_config:
        if len(entry) == 4:
            name, ModelCls, kwargs, w = entry
            try:
                m = ModelCls(**kwargs)
                m.fit(X_train, y_train)
                probs = m.predict_proba(X_pred)[0]
                ensemble += probs * w
                total_w += w
            except Exception:
                pass
        elif len(entry) == 3:
            name, fn, w = entry
            try:
                probs = fn(feats_pred)
                ensemble += probs * w
                total_w += w
            except Exception:
                pass
    if total_w == 0:
        return np.ones(POOL) / POOL
    return ensemble / total_w


def precompute(df, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    cache = {}
    print(f"  Precomputing {n_total - start} train sets...", flush=True)
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = build_features(history, min_history=30)
        cache[idx] = train_df
    print("  done", flush=True)
    return cache


def evaluate_5of5(df, train_cache, models_config, last_n=50):
    """Mide cuántos sorteos tienen los 5 números reales dentro de top-K (varios K)."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45]
    n_complete = {k: 0 for k in top_ks}  # con 5/5 hits
    hits_distribution = {k: [] for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        probs = predict_idx(df, idx, train_cache, models_config)
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            hits_distribution[k].append(hits)
            if hits == 5:
                n_complete[k] += 1

    result = {f"top{k}_5of5": n_complete[k] for k in top_ks}
    for k in top_ks:
        h = hits_distribution[k]
        result[f"top{k}_avg_hits"] = float(np.mean(h))
        result[f"top{k}_4plus"] = sum(1 for x in h if x >= 4)
        result[f"top{k}_3plus"] = sum(1 for x in h if x >= 3)
    return result


def gen_configs():
    configs = []
    # Best from iter 1
    configs.append(("BB3_winner", [
        ("bb1", BetaBinomialModel, {"decay": 0.99}, 0.7),
        ("bb2", BetaBinomialModel, {"decay": 0.95}, 0.2),
        ("bb3", BetaBinomialModel, {"decay": 0.70}, 0.1),
    ]))
    configs.append(("BB(0.99,0.7)_w0.7", [
        ("bb1", BetaBinomialModel, {"decay": 0.99}, 0.7),
        ("bb2", BetaBinomialModel, {"decay": 0.70}, 0.3),
    ]))
    configs.append(("Freq_only", [("f", FrequencyBaseline, {}, 1.0)]))

    # Variations
    for d1, d2 in [(0.99, 0.7), (0.99, 0.85), (0.95, 0.7), (1.0, 0.85)]:
        for w1 in [0.3, 0.5, 0.7]:
            configs.append((f"BB({d1},{d2})_w{w1}", [
                ("bb1", BetaBinomialModel, {"decay": d1}, w1),
                ("bb2", BetaBinomialModel, {"decay": d2}, 1 - w1),
            ]))

    return configs


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Pre-computing features...")
    cache = precompute(df, last_n=50)

    configs = gen_configs()
    print(f"\nTotal configs: {len(configs)}\n")
    print("META: 35/50 sorteos con LOS 5 NÚMEROS dentro del top-K\n")

    results = []
    bests_by_k = {}
    for i, (name, conf) in enumerate(configs):
        try:
            t0 = time.time()
            r = evaluate_5of5(df, cache, conf, last_n=50)
            r["config"] = name
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)

            # Encontrar el mínimo K donde alcanza 35/50
            min_k_for_35 = None
            for k in [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]:
                if r[f"top{k}_5of5"] >= 35:
                    min_k_for_35 = k
                    break
            r["min_k_for_35_50"] = min_k_for_35

            print(f"  [{i+1}/{len(configs)}] {name:30s} t10_5/5={r['top10_5of5']:2d} t15={r['top15_5of5']:2d} t20={r['top20_5of5']:2d} t25={r['top25_5of5']:2d} t30={r['top30_5of5']:2d} t35={r['top35_5of5']:2d} t40={r['top40_5of5']:2d} t45={r['top45_5of5']:2d} (min_k_35: {min_k_for_35})")
        except Exception as e:
            print(f"  [{i+1}] {name}: ERROR {e}")

    # Mejor config para cada K
    print("\n" + "=" * 100)
    print("MEJOR CONFIG POR TOP-K (objetivo: 35/50 con 5/5 hits)")
    print("=" * 100)
    for k in [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]:
        key = f"top{k}_5of5"
        ranked = sorted(results, key=lambda r: -r[key])
        best = ranked[0]
        check = "✅" if best[key] >= 35 else "❌"
        print(f"  Top-{k:2d}: {best['config']:30s} → {best[key]:2d}/50 {check} (vs random esperado: depende del K)")

    # Guardar
    df_res = pd.DataFrame(results)
    df_res.to_csv("reports/iter4_5of5.csv", index=False)
    with open("reports/iter4_summary.json", "w") as f:
        summary = {}
        for k in [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]:
            key = f"top{k}_5of5"
            ranked = sorted(results, key=lambda r: -r[key])
            best = ranked[0]
            summary[f"top{k}_best"] = {"config": best["config"], "5of5": best[key]}
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
