"""Iter 2: Features avanzadas + ML supervisado (LightGBM/XGB).

Objetivo: 35/50 sorteos con 3+ aciertos en top-K (probando K=10,15,20,25,30).
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.advanced import build_advanced_features, _build_advanced_features, POOL, NUM_COLS
from features.builder import build_features as build_basic_features
from ml.bayesian import BetaBinomialModel
from ml.lightgbm_model import LightGBMMultilabel


def predict_idx_advanced(df, target_idx, train_cache, models_config):
    history = df.iloc[:target_idx]
    if len(history) < 30:
        return np.ones(POOL) / POOL

    feats_pred = _build_advanced_features(history)
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
    for name, ModelCls, kwargs, w in models_config:
        try:
            m = ModelCls(**kwargs)
            m.fit(X_train, y_train)
            probs = m.predict_proba(X_pred)[0]
            ensemble += probs * w
            total_w += w
        except Exception as e:
            pass
    if total_w == 0:
        return np.ones(POOL) / POOL
    return ensemble / total_w


def precompute_advanced(df, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    cache = {}
    print(f"  Precomputing {n_total - start} advanced feature sets...", flush=True)
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = build_advanced_features(history, min_history=50)
        cache[idx] = train_df
        if (idx - start) % 10 == 0:
            print(f"    {idx-start}/{n_total-start}...", flush=True)
    print("  done")
    return cache


def evaluate_full(df, train_cache, models_config, last_n=50, top_ks=(10, 15, 20, 25, 30)):
    n_total = len(df)
    start = max(60, n_total - last_n)
    hits = {k: [] for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        probs = predict_idx_advanced(df, idx, train_cache, models_config)
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits[k].append(len(real & top_set))

    result = {}
    for k in top_ks:
        h = hits[k]
        result[f"top{k}_avg"] = float(np.mean(h))
        result[f"top{k}_3plus"] = sum(1 for x in h if x >= 3)
        result[f"top{k}_2plus"] = sum(1 for x in h if x >= 2)
        result[f"top{k}_zero"] = sum(1 for x in h if x == 0)
    result["hits_top10"] = hits[10]
    result["hits_top15"] = hits[15]
    result["hits_top20"] = hits[20]
    return result


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Pre-computing advanced features...")
    cache = precompute_advanced(df, last_n=50)

    # Probar varias configuraciones con features avanzadas
    configs = []

    # BB ensemble (lo que ya sabemos que funciona)
    configs.append(("BB_3model_advanced", [
        ("bb1", BetaBinomialModel, {"decay": 0.99}, 0.7),
        ("bb2", BetaBinomialModel, {"decay": 0.95}, 0.2),
        ("bb3", BetaBinomialModel, {"decay": 0.70}, 0.1),
    ]))

    # LightGBM puro
    configs.append(("LGBM_default", [
        ("lgbm", LightGBMMultilabel, {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05}, 1.0),
    ]))
    configs.append(("LGBM_deep", [
        ("lgbm", LightGBMMultilabel, {"max_depth": 6, "n_estimators": 200, "learning_rate": 0.03}, 1.0),
    ]))
    configs.append(("LGBM_shallow", [
        ("lgbm", LightGBMMultilabel, {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1}, 1.0),
    ]))

    # Ensemble BB + LGBM
    for w_bb in [0.3, 0.5, 0.7]:
        configs.append((f"BB+LGBM_w{w_bb}", [
            ("bb1", BetaBinomialModel, {"decay": 0.99}, w_bb * 0.7),
            ("bb2", BetaBinomialModel, {"decay": 0.70}, w_bb * 0.3),
            ("lgbm", LightGBMMultilabel, {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05}, 1 - w_bb),
        ]))

    # Solo BB con features avanzadas
    for d in [0.99, 0.95, 0.90, 0.85, 0.70]:
        configs.append((f"BB_adv_d{d}", [
            ("bb", BetaBinomialModel, {"decay": d}, 1.0),
        ]))

    print(f"Total configs: {len(configs)}\n")

    results = []
    best_top10 = None
    best_top15 = None
    best_top20 = None
    best_top25 = None
    best_top30 = None

    for i, (name, conf) in enumerate(configs):
        t0 = time.time()
        try:
            r = evaluate_full(df, cache, conf, last_n=50)
            r["config"] = name
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)

            updates = []
            for k, var_name, var in [(10, "top10", "best_top10"), (15, "top15", "best_top15"),
                                      (20, "top20", "best_top20"), (25, "top25", "best_top25"),
                                      (30, "top30", "best_top30")]:
                key = f"top{k}_3plus"
                cur_best = locals()[var]
                if cur_best is None or r[key] > cur_best[key]:
                    if var_name == "top10": best_top10 = {**r, "config_obj": conf}
                    elif var_name == "top15": best_top15 = {**r, "config_obj": conf}
                    elif var_name == "top20": best_top20 = {**r, "config_obj": conf}
                    elif var_name == "top25": best_top25 = {**r, "config_obj": conf}
                    elif var_name == "top30": best_top30 = {**r, "config_obj": conf}
                    updates.append(f"top{k}!")

            mark = " ⭐ " + ", ".join(updates) if updates else ""
            print(f"  [{i+1}/{len(configs)}] {name:35s} top10_3+={r['top10_3plus']:2d} top15_3+={r['top15_3plus']:2d} top20_3+={r['top20_3plus']:2d} top25_3+={r['top25_3plus']:2d} top30_3+={r['top30_3plus']:2d} ({r['time_s']}s){mark}")
        except Exception as e:
            print(f"  [{i+1}] {name}: ERROR {e}")

    # Ranking
    print("\n" + "=" * 100)
    print("RANKING TOP-K_3+ POR SORTEOS (de 50)")
    print("=" * 100)

    df_res = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in results])
    df_res.to_csv("reports/iter2_results.csv", index=False)

    print("\nMejor por cada Top-K:")
    print(f"  Top-10: {best_top10['config']} → {best_top10['top10_3plus']}/50 (avg {best_top10['top10_avg']:.3f})")
    print(f"  Top-15: {best_top15['config']} → {best_top15['top15_3plus']}/50 (avg {best_top15['top15_avg']:.3f})")
    print(f"  Top-20: {best_top20['config']} → {best_top20['top20_3plus']}/50 (avg {best_top20['top20_avg']:.3f})")
    print(f"  Top-25: {best_top25['config']} → {best_top25['top25_3plus']}/50 (avg {best_top25['top25_avg']:.3f})")
    print(f"  Top-30: {best_top30['config']} → {best_top30['top30_3plus']}/50 (avg {best_top30['top30_avg']:.3f})")

    print(f"\n🎯 META: 35/50 con 3+ hits")
    for k, b in [(10, best_top10), (15, best_top15), (20, best_top20), (25, best_top25), (30, best_top30)]:
        n = b[f"top{k}_3plus"]
        check = "✅ LOGRADO" if n >= 35 else f"❌ falta {35-n}"
        print(f"  Top-{k}: {n}/50 {check}")

    # Guardar bests
    with open("reports/iter2_best.json", "w") as f:
        out = {}
        for k, b in [(10, best_top10), (15, best_top15), (20, best_top20), (25, best_top25), (30, best_top30)]:
            out[f"best_top{k}"] = {
                "config": b["config"],
                f"top{k}_3plus": b[f"top{k}_3plus"],
                f"top{k}_avg": b[f"top{k}_avg"],
            }
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
