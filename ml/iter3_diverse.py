"""Iter 3: Maximizar SORTEOS con ≥1 hit en top-10 (minimizar n_zero).

Estrategia:
- Probar muchos top-K (10, 12, 15) y ver cuál llega a 35/50 con ≥1 hit
- Usar ensembles muy diversos para evitar zero-hits
- Boost: si un número no aparece en X sorteos, aumentar su peso ("debido")
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL, NUM_COLS
from ml.bayesian import BetaBinomialModel
from ml.baseline import FrequencyBaseline


def overdue_predictor(history, alpha=2.0):
    """Modelo 'debido': números que llevan más tiempo sin salir tienen mayor prob.
    No es un modelo válido (gambler's fallacy) pero diversifica el ensemble.
    """
    if "ausencia" in str(history):  # already a feats dict
        feats = history
        probs = np.zeros(POOL)
        for n in range(1, POOL + 1):
            probs[n-1] = (feats.get(f"ausencia_{n}", 0) + 1) ** alpha
        return probs / probs.sum()
    return np.ones(POOL) / POOL


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
            # Modelo custom: name, function(feats_pred), weight
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
    for idx in range(start, n_total):
        history = df.iloc[:idx]
        train_df = build_features(history, min_history=30)
        cache[idx] = train_df
    return cache


def evaluate_full(df, train_cache, models_config, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    hits = {k: [] for k in [5, 10, 12, 15, 20]}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        probs = predict_idx(df, idx, train_cache, models_config)
        sorted_idx = np.argsort(probs)[::-1]
        for k in hits:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits[k].append(len(real & top_set))

    result = {}
    for k in hits:
        h = hits[k]
        result[f"top{k}_avg"] = float(np.mean(h))
        result[f"top{k}_n_zero"] = sum(1 for x in h if x == 0)
        result[f"top{k}_n_1plus"] = sum(1 for x in h if x >= 1)
        result[f"top{k}_n_2plus"] = sum(1 for x in h if x >= 2)
        result[f"top{k}_n_3plus"] = sum(1 for x in h if x >= 3)
    result["hits_top10"] = hits[10]
    return result


def gen_diverse_configs():
    configs = []

    # Mezclas hot-cold para diversificar
    decays_hot = [0.80, 0.70, 0.60]  # más peso a recientes
    decays_cold = [1.0, 0.99, 0.95]  # peso uniforme a histórico

    # 1. Configs de iter 1 ganadoras
    configs.append(("BB3_winner", [
        ("bb1", BetaBinomialModel, {"decay": 0.99}, 0.7),
        ("bb2", BetaBinomialModel, {"decay": 0.95}, 0.2),
        ("bb3", BetaBinomialModel, {"decay": 0.70}, 0.1),
    ]))

    # 2. Mezclas hot+cold balanceadas
    for d_hot in decays_hot:
        for d_cold in decays_cold:
            for w_hot in [0.3, 0.4, 0.5, 0.6, 0.7]:
                w_cold = 1 - w_hot
                configs.append((f"Mix_h{d_hot}_c{d_cold}_w{w_hot}", [
                    ("hot", BetaBinomialModel, {"decay": d_hot}, w_hot),
                    ("cold", BetaBinomialModel, {"decay": d_cold}, w_cold),
                ]))

    # 3. 4-modelos combinados
    for w in [(0.4, 0.3, 0.2, 0.1), (0.3, 0.3, 0.2, 0.2), (0.5, 0.25, 0.15, 0.1),
              (0.25, 0.25, 0.25, 0.25)]:
        configs.append((f"BB4_w{w[0]}", [
            ("bb1", BetaBinomialModel, {"decay": 1.0}, w[0]),
            ("bb2", BetaBinomialModel, {"decay": 0.95}, w[1]),
            ("bb3", BetaBinomialModel, {"decay": 0.85}, w[2]),
            ("bb4", BetaBinomialModel, {"decay": 0.65}, w[3]),
        ]))

    # 4. Custom: overdue boost
    def overdue_fn(alpha):
        def fn(feats):
            probs = np.zeros(POOL)
            for n in range(1, POOL + 1):
                probs[n-1] = (feats.get(f"ausencia_{n}", 0) + 1) ** alpha
            return probs / probs.sum()
        return fn

    for alpha in [0.3, 0.5, 0.8, 1.0, 1.5]:
        configs.append((f"Overdue_a{alpha}", [
            ("overdue", overdue_fn(alpha), 1.0),
        ]))

    # 5. BB + overdue
    for w_o in [0.2, 0.3, 0.5]:
        for alpha in [0.5, 1.0]:
            configs.append((f"BB+Overdue_w{w_o}_a{alpha}", [
                ("bb1", BetaBinomialModel, {"decay": 0.99}, (1-w_o) * 0.7),
                ("bb2", BetaBinomialModel, {"decay": 0.70}, (1-w_o) * 0.3),
                ("overdue", overdue_fn(alpha), w_o),
            ]))

    return configs


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    print("Pre-computing features (basic)...")
    cache = precompute(df, last_n=50)

    configs = gen_diverse_configs()
    print(f"Total configs: {len(configs)}\n")

    results = []
    bests = {f"top{k}": None for k in [10, 12, 15, 20]}

    for i, (name, conf) in enumerate(configs):
        try:
            t0 = time.time()
            r = evaluate_full(df, cache, conf, last_n=50)
            r["config"] = name
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)

            updates = []
            for k in [10, 12, 15, 20]:
                key = f"top{k}_n_1plus"
                if bests[f"top{k}"] is None or r[key] > bests[f"top{k}"][key]:
                    bests[f"top{k}"] = {**r, "config_obj": conf}
                    updates.append(f"top{k}_1+")

            mark = " ⭐ " + ", ".join(updates) if updates else ""
            if i % 5 == 0 or mark:
                print(f"  [{i+1}/{len(configs)}] {name:35s} t10_1+={r['top10_n_1plus']:2d}/50 t10_0={r['top10_n_zero']:2d} t12_1+={r['top12_n_1plus']:2d} t15_1+={r['top15_n_1plus']:2d} t20_1+={r['top20_n_1plus']:2d}{mark}")
        except Exception as e:
            print(f"  [{i+1}] {name}: ERROR {e}")

    print("\n" + "=" * 100)
    print("MEJORES POR # SORTEOS CON ≥1 HIT")
    print("=" * 100)
    for k in [10, 12, 15, 20]:
        b = bests[f"top{k}"]
        n1 = b[f"top{k}_n_1plus"]
        check = "✅ LOGRADO" if n1 >= 35 else f"❌ falta {35-n1}"
        print(f"  Top-{k}: {b['config']} → {n1}/50 con ≥1 hit {check}")
        print(f"          (n_zero={b[f'top{k}_n_zero']}, n_2+={b[f'top{k}_n_2plus']}, n_3+={b[f'top{k}_n_3plus']})")

    # Guardar
    df_res = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in results])
    df_res.to_csv("reports/iter3_results.csv", index=False)

    out = {}
    for k, b in bests.items():
        out[f"best_{k}"] = {kk: vv for kk, vv in b.items() if not isinstance(vv, (list, type(lambda: 1)))}
        out[f"best_{k}"].pop("config_obj", None)
    with open("reports/iter3_best.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
