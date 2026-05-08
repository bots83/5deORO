"""Iter 14: Optimización extrema de pair_boost.

PairBoost mostró buenos resultados aislados. Voy a:
1. Probar muchas variantes
2. Combinar pair_boost con triple-coocurrencia
3. Usar pair_boost adaptativo según el último sorteo
"""
import sys
import json
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS


def pair_boost_v2(history_df, window_pairs=100, base_decay=0.95, n_last_for_boost=5,
                   pair_weight=0.3, base_weight=0.7):
    """Pair boost mejorado con parámetros ajustables."""
    last = history_df.tail(window_pairs) if len(history_df) > window_pairs else history_df

    pair_count = Counter()
    for _, row in last.iterrows():
        nums = sorted([int(row[c]) for c in NUM_COLS])
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_count[(nums[i], nums[j])] += 1

    last_n = len(last)
    base_freq = np.zeros(POOL)
    weights = base_decay ** np.arange(last_n)[::-1]
    for idx, (_, row) in enumerate(last.iterrows()):
        for c in NUM_COLS:
            n = int(row[c])
            base_freq[n - 1] += weights[idx]
    if base_freq.sum() > 0:
        base_freq /= base_freq.sum()

    # Tomar n_last_for_boost ultimos sorteos para hacer boost
    last_set_aggregated = set()
    for k in range(min(n_last_for_boost, len(history_df))):
        for c in NUM_COLS:
            last_set_aggregated.add(int(history_df.iloc[-1 - k][c]))

    boost = np.zeros(POOL)
    for n in range(1, POOL + 1):
        for last_n_val in last_set_aggregated:
            if n == last_n_val:
                continue
            key = tuple(sorted([n, last_n_val]))
            boost[n - 1] += pair_count.get(key, 0)
    if boost.sum() > 0:
        boost /= boost.sum()

    final = base_weight * base_freq + pair_weight * boost
    return final / final.sum() if final.sum() > 0 else np.ones(POOL) / POOL


def triple_cooc_predictor(history_df, window=80):
    """Coocurrencia de tripletas."""
    last = history_df.tail(window) if len(history_df) > window else history_df
    triple_count = Counter()
    for _, row in last.iterrows():
        nums = sorted([int(row[c]) for c in NUM_COLS])
        for combo in combinations(nums, 3):
            triple_count[combo] += 1

    # Score: para cada número, suma de triples que lo incluyen relacionados con últimos
    last_set = set()
    for k in range(min(3, len(history_df))):
        for c in NUM_COLS:
            last_set.add(int(history_df.iloc[-1 - k][c]))

    probs = np.zeros(POOL)
    for n in range(1, POOL + 1):
        for combo in combinations(last_set, 2):
            t = tuple(sorted(list(combo) + [n]))
            if len(set(t)) == 3:
                probs[n - 1] += triple_count.get(t, 0)
    if probs.sum() > 0:
        probs /= probs.sum()
    else:
        probs = np.ones(POOL) / POOL
    return probs


def evaluate_predictor(df, predictor_fn, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 20, 25, 30, 40, 45]
    n_5of5 = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        try:
            probs = predictor_fn(history)
        except Exception:
            probs = np.ones(POOL) / POOL
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 3: n_3plus[k] += 1

    return {**{f"top{k}_5of5": n_5of5[k] for k in top_ks},
            **{f"top{k}_3plus": n_3plus[k] for k in top_ks}}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    # Probar varias variantes de pair_boost
    configs = []
    for window in [50, 80, 100, 150, 200]:
        for decay in [0.85, 0.92, 0.95, 0.97]:
            for n_last in [3, 5, 7]:
                for pw in [0.2, 0.3, 0.4]:
                    configs.append((window, decay, n_last, pw))

    results = []
    print(f"Probando {len(configs)} variantes de pair_boost...\n")
    for i, (w, d, nl, pw) in enumerate(configs):
        if i % 20 == 0:
            print(f"  [{i+1}/{len(configs)}]")
        r = evaluate_predictor(df, lambda h, w=w, d=d, nl=nl, pw=pw: pair_boost_v2(h, window_pairs=w, base_decay=d, n_last_for_boost=nl, pair_weight=pw, base_weight=1-pw))
        r["config"] = f"PB(w{w},d{d},nl{nl},pw{pw})"
        results.append(r)

    print("\nTriple-coocurrence:")
    r_triple = evaluate_predictor(df, triple_cooc_predictor)
    r_triple["config"] = "TripleCooc"
    results.append(r_triple)
    print(f"  TripleCooc: t10_5/5={r_triple['top10_5of5']} t15={r_triple['top15_5of5']} t20={r_triple['top20_5of5']} t30={r_triple['top30_5of5']} t10_3+={r_triple['top10_3plus']}")

    # Top configs
    print("\n=== TOP 10 CONFIGS POR top10_3+ ===")
    sorted_res = sorted(results, key=lambda r: -r["top10_3plus"])
    for r in sorted_res[:10]:
        print(f"  {r['config']:35s} t10_5/5={r['top10_5of5']} t15={r['top15_5of5']} t20={r['top20_5of5']} | t10_3+={r['top10_3plus']} t15_3+={r['top15_3plus']}")

    print("\n=== TOP 10 POR top15_5/5 ===")
    sorted_res = sorted(results, key=lambda r: -r["top15_5of5"])
    for r in sorted_res[:10]:
        print(f"  {r['config']:35s} t10_5/5={r['top10_5of5']} t15={r['top15_5of5']} t20={r['top20_5of5']} | t10_3+={r['top10_3plus']}")

    pd.DataFrame(results).to_csv("reports/iter14_pair_opt.csv", index=False)


if __name__ == "__main__":
    main()
