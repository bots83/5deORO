"""Iter 19: Per-sorteo adaptive selection.

Para cada sorteo a predecir:
1. Calcular features del sorteo (e.g., suma del último, paridad, etc.)
2. Encontrar los N sorteos históricos MÁS SIMILARES
3. Para esos N similares, ver QUÉ predictor base predijo mejor
4. Usar ESE predictor para el sorteo actual
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs

from sklearn.neighbors import NearestNeighbors


def context_features(history):
    """Features del contexto del último sorteo."""
    if len(history) == 0:
        return np.zeros(20)
    last = history.iloc[-1]
    nums = sorted([int(last[c]) for c in NUM_COLS])
    feats = [
        sum(nums) / 5,  # promedio
        np.std(nums),
        max(nums) - min(nums),  # rango
        sum(1 for n in nums if n % 2 == 0),  # pares
        sum(1 for n in nums if n <= 12),  # decena 1
        sum(1 for n in nums if 13 <= n <= 24),
        sum(1 for n in nums if 25 <= n <= 36),
        sum(1 for n in nums if 37 <= n <= 48),
        nums[0], nums[-1],
    ]
    # Stats últimos 5 sorteos
    if len(history) >= 5:
        last5 = history.tail(5)
        sums5 = [sum([int(r[c]) for c in NUM_COLS]) for _, r in last5.iterrows()]
        feats.extend([np.mean(sums5), np.std(sums5), min(sums5), max(sums5)])
    else:
        feats.extend([sum(nums), 0, sum(nums), sum(nums)])

    # Frecuencia de los últimos 20
    last_n = min(20, len(history))
    freq = np.zeros(POOL)
    for _, r in history.tail(last_n).iterrows():
        for c in NUM_COLS:
            freq[int(r[c]) - 1] += 1
    feats.extend([
        freq.std(),
        freq.max() - freq.min(),
        np.argmax(freq) + 1,  # número más frecuente
    ])
    feats.extend([0, 0, 0])  # padding
    return np.array(feats[:20], dtype=np.float32)


def evaluate_with_predictor(probs_arr, real, top_k=15):
    """Cuenta hits para un predictor."""
    sorted_idx = np.argsort(probs_arr)[::-1]
    top_set = set((sorted_idx[:top_k] + 1).tolist())
    return len(real & top_set)


def adaptive_predict(df, idx, n_neighbors=10, target_k=15):
    """
    Para el sorteo idx:
    1. Calcular features del contexto
    2. Encontrar los n_neighbors sorteos más similares (entre los anteriores)
    3. Para cada predictor base, ver su recall@target_k en esos n_neighbors
    4. Combinar predictores con pesos basados en su performance histórica
    """
    history = df.iloc[:idx]
    if len(history) < 60:
        return np.ones(POOL) / POOL

    # Features del sorteo a predecir (basado en contexto)
    target_feats = context_features(history)

    # Para cada sorteo en el histórico, su contexto
    context_vectors = []
    indices_to_consider = list(range(60, len(history)))  # Solo sorteos con suficiente historia
    for i in indices_to_consider:
        h = df.iloc[:i]
        cf = context_features(h)
        context_vectors.append(cf)
    context_array = np.array(context_vectors)

    # NN search
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(context_array)))
    nn.fit(context_array)
    distances, neighbor_indices = nn.kneighbors([target_feats])
    neighbor_idxs = [indices_to_consider[i] for i in neighbor_indices[0]]

    # Para cada neighbor, evaluar predictores
    predictor_scores = defaultdict(list)
    for n_idx in neighbor_idxs:
        n_history = df.iloc[:n_idx]
        n_real = {df.iloc[n_idx][f"n{i}"] for i in range(1, 6)}
        outputs = get_predictor_outputs(n_history)
        for name, probs in outputs.items():
            score = evaluate_with_predictor(probs, n_real, top_k=target_k)
            predictor_scores[name].append(score)

    # Normalizar scores → pesos
    avg_scores = {name: np.mean(scores) for name, scores in predictor_scores.items() if scores}
    if not avg_scores:
        return np.ones(POOL) / POOL
    total = sum(avg_scores.values())
    if total == 0:
        weights = {name: 1.0 / len(avg_scores) for name in avg_scores}
    else:
        weights = {name: score / total for name, score in avg_scores.items()}

    # Aplicar predictores al sorteo actual con pesos adaptativos
    outputs = get_predictor_outputs(history)
    ensemble = np.zeros(POOL)
    total_w = 0
    for name, w in weights.items():
        if name in outputs:
            ensemble += outputs[name] * w
            total_w += w
    if total_w == 0:
        return np.ones(POOL) / POOL
    return ensemble / total_w


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    n_total = len(df)
    last_n = 50
    start = n_total - last_n

    print(f"Backtest adaptive sobre {last_n} sorteos...\n")

    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for i, idx in enumerate(range(start, n_total)):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{j}"] for j in range(1, 6)}
        try:
            probs = adaptive_predict(df, idx, n_neighbors=10, target_k=15)
        except Exception as e:
            print(f"  err idx {idx}: {e}")
            probs = np.ones(POOL) / POOL
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{last_n} done. Top-31_3+={n_3plus[31]}", flush=True)

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in top_ks:
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    with open("reports/iter19_adaptive.json", "w") as f:
        out = {f"top{k}_5of5": n_5of5[k] for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]} | \
              {f"top{k}_4plus": n_4plus[k] for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]} | \
              {f"top{k}_3plus": n_3plus[k] for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]}
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in out.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
