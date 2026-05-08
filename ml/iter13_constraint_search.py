"""Iter 13: Selección con restricciones del histórico.

Análisis: en el histórico, ¿qué propiedades tienen los 5 números ganadores?
- Suma típica: 122-125
- Pares: 2-3 de los 5
- Decenas: distribución específica
- Diferencias mín/máx: típicas

Si predeicemos un Top-K y luego FILTRAMOS por restricciones,
podemos mejorar el match con el ganador real.
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
from ml.iter5_novel_algos import pair_boost_predictor, cluster_predictor
from ml.iter9_cdm import cdm_predictor


WEIGHTS_OPTIMAL = {
    "cdm_last150": 0.218,
    "pair": 0.045,
    "cluster085": 0.737,
}


def predict_optimal(history):
    components = {}
    for name, fn in [
        ("cdm_last150", lambda h: cdm_predictor(h, last_n=150)),
        ("pair", pair_boost_predictor),
        ("cluster085", lambda h: cluster_predictor(h, decay=0.85)),
    ]:
        try:
            p = fn(history)
            if p.sum() > 0:
                components[name] = p / p.sum()
        except Exception:
            pass

    ensemble = np.zeros(POOL)
    total_w = 0
    for n, w in WEIGHTS_OPTIMAL.items():
        if n in components:
            ensemble += components[n] * w
            total_w += w
    return ensemble / total_w if total_w > 0 else np.ones(POOL) / POOL


def analyze_winning_patterns(df):
    """Analiza propiedades del histórico de ganadores."""
    sums = []
    parities = []
    decades_per_draw = []
    spreads = []  # max - min
    consec = []   # # números consecutivos

    for _, row in df.iterrows():
        nums = sorted([int(row[c]) for c in NUM_COLS])
        sums.append(sum(nums))
        parities.append(sum(1 for n in nums if n % 2 == 0))
        d = [0,0,0,0]
        for n in nums:
            d[(n-1)//12] += 1
        decades_per_draw.append(tuple(d))
        spreads.append(nums[-1] - nums[0])
        # Consecutivos
        c = sum(1 for i in range(4) if nums[i+1] - nums[i] == 1)
        consec.append(c)

    return {
        "sum_mean": np.mean(sums), "sum_std": np.std(sums),
        "sum_min": min(sums), "sum_max": max(sums),
        "parity_mean": np.mean(parities), "parity_dist": Counter(parities),
        "decade_dist": Counter(decades_per_draw),
        "spread_mean": np.mean(spreads), "spread_std": np.std(spreads),
        "consec_dist": Counter(consec),
    }


def is_typical_draw(nums, patterns, tolerance=2.0):
    """¿Esta combinación se ve "típica" del histórico?"""
    nums = sorted(nums)
    s = sum(nums)
    par = sum(1 for n in nums if n % 2 == 0)
    spread = nums[-1] - nums[0]

    # Suma dentro de 2 std
    if abs(s - patterns["sum_mean"]) > tolerance * patterns["sum_std"]:
        return False

    # Spread dentro de 2 std
    if abs(spread - patterns["spread_mean"]) > tolerance * patterns["spread_std"]:
        return False

    return True


def smart_select_with_constraints(probs, k, patterns):
    """
    Selecciona los Top-K, pero PRIMERO verifica que el conjunto se vea típico.
    Si los top-K naive son atípicos, intercambia con candidatos cercanos.
    """
    sorted_idx = np.argsort(probs)[::-1]
    top_k_naive = sorted((sorted_idx[:k] + 1).tolist())

    # Si la combinación naive es típica, devolverla
    if is_typical_draw(top_k_naive, patterns):
        return top_k_naive

    # Sino, buscar combinación de candidates cercanos que cumpla restricciones
    # Tomar top-(k+10) y buscar combinatoriamente
    candidates = (sorted_idx[:k+8] + 1).tolist()
    if k <= 8:  # solo para top pequeños
        best_combo = top_k_naive
        for combo in combinations(candidates, k):
            if is_typical_draw(combo, patterns):
                # Score por suma de probs
                total_p = sum(probs[n-1] for n in combo)
                if total_p > sum(probs[n-1] for n in best_combo):
                    best_combo = sorted(combo)
        return list(best_combo)
    return top_k_naive


def evaluate_with_constraints(df, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 18, 20, 25, 30]
    n_naive_5of5 = {k: 0 for k in top_ks}
    n_naive_4plus = {k: 0 for k in top_ks}
    n_naive_3plus = {k: 0 for k in top_ks}
    n_const_5of5 = {k: 0 for k in top_ks}
    n_const_4plus = {k: 0 for k in top_ks}
    n_const_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        patterns = analyze_winning_patterns(history)
        probs = predict_optimal(history)
        sorted_idx = np.argsort(probs)[::-1]

        for k in top_ks:
            naive_top = set((sorted_idx[:k] + 1).tolist())
            const_top = set(smart_select_with_constraints(probs, k, patterns))

            h_n = len(real & naive_top)
            h_c = len(real & const_top)
            if h_n == 5: n_naive_5of5[k] += 1
            if h_n >= 4: n_naive_4plus[k] += 1
            if h_n >= 3: n_naive_3plus[k] += 1
            if h_c == 5: n_const_5of5[k] += 1
            if h_c >= 4: n_const_4plus[k] += 1
            if h_c >= 3: n_const_3plus[k] += 1

    print(f"\n{'Top-K':<8} {'NAIVE 5/5':>10} {'CONST 5/5':>10} {'NAIVE 3+':>10} {'CONST 3+':>10}")
    for k in top_ks:
        marker_5 = "✓" if n_const_5of5[k] > n_naive_5of5[k] else "" if n_const_5of5[k] == n_naive_5of5[k] else "✗"
        marker_3 = "✓" if n_const_3plus[k] > n_naive_3plus[k] else "" if n_const_3plus[k] == n_naive_3plus[k] else "✗"
        print(f"top-{k:<4} {n_naive_5of5[k]:>10} {n_const_5of5[k]:>10} {marker_5} {n_naive_3plus[k]:>9} {n_const_3plus[k]:>9} {marker_3}")


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    # Analizar patrones
    patterns = analyze_winning_patterns(df)
    print("PATRONES DEL HISTÓRICO:")
    print(f"  Suma: {patterns['sum_mean']:.1f} ± {patterns['sum_std']:.1f} (rango {patterns['sum_min']}-{patterns['sum_max']})")
    print(f"  Pares por sorteo: media {patterns['parity_mean']:.2f}")
    print(f"  Spread: {patterns['spread_mean']:.1f} ± {patterns['spread_std']:.1f}")
    print(f"  Decenas más comunes: {patterns['decade_dist'].most_common(5)}")

    print("\nEvaluando NAIVE vs CONSTRAINT-AWARE:\n")
    evaluate_with_constraints(df)


if __name__ == "__main__":
    main()
