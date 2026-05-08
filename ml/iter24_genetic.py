"""Iter 24: Genetic algorithm para encontrar combinaciones óptimas.

Para cada sorteo:
1. Generamos 100 candidatos (combinaciones de 10 números)
2. Cada candidato se evalúa contra patrones del histórico (suma, paridad, etc.)
3. Crossover y mutación entre los mejores
4. Después de 50 generaciones, top-1 candidato

Esto puede capturar interacciones no lineales entre números.
"""
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs


def fitness(candidate_set, base_probs, pair_count, target_size):
    """Evalúa una combinación basándose en:
    - Suma de probs base de los números
    - Coocurrencia entre los números (favorecer pares frecuentes)
    - Diversidad de decenas
    """
    score = 0
    nums = list(candidate_set)
    # Probs base
    for n in nums:
        score += base_probs[n - 1] * 1.0
    # Pares
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            key = tuple(sorted([nums[i], nums[j]]))
            score += pair_count.get(key, 0) * 0.001
    # Diversidad de decenas: penalty por concentración
    decenas = [0, 0, 0, 0]
    for n in nums:
        decenas[(n - 1) // 12] += 1
    target_per_dec = target_size / 4
    diversity_penalty = sum(abs(d - target_per_dec) for d in decenas) * 0.5
    score -= diversity_penalty
    return score


def genetic_search(base_probs, pair_count, top_k=10, n_pop=50, n_gen=30):
    """GA: encontrar combinación de top_k números que maximice fitness."""
    rng = np.random.default_rng(42)

    # Inicializar población con biased toward higher base_probs
    sorted_idx = np.argsort(base_probs)[::-1]
    pop = []
    for _ in range(n_pop):
        # Tomar top_k numbers con cierto noise
        if rng.random() < 0.5:
            # Variant del top-K base
            cand = list(sorted_idx[:top_k] + 1)
            # Mutar 1-2
            n_mut = rng.integers(1, 3)
            for _ in range(n_mut):
                i = rng.integers(0, len(cand))
                # Reemplazar con número de top-(K+5)
                pool_replace = list(set((sorted_idx[:top_k + 8] + 1).tolist()) - set(cand))
                if pool_replace:
                    cand[i] = rng.choice(pool_replace)
        else:
            # Random sample weighted by probs
            probs = base_probs / base_probs.sum()
            cand = rng.choice(POOL, top_k, replace=False, p=probs) + 1
        pop.append(set(cand.tolist()))

    for gen in range(n_gen):
        # Evaluar
        scored = [(fitness(c, base_probs, pair_count, top_k), c) for c in pop]
        scored.sort(reverse=True)
        # Top 50% sobreviven
        survivors = [c for _, c in scored[:n_pop // 2]]
        # Crossover entre survivors
        new_pop = list(survivors)
        while len(new_pop) < n_pop:
            p1 = list(rng.choice(survivors))
            p2 = list(rng.choice(survivors))
            child = set(p1[:top_k // 2] + p2[top_k // 2:])
            # Asegurar tamaño correcto
            while len(child) < top_k:
                child.add(rng.integers(1, POOL + 1))
            while len(child) > top_k:
                child.discard(rng.choice(list(child)))
            # Mutación
            if rng.random() < 0.3:
                to_remove = rng.choice(list(child))
                child.remove(to_remove)
                avail = list(set(range(1, POOL + 1)) - child)
                child.add(rng.choice(avail))
            new_pop.append(child)
        pop = new_pop

    # Mejor final
    scored = [(fitness(c, base_probs, pair_count, top_k), c) for c in pop]
    scored.sort(reverse=True)
    best = scored[0][1]
    return sorted(best)


def evaluate(df, last_n=50):
    n_total = len(df)
    start = n_total - last_n
    top_ks = [10, 15, 20, 25, 30, 35]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]

        # Base probs from CDM
        from ml.iter9_cdm import cdm_predictor
        from ml.iter5_novel_algos import cluster_predictor, pair_boost_predictor
        cdm = cdm_predictor(history, last_n=150)
        cluster = cluster_predictor(history, decay=0.85)
        pair = pair_boost_predictor(history)
        base = 0.74 * cluster / cluster.sum() + 0.22 * cdm / cdm.sum() + 0.04 * pair / pair.sum()

        # Pair count
        last100 = history.tail(100)
        pair_count = Counter()
        for _, r in last100.iterrows():
            nums = sorted([int(r[c]) for c in NUM_COLS])
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair_count[tuple(sorted([nums[i], nums[j]]))] += 1

        for k in top_ks:
            best_combo = genetic_search(base, pair_count, top_k=k, n_pop=30, n_gen=15)
            top_set = set(best_combo)
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in top_ks:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}")


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")
    print("Genetic search backtest...")
    evaluate(df)


if __name__ == "__main__":
    main()
