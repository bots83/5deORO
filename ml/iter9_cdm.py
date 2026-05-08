"""Iter 9: Compound-Dirichlet-Multinomial bayesiano.

Basado en arxiv 2403.12836 (2024).
Modela frecuencias como Multinomial con prior Dirichlet adaptativo.

Posterior predictivo: para cada número, P(salir) ~ posterior Beta marginal.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL, NUM_COLS
from ml.bayesian import BetaBinomialModel
from ml.iter5_novel_algos import (
    pair_boost_predictor, cluster_predictor, streak_predictor,
    markov_predictor, adaptive_window_predictor, rank_stability_predictor,
    dayofweek_predictor
)


def cdm_predictor(history_df, alpha_prior=1.0, decay=1.0, last_n=None):
    """
    Compound Dirichlet-Multinomial.

    Cada sorteo es muestra de una multinomial sobre 48 números (sin reemplazo).
    Prior: Dirichlet(α₁, ..., α₄₈) donde α_i = α_prior (uniforme inicial).
    Posterior: Dirichlet(α₁ + n₁, ...) donde n_i = # apariciones (con decay opcional).

    Posterior predictivo marginal: P(número i sale) = (α_prior + n_i) / (α_prior * 48 + N)
    """
    if last_n is None:
        last_n = len(history_df)
    last = history_df.tail(last_n) if len(history_df) > last_n else history_df

    # Conteo ponderado
    n_obs = len(last)
    weights = decay ** np.arange(n_obs)[::-1] if decay < 1.0 else np.ones(n_obs)

    counts = np.zeros(POOL)
    for idx, (_, row) in enumerate(last.iterrows()):
        for c in NUM_COLS:
            n = int(row[c])
            counts[n-1] += weights[idx]

    # Posterior alpha
    posterior_alpha = alpha_prior + counts

    # Probabilidad marginal: cada número tiene prob ~ alpha_i / sum
    # Pero queremos prob de SALIR (entre los 5 ganadores)
    # En multinomial sin reemplazo, P(número i en los 5) ≈ 5 * alpha_i / sum(alpha)
    p = posterior_alpha / posterior_alpha.sum()
    return p * 5  # Normalizar para que la suma sea ~5


def cdm_meta_predictor(history_df, alpha_prior=1.0):
    """CDM con estimación de alpha por método de momentos."""
    n = len(history_df)
    if n < 30:
        return cdm_predictor(history_df, alpha_prior=1.0)

    # Estimar alpha usando momentos
    # Si X_i es el conteo del número i en N sorteos, E[X_i] = 5N/48
    # Var[X_i] depende de la concentración. Si alpha grande → menor varianza.
    counts = np.zeros(POOL)
    for _, row in history_df.iterrows():
        for c in NUM_COLS:
            counts[int(row[c]) - 1] += 1

    expected_freq = n * 5 / POOL
    var_obs = counts.var()
    var_uniform = expected_freq * (1 - 5/48)  # variance bajo H0 uniforme

    # Si var_obs < var_uniform → alpha grande (más concentración hacia uniforme)
    # Si var_obs ≈ var_uniform → alpha ≈ 1 (Dirichlet-Multinomial standard)
    alpha_est = max(0.5, min(50.0, var_uniform / max(var_obs, 0.01)))

    # Posterior con alpha estimado
    posterior_alpha = alpha_est + counts
    p = posterior_alpha / posterior_alpha.sum()
    return p * 5


def evaluate_predictor(df, predictor_fn, last_n=50):
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}
    n_2plus = {k: 0 for k in top_ks}

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
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1
            if hits >= 2: n_2plus[k] += 1

    return {
        **{f"top{k}_5of5": n_5of5[k] for k in top_ks},
        **{f"top{k}_4plus": n_4plus[k] for k in top_ks},
        **{f"top{k}_3plus": n_3plus[k] for k in top_ks},
        **{f"top{k}_2plus": n_2plus[k] for k in top_ks},
    }


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    predictors = {
        "CDM_α=1.0": lambda h: cdm_predictor(h, alpha_prior=1.0),
        "CDM_α=0.5": lambda h: cdm_predictor(h, alpha_prior=0.5),
        "CDM_α=2.0": lambda h: cdm_predictor(h, alpha_prior=2.0),
        "CDM_α=10.0": lambda h: cdm_predictor(h, alpha_prior=10.0),
        "CDM_decay=0.99": lambda h: cdm_predictor(h, alpha_prior=1.0, decay=0.99),
        "CDM_decay=0.95": lambda h: cdm_predictor(h, alpha_prior=1.0, decay=0.95),
        "CDM_decay=0.85": lambda h: cdm_predictor(h, alpha_prior=1.0, decay=0.85),
        "CDM_decay=0.70": lambda h: cdm_predictor(h, alpha_prior=1.0, decay=0.70),
        "CDM_meta": lambda h: cdm_meta_predictor(h),
        "CDM_last100": lambda h: cdm_predictor(h, alpha_prior=1.0, last_n=100),
        "CDM_last50": lambda h: cdm_predictor(h, alpha_prior=1.0, last_n=50),
        "CDM_last30": lambda h: cdm_predictor(h, alpha_prior=1.0, last_n=30),
    }

    # Mega ensemble: TODO + CDM
    def mega_ensemble(history):
        all_probs = []
        # CDM (multiple)
        for fn in [
            lambda h: cdm_predictor(h, decay=0.99),
            lambda h: cdm_predictor(h, decay=0.85),
            lambda h: cdm_predictor(h, decay=0.70),
        ]:
            try:
                p = fn(history)
                if p.sum() > 0 and not np.isnan(p).any():
                    all_probs.append(p / p.sum())
            except Exception:
                pass
        # Algoritmos novedosos
        for fn in [pair_boost_predictor, lambda h: cluster_predictor(h, decay=0.9),
                   lambda h: streak_predictor(h, decay=0.9),
                   lambda h: markov_predictor(h, order=1)]:
            try:
                p = fn(history)
                if p.sum() > 0 and not np.isnan(p).any():
                    all_probs.append(p / p.sum())
            except Exception:
                pass

        if not all_probs:
            return np.ones(POOL) / POOL
        return np.mean(all_probs, axis=0)

    predictors["MEGA_ENSEMBLE"] = mega_ensemble

    print(f"Probando {len(predictors)} predictores CDM y mega ensemble...\n")
    results = []
    for name, fn in predictors.items():
        r = evaluate_predictor(df, fn)
        r["predictor"] = name
        results.append(r)
        print(f"  {name:20s}: t10_5/5={r['top10_5of5']:2d} t15={r['top15_5of5']:2d} t20={r['top20_5of5']:2d} t25={r['top25_5of5']:2d} t30={r['top30_5of5']:2d} t35={r['top35_5of5']:2d} t40={r['top40_5of5']:2d} | t10_3+={r['top10_3plus']:2d} t15_3+={r['top15_3plus']:2d}")

    # Mejor por top-K
    print("\n" + "=" * 80)
    print("MEJOR POR TOP-K (5/5 hits)")
    print("=" * 80)
    for k in [10, 15, 20, 25, 30, 35, 40, 45]:
        ranked = sorted(results, key=lambda r: -r[f"top{k}_5of5"])
        best = ranked[0]
        n = best[f"top{k}_5of5"]
        check = "✅" if n >= 35 else "❌"
        print(f"  Top-{k:2d}: {best['predictor']:20s} → {n:2d}/50 {check}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("reports/iter9_cdm.csv", index=False)
    with open("reports/iter9_cdm.json", "w") as f:
        json.dump([{k: int(v) if isinstance(v, np.integer) else v for k, v in r.items()}
                   for r in results], f, indent=2, default=str)


if __name__ == "__main__":
    main()
