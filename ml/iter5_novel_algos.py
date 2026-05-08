"""Iter 5: Algoritmos novedosos que buscan ventaja real en el historial.

Algoritmos:
1. Markov Chain N-orden (transiciones entre sorteos)
2. Pair-correlation boost (qué números aparecen juntos frecuentemente)
3. Day-of-week patterns (¿hay sesgo por día de la semana?)
4. Sliding bayes con prior dinámico
5. Ensemble adaptativo: pesos cambian según contexto
6. NUEVO: Frequency rank stability tracking
7. NUEVO: Number cluster detection
"""
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL, NUM_COLS
from ml.bayesian import BetaBinomialModel
from ml.baseline import FrequencyBaseline


# =============================================================================
# ALGORITMOS NOVELES
# =============================================================================

def markov_predictor(history_df, order=1):
    """Modelo de Markov: P(num en t | num en t-1, t-2, ..., t-order)."""
    nums_per_draw = [set([row[c] for c in NUM_COLS]) for _, row in history_df.iterrows()]
    if len(nums_per_draw) < order + 5:
        return np.ones(POOL) / POOL

    # Para cada número, P(salir | salió en t-1)
    # transitions[n] = (count_after_present, count_after_absent)
    counts = {n: [0, 0, 0, 0] for n in range(1, POOL+1)}  # [present->present, present->absent, absent->present, absent->absent]
    for t in range(1, len(nums_per_draw)):
        for n in range(1, POOL+1):
            in_prev = n in nums_per_draw[t-1]
            in_curr = n in nums_per_draw[t]
            if in_prev and in_curr:
                counts[n][0] += 1
            elif in_prev and not in_curr:
                counts[n][1] += 1
            elif not in_prev and in_curr:
                counts[n][2] += 1
            else:
                counts[n][3] += 1

    last_set = nums_per_draw[-1]
    probs = np.zeros(POOL)
    for n in range(1, POOL+1):
        c = counts[n]
        if n in last_set:
            # Estaba en el último → P(salir ahora) = c[0] / (c[0]+c[1])
            denom = c[0] + c[1]
            probs[n-1] = (c[0] + 0.5) / (denom + 1) if denom > 0 else 5/48
        else:
            denom = c[2] + c[3]
            probs[n-1] = (c[2] + 0.5) / (denom + 1) if denom > 0 else 5/48
    return probs / probs.sum() * 5  # normalizar para que la suma sea ~5 (5 nums por sorteo)


def pair_boost_predictor(history_df, last_n_for_pairs=100, base_decay=0.95):
    """Aumenta probabilidad de números que han aparecido junto con los recientes."""
    last = history_df.tail(last_n_for_pairs) if len(history_df) > last_n_for_pairs else history_df

    # Frecuencia de pares
    pair_count = Counter()
    for _, row in last.iterrows():
        nums = sorted([row[c] for c in NUM_COLS])
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                pair_count[(nums[i], nums[j])] += 1

    # Frecuencia base
    last_n = len(last)
    base_freq = np.zeros(POOL)
    weights = base_decay ** np.arange(last_n)[::-1]
    for idx, (_, row) in enumerate(last.iterrows()):
        for c in NUM_COLS:
            n = int(row[c])
            base_freq[n-1] += weights[idx]
    base_freq /= base_freq.sum()

    # Boost: para cada número, sumar frecuencias de sus pares con los más probables del último sorteo
    last_set = [int(history_df.iloc[-1][c]) for c in NUM_COLS]
    boost = np.zeros(POOL)
    for n in range(1, POOL+1):
        for last_n_val in last_set:
            if n == last_n_val:
                continue
            key = tuple(sorted([n, last_n_val]))
            boost[n-1] += pair_count.get(key, 0)
    if boost.sum() > 0:
        boost /= boost.sum()

    # Mezclar 70% base + 30% pairs
    final = 0.7 * base_freq + 0.3 * boost
    return final / final.sum() * 5


def cluster_predictor(history_df, n_clusters=5, decay=0.95):
    """Detecta clusters de números que tienden a salir juntos y usa eso para predecir."""
    # Matriz de coocurrencia
    cooc = np.zeros((POOL, POOL))
    last_n = min(len(history_df), 200)
    weights = decay ** np.arange(last_n)[::-1]

    for idx, (_, row) in enumerate(history_df.tail(last_n).iterrows()):
        nums = [int(row[c]) - 1 for c in NUM_COLS]
        w = weights[idx]
        for i in nums:
            for j in nums:
                if i != j:
                    cooc[i, j] += w

    # Para cada número, prob proporcional a cooc con los del último sorteo
    last_nums = [int(history_df.iloc[-1][c]) - 1 for c in NUM_COLS]
    probs = np.zeros(POOL)
    for n in range(POOL):
        for ln in last_nums:
            probs[n] += cooc[n, ln]
    if probs.sum() > 0:
        probs /= probs.sum()
    else:
        probs = np.ones(POOL) / POOL
    return probs * 5


def streak_predictor(history_df, decay=0.95):
    """Penaliza números 'calientes' (recién salieron) y favorece 'fríos'."""
    last_n = min(len(history_df), 100)
    weights = decay ** np.arange(last_n)[::-1]

    # Frecuencia ponderada
    freq = np.zeros(POOL)
    for idx, (_, row) in enumerate(history_df.tail(last_n).iterrows()):
        for c in NUM_COLS:
            n = int(row[c])
            freq[n-1] += weights[idx]
    freq /= freq.sum()

    # Streak: ¿salió en el último?
    last_set = set([int(history_df.iloc[-1][c]) for c in NUM_COLS])
    # Penalizar últimos 1-2 sorteos
    last2_set = set()
    for k in range(min(2, len(history_df))):
        for c in NUM_COLS:
            last2_set.add(int(history_df.iloc[-1-k][c]))

    # Boost a los que NO han salido recientemente
    probs = freq.copy()
    for n in range(1, POOL+1):
        if n in last_set:
            probs[n-1] *= 0.7  # penalizar levemente
        elif n in last2_set:
            probs[n-1] *= 0.9
        else:
            probs[n-1] *= 1.05
    probs /= probs.sum()
    return probs * 5


def dayofweek_predictor(history_df, target_day=None):
    """Frecuencia condicional por día de la semana."""
    if "fecha" not in history_df.columns:
        return np.ones(POOL) / POOL * 5

    history_df = history_df.copy()
    history_df["fecha"] = pd.to_datetime(history_df["fecha"])
    if target_day is None:
        # Predecir para el siguiente sorteo, asumir día más común
        target_day = history_df.iloc[-1]["fecha"].dayofweek

    # Frecuencia solo de sorteos del mismo día
    same_day = history_df[history_df["fecha"].dt.dayofweek == target_day]
    if len(same_day) < 10:
        same_day = history_df  # fallback

    freq = np.zeros(POOL)
    for _, row in same_day.iterrows():
        for c in NUM_COLS:
            n = int(row[c])
            freq[n-1] += 1
    if freq.sum() > 0:
        freq /= freq.sum()
    else:
        freq = np.ones(POOL) / POOL
    return freq * 5


def adaptive_window_predictor(history_df, target_idx=None):
    """Usa ventana dinámica que se ajusta al sorteo."""
    # Multi-window ensemble: combinar windows 20, 50, 100, all-time
    windows = [20, 50, 100, 200]
    probs_combined = np.zeros(POOL)
    weights_per_window = [0.4, 0.3, 0.2, 0.1]

    for w_idx, window in enumerate(windows):
        last = history_df.tail(window) if len(history_df) > window else history_df
        if len(last) < 5:
            continue
        freq = np.zeros(POOL)
        for _, row in last.iterrows():
            for c in NUM_COLS:
                n = int(row[c])
                freq[n-1] += 1
        freq /= freq.sum()
        probs_combined += freq * weights_per_window[w_idx]

    return probs_combined / probs_combined.sum() * 5


def rank_stability_predictor(history_df, lookback=30):
    """Identifica números cuyo ranking de frecuencia ha sido estable arriba."""
    last = history_df.tail(lookback) if len(history_df) > lookback else history_df
    if len(last) < 10:
        return np.ones(POOL) / POOL * 5

    # Calcular frecuencia en cada subventana de 10 sorteos
    rankings_history = []
    for end in range(10, len(last)+1, 5):
        sub = last.iloc[max(0, end-20):end]
        freq = np.zeros(POOL)
        for _, row in sub.iterrows():
            for c in NUM_COLS:
                freq[int(row[c])-1] += 1
        rank = (-freq).argsort().argsort()  # rank 0 = más frecuente
        rankings_history.append(rank)

    if not rankings_history:
        return np.ones(POOL) / POOL * 5

    rankings = np.array(rankings_history)
    # Score: -mean(rank) - std(rank)
    mean_rank = rankings.mean(axis=0)
    std_rank = rankings.std(axis=0)
    score = -mean_rank - std_rank * 0.5
    score = np.exp(score / 5)  # softmax-like
    score /= score.sum()
    return score * 5


# =============================================================================
# EVALUADOR
# =============================================================================

def evaluate_algo(df, predictor_fn, predictor_name, last_n=50):
    """Evalúa un predictor sobre los últimos last_n sorteos."""
    n_total = len(df)
    start = max(60, n_total - last_n)
    top_ks = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        try:
            probs = predictor_fn(history)
        except Exception as e:
            print(f"    Error en idx {idx}: {e}")
            probs = np.ones(POOL) / POOL
        sorted_idx = np.argsort(probs)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    return {
        "predictor": predictor_name,
        **{f"top{k}_5of5": n_5of5[k] for k in top_ks},
        **{f"top{k}_4plus": n_4plus[k] for k in top_ks},
        **{f"top{k}_3plus": n_3plus[k] for k in top_ks},
    }


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")
    print("Probando algoritmos novedosos...\n")

    predictors = [
        ("Markov order=1", lambda h: markov_predictor(h, order=1)),
        ("PairBoost", lambda h: pair_boost_predictor(h)),
        ("PairBoost(50, 0.97)", lambda h: pair_boost_predictor(h, last_n_for_pairs=50, base_decay=0.97)),
        ("Cluster(decay=0.95)", lambda h: cluster_predictor(h, decay=0.95)),
        ("Cluster(decay=0.85)", lambda h: cluster_predictor(h, decay=0.85)),
        ("Streak(0.95)", lambda h: streak_predictor(h, decay=0.95)),
        ("Streak(0.85)", lambda h: streak_predictor(h, decay=0.85)),
        ("DayOfWeek", lambda h: dayofweek_predictor(h)),
        ("AdaptiveWindow", lambda h: adaptive_window_predictor(h)),
        ("RankStability", lambda h: rank_stability_predictor(h)),
    ]

    results = []
    for name, fn in predictors:
        t0 = time.time()
        try:
            r = evaluate_algo(df, fn, name)
            r["time_s"] = round(time.time() - t0, 1)
            results.append(r)
            print(f"  {name:25s}: t10_5/5={r['top10_5of5']:2d} t20_5/5={r['top20_5of5']:2d} t30_5/5={r['top30_5of5']:2d} t40_5/5={r['top40_5of5']:2d} t10_3+={r['top10_3plus']:2d} ({r['time_s']}s)")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    # Ahora ENSEMBLE de los algoritmos novedosos
    print("\n" + "=" * 100)
    print("ENSEMBLE DE TODOS LOS ALGORITMOS")
    print("=" * 100)

    def ensemble_all(history):
        probs_list = []
        for _, fn in predictors:
            try:
                p = fn(history)
                if not np.isnan(p).any() and p.sum() > 0:
                    probs_list.append(p / p.sum())
            except Exception:
                pass
        if not probs_list:
            return np.ones(POOL) / POOL
        return np.mean(probs_list, axis=0)

    r_ens = evaluate_algo(df, ensemble_all, "Ensemble_All")
    print(f"\nEnsemble_All: t10_5/5={r_ens['top10_5of5']} t15={r_ens['top15_5of5']} t20={r_ens['top20_5of5']} t25={r_ens['top25_5of5']} t30={r_ens['top30_5of5']} t35={r_ens['top35_5of5']} t40={r_ens['top40_5of5']} t45={r_ens['top45_5of5']}")
    results.append(r_ens)

    # BB ensemble + algoritmos novedosos
    def super_ensemble(history):
        # BB tradicional
        train_df = build_features(history, min_history=30)
        if len(train_df) < 20:
            return np.ones(POOL) / POOL
        feat_cols = [c for c in train_df.columns if not c.startswith("target_")]
        target_cols = [c for c in train_df.columns if c.startswith("target_")]
        X_train = train_df[feat_cols].values.astype(np.float32)
        y_train = train_df[target_cols].values.astype(np.int32)

        feats_pred = _build_features_for_row(history)
        X_pred = np.array([[feats_pred.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

        ensemble_probs = np.zeros(POOL)
        # BB(0.99) y BB(0.7)
        for d, w in [(0.99, 0.4), (0.7, 0.2)]:
            try:
                m = BetaBinomialModel(decay=d)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_pred)[0]
                ensemble_probs += p * w
            except Exception:
                pass
        # Algoritmos novedosos
        for fn_pair, weight in [(pair_boost_predictor, 0.15),
                                (lambda h: cluster_predictor(h, decay=0.9), 0.1),
                                (lambda h: streak_predictor(h, decay=0.9), 0.05),
                                (lambda h: adaptive_window_predictor(h), 0.1)]:
            try:
                p = fn_pair(history)
                if p.sum() > 0:
                    ensemble_probs += (p / p.sum()) * weight
            except Exception:
                pass

        if ensemble_probs.sum() == 0:
            return np.ones(POOL) / POOL
        return ensemble_probs / ensemble_probs.sum()

    r_super = evaluate_algo(df, super_ensemble, "SUPER_ENSEMBLE")
    print(f"\nSUPER_ENSEMBLE: t10_5/5={r_super['top10_5of5']} t15={r_super['top15_5of5']} t20={r_super['top20_5of5']} t25={r_super['top25_5of5']} t30={r_super['top30_5of5']} t35={r_super['top35_5of5']} t40={r_super['top40_5of5']} t45={r_super['top45_5of5']}")
    print(f"  3+: t10={r_super['top10_3plus']}, t15={r_super['top15_3plus']}, t20={r_super['top20_3plus']}")
    results.append(r_super)

    df_res = pd.DataFrame(results)
    df_res.to_csv("reports/iter5_novel.csv", index=False)
    with open("reports/iter5_summary.json", "w") as f:
        json.dump([{k: int(v) if isinstance(v, np.integer) else v for k, v in r.items()} for r in results], f, indent=2, default=str)


if __name__ == "__main__":
    main()
