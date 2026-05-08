"""Tests de aleatoriedad rigurosos para el dataset del 5 de Oro.

Si el juego es genuinamente aleatorio:
- Frecuencias deben ser uniformes (chi-cuadrado)
- Gaps entre apariciones deben seguir distribución geométrica
- No debe haber autocorrelación temporal
- Runs (rachas) deben tener longitud esperada
- Wald-Wolfowitz runs test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

POOL = 48
DRAW_SIZE = 5
NUM_COLS = ["n1", "n2", "n3", "n4", "n5"]


def _all_numbers(df: pd.DataFrame) -> np.ndarray:
    return df[NUM_COLS].values.flatten()


def _presence_matrix(df: pd.DataFrame) -> np.ndarray:
    """Matriz (n_sorteos, POOL) booleana — True si número apareció."""
    n = len(df)
    mat = np.zeros((n, POOL), dtype=bool)
    for i, row in df[NUM_COLS].iterrows():
        for v in row:
            if 1 <= v <= POOL:
                mat[i, v - 1] = True
    return mat


def chi_squared_uniform(df: pd.DataFrame, pool: int = POOL) -> dict:
    """Chi-cuadrado de uniformidad de frecuencias."""
    nums = _all_numbers(df)
    counts = np.array([(nums == n).sum() for n in range(1, pool + 1)])
    expected = len(nums) / pool
    chi2, p = stats.chisquare(counts, [expected] * pool)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "uniforme": p > 0.05,
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "esperado": float(expected),
        "df": pool - 1,
    }


def gap_test_geometric(df: pd.DataFrame, pool: int = POOL) -> dict:
    """
    Para cada número, los gaps entre apariciones deben seguir distribución
    geométrica con p = DRAW_SIZE/POOL.
    Test KS de la distribución empírica vs teórica.
    """
    presence = _presence_matrix(df)
    n_sorteos = len(df)
    p_aparicion = DRAW_SIZE / pool

    all_gaps = []
    for num_idx in range(pool):
        positions = np.where(presence[:, num_idx])[0]
        if len(positions) < 2:
            continue
        gaps = np.diff(positions)
        all_gaps.extend(gaps.tolist())

    if not all_gaps:
        return {"error": "no gaps"}

    all_gaps = np.array(all_gaps)
    # Para gaps discretos usamos chi² de bondad de ajuste a la geométrica.
    # Agrupamos en bins para que el conteo esperado por bin sea >= 5.
    max_bin = 30
    obs_bins = np.zeros(max_bin)
    for g in all_gaps:
        if 1 <= g < max_bin:
            obs_bins[int(g)] += 1
        else:
            obs_bins[max_bin - 1] += 1
    # P(gap=k) = (1-p)^(k-1) * p para k>=1
    expected_bins = np.zeros(max_bin)
    for k in range(1, max_bin):
        expected_bins[k] = (1 - p_aparicion) ** (k - 1) * p_aparicion * len(all_gaps)
    expected_bins[max_bin - 1] = len(all_gaps) - expected_bins[1:max_bin - 1].sum()

    # Tomar bins desde k=1 (excluir el 0). Combinar bins con expected < 5
    obs = obs_bins[1:].copy()
    exp = expected_bins[1:].copy()
    keep = exp >= 5
    obs_use = obs[keep]
    exp_use = exp[keep]
    # Reescalar exp para que sume igual que obs (necesario por scipy.chisquare)
    if obs_use.sum() > 0 and exp_use.sum() > 0:
        exp_use = exp_use * obs_use.sum() / exp_use.sum()
    if len(obs_use) > 1:
        chi2_stat, p_chi2 = stats.chisquare(obs_use, exp_use)
    else:
        chi2_stat, p_chi2 = float("nan"), float("nan")

    return {
        "chi2": float(chi2_stat),
        "p_value": float(p_chi2),
        "geometrico": p_chi2 > 0.05,
        "media_observada": float(all_gaps.mean()),
        "media_esperada": float(1 / p_aparicion),
        "n_gaps": int(len(all_gaps)),
    }


def autocorrelation_test(df: pd.DataFrame, max_lag: int = 30) -> dict:
    """
    Test de autocorrelación de la suma por sorteo.
    Si los sorteos son IID, no debe haber correlación significativa.
    """
    sumas = df[NUM_COLS].sum(axis=1).values
    from statsmodels.tsa.stattools import acf
    acf_vals, confint = acf(sumas, nlags=max_lag, alpha=0.05, fft=False)

    # Lag 0 siempre es 1, ignorar
    significant_lags = []
    for lag in range(1, len(acf_vals)):
        lower = confint[lag, 0] - acf_vals[lag]
        upper = confint[lag, 1] - acf_vals[lag]
        # confint excluyendo el ACF: si 0 no está en [lower, upper], es significativo
        if not (lower <= 0 <= upper):
            significant_lags.append(int(lag))

    # Test de Ljung-Box
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(sumas, lags=[10, 20, max_lag], return_df=True)

    return {
        "max_acf_excluding_lag0": float(max(abs(acf_vals[1:]))),
        "lags_significativos": significant_lags,
        "ljung_box_lag10_p": float(lb.iloc[0]["lb_pvalue"]),
        "ljung_box_lag20_p": float(lb.iloc[1]["lb_pvalue"]),
        "ljung_box_lagmax_p": float(lb.iloc[2]["lb_pvalue"]),
        "iid": all([
            lb.iloc[0]["lb_pvalue"] > 0.05,
            lb.iloc[1]["lb_pvalue"] > 0.05,
        ]),
    }


def runs_test(df: pd.DataFrame, num: int) -> dict:
    """
    Wald-Wolfowitz runs test para la secuencia de apariciones de `num`.
    Si las apariciones son IID, el número de runs debe ser ~ N(esperado, var).
    """
    presence = _presence_matrix(df)
    seq = presence[:, num - 1].astype(int)
    n = len(seq)
    n1 = int(seq.sum())  # apariciones
    n2 = n - n1          # ausencias
    if n1 == 0 or n2 == 0:
        return {"error": "secuencia constante"}

    # Contar runs
    runs = 1
    for i in range(1, n):
        if seq[i] != seq[i - 1]:
            runs += 1

    # Esperado bajo H0
    expected = (2 * n1 * n2) / n + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))
    z = (runs - expected) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "numero": num,
        "runs": runs,
        "expected": float(expected),
        "z": float(z),
        "p_value": float(p),
        "iid": p > 0.05,
    }


def runs_test_all(df: pd.DataFrame, pool: int = POOL) -> pd.DataFrame:
    results = [runs_test(df, num) for num in range(1, pool + 1)]
    return pd.DataFrame([r for r in results if "error" not in r])


def cooccurrence_chi2(df: pd.DataFrame, pool: int = POOL) -> dict:
    """
    Test si los pares (i,j) ocurren con la frecuencia esperada bajo independencia.
    Bajo H0 (IID): P(i,j juntos) = C(POOL-2, DRAW_SIZE-2)/C(POOL,DRAW_SIZE).
    """
    from math import comb
    n = len(df)
    mat = np.zeros((pool + 1, pool + 1), dtype=int)
    for _, row in df.iterrows():
        nums = [row[c] for c in NUM_COLS]
        for i_pair in range(len(nums)):
            for j_pair in range(i_pair + 1, len(nums)):
                a, b = sorted([nums[i_pair], nums[j_pair]])
                mat[a][b] += 1

    cooc_pairs = []
    for a in range(1, pool + 1):
        for b in range(a + 1, pool + 1):
            cooc_pairs.append(mat[a][b])
    cooc_pairs = np.array(cooc_pairs)

    expected_p = comb(pool - 2, DRAW_SIZE - 2) / comb(pool, DRAW_SIZE)
    expected_count = expected_p * n

    chi2_stat = ((cooc_pairs - expected_count) ** 2 / expected_count).sum()
    df_chi = len(cooc_pairs) - 1
    p = 1 - stats.chi2.cdf(chi2_stat, df_chi)

    return {
        "chi2": float(chi2_stat),
        "df": int(df_chi),
        "p_value": float(p),
        "independientes": p > 0.05,
        "media_observada": float(cooc_pairs.mean()),
        "media_esperada": float(expected_count),
        "min_observado": int(cooc_pairs.min()),
        "max_observado": int(cooc_pairs.max()),
    }


def monte_carlo_simulation(n_sorteos: int, n_sims: int = 1000,
                          pool: int = POOL, draw_size: int = DRAW_SIZE,
                          seed: int = 42) -> pd.DataFrame:
    """
    Simula `n_sims` historias de `n_sorteos` sorteos cada una bajo H0 (IID uniforme).
    Para cada simulación, calcula estadísticos clave para comparar con el observado.
    """
    rng = np.random.default_rng(seed)
    stats_list = []
    for sim_id in range(n_sims):
        # Generar n_sorteos sorteos
        all_nums = []
        for _ in range(n_sorteos):
            all_nums.extend(rng.choice(pool, draw_size, replace=False) + 1)
        all_nums = np.array(all_nums)
        counts = np.bincount(all_nums, minlength=pool + 1)[1:]
        # chi2
        expected = len(all_nums) / pool
        chi2 = ((counts - expected) ** 2 / expected).sum()
        stats_list.append({
            "sim_id": sim_id,
            "chi2": chi2,
            "min_freq": int(counts.min()),
            "max_freq": int(counts.max()),
            "std_freq": float(counts.std()),
        })
    return pd.DataFrame(stats_list)


def run_all_tests(df: pd.DataFrame) -> dict:
    """Ejecuta todos los tests y devuelve un resumen."""
    return {
        "chi_squared_uniform": chi_squared_uniform(df),
        "gap_test_geometric": gap_test_geometric(df),
        "autocorrelation_test": autocorrelation_test(df),
        "cooccurrence_chi2": cooccurrence_chi2(df),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--output", default="reports/randomness_tests.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Dataset: {len(df)} sorteos")
    print(f"Numbers per draw: {DRAW_SIZE}, Pool: 1-{POOL}")
    print()

    results = run_all_tests(df)

    print("=" * 70)
    print("TESTS DE ALEATORIEDAD RIGUROSOS")
    print("=" * 70)

    chi = results["chi_squared_uniform"]
    print(f"\n[1] Chi-cuadrado de uniformidad de frecuencias:")
    print(f"    chi²={chi['chi2']:.2f} (df={chi['df']}), p={chi['p_value']:.4f}")
    print(f"    Conclusión: {'UNIFORME ✓' if chi['uniforme'] else 'NO UNIFORME ✗'}")

    gap = results["gap_test_geometric"]
    if "error" not in gap:
        print(f"\n[2] Gap test (distribución geométrica):")
        print(f"    chi²={gap['chi2']:.2f}, p={gap['p_value']:.4f}")
        print(f"    Media gaps: {gap['media_observada']:.2f} (esperado: {gap['media_esperada']:.2f})")
        print(f"    Conclusión: {'GEOMÉTRICO ✓' if gap['geometrico'] else 'NO GEOMÉTRICO ✗'}")

    auto = results["autocorrelation_test"]
    print(f"\n[3] Test de autocorrelación (suma por sorteo):")
    print(f"    Max |ACF| (lag>0): {auto['max_acf_excluding_lag0']:.4f}")
    print(f"    Ljung-Box p (lag=10): {auto['ljung_box_lag10_p']:.4f}")
    print(f"    Ljung-Box p (lag=20): {auto['ljung_box_lag20_p']:.4f}")
    print(f"    Lags significativos: {auto['lags_significativos']}")
    print(f"    Conclusión: {'IID ✓' if auto['iid'] else 'NO IID ✗'}")

    cooc = results["cooccurrence_chi2"]
    print(f"\n[4] Test de coocurrencia (pares independientes):")
    print(f"    chi²={cooc['chi2']:.2f}, p={cooc['p_value']:.4f}")
    print(f"    Media coocurrencia: {cooc['media_observada']:.2f} (esperado: {cooc['media_esperada']:.2f})")
    print(f"    Conclusión: {'INDEPENDIENTES ✓' if cooc['independientes'] else 'DEPENDIENTES ✗'}")

    # Runs test (resumen)
    runs_df = runs_test_all(df)
    n_significant = (runs_df["p_value"] < 0.05).sum()
    print(f"\n[5] Wald-Wolfowitz runs test (por número):")
    print(f"    Números con p<0.05: {n_significant}/{len(runs_df)} (esperado bajo H0: ~{int(0.05*POOL)})")

    # Guardar
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    runs_df.to_csv(Path(args.output).parent / "runs_test_per_number.csv", index=False)
    print(f"\nResultados guardados en {args.output}")
