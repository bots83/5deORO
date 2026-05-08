"""Tests unitarios para verificar correctitud del pipeline."""
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.builder import build_features, get_feature_cols, get_target_cols, NUM_COLS, POOL, DRAW_SIZE
from ml.evaluator import top_k_recall, random_baseline_expected, monte_carlo_baseline


def make_synthetic_uniform(n_sorteos: int = 200, seed: int = 42) -> pd.DataFrame:
    """Genera sorteos sintéticos perfectamente uniformes 1-45 sin reemplazo."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sorteos):
        nums = sorted(rng.choice(POOL, DRAW_SIZE, replace=False) + 1)
        rows.append({
            "draw_num": i,
            "fecha": date(2010, 1, 1) + pd.Timedelta(days=i * 7),
            "dia_semana": "sabado",
            "n1": nums[0], "n2": nums[1], "n3": nums[2],
            "n4": nums[3], "n5": nums[4], "n6": nums[5],
            "supp1": None, "supp2": None,
            "fuente": "synthetic",
        })
    return pd.DataFrame(rows)


def test_zero_leakage():
    """Las features para sorteo t no deben usar datos de t+."""
    df = make_synthetic_uniform(100)
    features_df = build_features(df, min_history=20)

    assert len(features_df) > 0, "No se generaron features"

    # Verificar que la primera fila construida tenga features que solo dependen de los primeros 20 sorteos
    first_row = features_df.iloc[0]
    fecha_actual = first_row.name

    # La fecha actual NO debe estar incluida en el cálculo de freq_hist
    # Si está incluida, los conteos serían diferentes
    history = df[pd.to_datetime(df["fecha"]) < fecha_actual]
    expected_freq_1 = (history[NUM_COLS] == 1).any(axis=1).sum() / len(history)
    actual_freq_1 = first_row["freq_hist_1"]

    assert abs(expected_freq_1 - actual_freq_1) < 1e-6, (
        f"Zero leakage violado: freq_hist_1 esperado={expected_freq_1}, "
        f"actual={actual_freq_1}"
    )
    print(f"  ✓ Zero leakage en freq_hist: esperado={expected_freq_1:.4f}, actual={actual_freq_1:.4f}")


def test_target_correctness():
    """Los targets deben corresponder exactamente a los números del sorteo."""
    df = make_synthetic_uniform(100)
    features_df = build_features(df, min_history=20)

    # Tomar el primer sorteo y verificar
    first_idx = 20  # primero después de min_history
    sorteo = df.iloc[first_idx]
    nums_sorteados = set([sorteo[c] for c in NUM_COLS])

    fecha_actual = pd.to_datetime(sorteo["fecha"])
    feat_row = features_df.loc[fecha_actual]

    # Verificar todos los targets
    for num in range(1, POOL + 1):
        expected = 1 if num in nums_sorteados else 0
        actual = feat_row[f"target_{num}"]
        assert expected == actual, f"target_{num}: esperado={expected}, actual={actual}"

    # Suma de targets debe ser exactamente DRAW_SIZE
    target_sum = sum(feat_row[f"target_{n}"] for n in range(1, POOL + 1))
    assert target_sum == DRAW_SIZE, f"Suma de targets={target_sum} != {DRAW_SIZE}"
    print(f"  ✓ Target correcto: suma={target_sum}, números={sorted(nums_sorteados)}")


def test_random_baseline_convergence():
    """El recall aleatorio en suficientes simulaciones debe converger al esperado teórico."""
    n_sorteos = 1000
    n_sims = 5000
    for k in [6, 10, 15]:
        mc = monte_carlo_baseline(k=k, n_sorteos=n_sorteos, n_sims=n_sims)
        expected, _ = random_baseline_expected(k)
        diff = abs(mc["media"] - expected)
        assert diff < 0.01, f"K={k}: MC={mc['media']:.4f} vs teórico={expected:.4f}"
        print(f"  ✓ K={k}: MC={mc['media']:.4f}, teórico={expected:.4f}, diff={diff:.4f}")


def test_top_k_recall_perfect():
    """Si la predicción es perfecta, top-K recall debe ser DRAW_SIZE para K>=DRAW_SIZE."""
    n = 100
    y_true = np.zeros((n, POOL), dtype=np.int32)
    y_prob = np.zeros((n, POOL))

    rng = np.random.default_rng(42)
    for i in range(n):
        nums = rng.choice(POOL, DRAW_SIZE, replace=False)
        y_true[i, nums] = 1
        # Predicción perfecta
        y_prob[i, nums] = 1.0
        y_prob[i, [j for j in range(POOL) if j not in nums]] = 0.0

    for k in [6, 10, 15]:
        recall = top_k_recall(y_true, y_prob, k=k)
        assert abs(recall - DRAW_SIZE) < 1e-6, f"K={k}: recall={recall} != {DRAW_SIZE}"
        print(f"  ✓ Predicción perfecta K={k}: recall={recall} = {DRAW_SIZE}")


def test_target_count_correct():
    """En cualquier features_df válido, todos los targets deben sumar exactamente DRAW_SIZE."""
    df = make_synthetic_uniform(200)
    features_df = build_features(df, min_history=30)

    target_cols = get_target_cols(features_df)
    sums = features_df[target_cols].sum(axis=1)

    assert (sums == DRAW_SIZE).all(), f"Sumas de targets: {sums.unique()}"
    print(f"  ✓ Todos los {len(features_df)} sorteos tienen exactamente {DRAW_SIZE} targets=1")


def test_no_nan_in_features():
    df = make_synthetic_uniform(200)
    features_df = build_features(df, min_history=30)
    feat_cols = get_feature_cols(features_df)

    nans = features_df[feat_cols].isnull().sum().sum()
    assert nans == 0, f"NaN en features: {nans}"
    print(f"  ✓ Sin NaN en {len(feat_cols)} features × {len(features_df)} sorteos")


def run_all_tests():
    print("Ejecutando tests del pipeline...\n")
    tests = [
        test_zero_leakage,
        test_target_correctness,
        test_random_baseline_convergence,
        test_top_k_recall_perfect,
        test_target_count_correct,
        test_no_nan_in_features,
    ]
    for test in tests:
        print(f"[{test.__name__}]")
        test()
        print()
    print("✓ Todos los tests pasaron")


if __name__ == "__main__":
    run_all_tests()
