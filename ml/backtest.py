"""Back-testing exhaustivo del sistema de predicción.

Para cada modelo y combinación de hiperparámetros, ejecuta walk-forward CV
y compara con el baseline aleatorio Monte Carlo. Calibra los pesos del ensemble.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols, POOL, DRAW_SIZE
from ml.dataset import walk_forward_splits, temporal_split
from ml.evaluator import top_k_recall, monte_carlo_baseline, random_baseline_expected
from ml.baseline import FrequencyBaseline
from ml.bayesian import BetaBinomialModel
from ml.xgboost_model import XGBoostMultilabel
from ml.lightgbm_model import LightGBMMultilabel

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def evaluate_walk_forward_single_model(features_df, model_factory, name,
                                        initial_train=50, step=10, k_values=None):
    """Walk-forward CV para un solo modelo. Retorna recalls por fold."""
    if k_values is None:
        k_values = [5, 10, 15, 20]

    fold_results = []
    for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(walk_forward_splits(features_df, initial_train=initial_train, step=step)):
        try:
            model = model_factory()
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)
            row = {"modelo": name, "fold": fold_idx, "n_train": len(X_tr), "n_test": len(X_te)}
            for k in k_values:
                row[f"top{k}"] = top_k_recall(y_te, y_prob, k=k)
            fold_results.append(row)
        except Exception as e:
            print(f"    fold {fold_idx} error: {e}")
    return pd.DataFrame(fold_results)


def calibrate_ensemble_weights(features_df, base_models_factory, k=10,
                                initial_train=50, step=10) -> dict:
    """
    Calibra pesos del ensemble por inverse-variance:
    los modelos con menor varianza (más estables) y mayor recall reciben más peso.
    """
    print("Calibrando pesos del ensemble...")
    weights = {}
    performance = {}
    for name, factory in base_models_factory.items():
        df_results = evaluate_walk_forward_single_model(
            features_df, factory, name, initial_train=initial_train, step=step
        )
        if len(df_results) == 0:
            continue
        col = f"top{k}"
        recall_mean = df_results[col].mean()
        recall_std = df_results[col].std()
        random_expected, _ = random_baseline_expected(k)
        # Score: ratio sobre baseline penalizado por inestabilidad
        score = max(0.0, recall_mean - random_expected) / (recall_std + 0.1)
        performance[name] = {
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "vs_random": recall_mean / random_expected,
            "score": score,
            "n_folds": len(df_results),
        }

    # Pesos proporcionales al score (modelos malos = peso 0.1 mínimo)
    total_score = sum(max(p["score"], 0.1) for p in performance.values())
    for name, p in performance.items():
        weights[name] = max(p["score"], 0.1) / total_score

    return weights, performance


def monte_carlo_evaluation(model, X_test, y_test, n_runs=100, seed=42) -> dict:
    """Re-evalúa el modelo con perturbaciones aleatorias para medir varianza."""
    rng = np.random.default_rng(seed)
    recalls = {k: [] for k in [5, 10, 15, 20]}
    base_prob = model.predict_proba(X_test)
    for run in range(n_runs):
        # Añadir ruido pequeño a las probabilidades para medir robustez del top-K
        noise = rng.normal(0, 0.001, base_prob.shape)
        y_prob = np.clip(base_prob + noise, 0, 1)
        for k in recalls:
            recalls[k].append(top_k_recall(y_test, y_prob, k=k))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in recalls.items()}


def run_backtest(features_csv: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_df = pd.read_csv(features_csv, index_col="fecha", parse_dates=True)
    feat_cols = get_feature_cols(features_df)
    target_cols = get_target_cols(features_df)
    print(f"Dataset: {len(features_df)} sorteos, {len(feat_cols)} features")

    # Modelos para evaluar
    factories = {
        "Frecuencia": lambda: FrequencyBaseline(),
        "BetaBinomial decay=1.0": lambda: BetaBinomialModel(decay=1.0),
        "BetaBinomial decay=0.99": lambda: BetaBinomialModel(decay=0.99),
        "BetaBinomial decay=0.95": lambda: BetaBinomialModel(decay=0.95),
        "BetaBinomial decay=0.90": lambda: BetaBinomialModel(decay=0.90),
        "LightGBM": lambda: LightGBMMultilabel(max_depth=3, n_estimators=50, learning_rate=0.1),
        "XGBoost": lambda: XGBoostMultilabel(max_depth=3, n_estimators=50, learning_rate=0.1),
    }

    # 1. Calibrar pesos por walk-forward CV
    print("\n" + "=" * 70)
    print("FASE 1: Calibración de pesos vía walk-forward CV")
    print("=" * 70)
    weights, perf = calibrate_ensemble_weights(features_df, factories, k=10, initial_train=50, step=10)

    perf_df = pd.DataFrame(perf).T.reset_index().rename(columns={"index": "modelo"})
    perf_df["weight"] = perf_df["modelo"].map(weights)
    perf_df = perf_df.sort_values("score", ascending=False)
    perf_df.to_csv(output_dir / "model_calibration.csv", index=False)

    print("\n--- PERFORMANCE POR MODELO ---")
    cols_show = ["recall_mean", "recall_std", "vs_random", "score", "weight", "n_folds"]
    print(perf_df.set_index("modelo")[cols_show].to_string())

    # 2. Comparar contra Monte Carlo baseline
    print("\n" + "=" * 70)
    print("FASE 2: Significancia estadística (Monte Carlo baseline)")
    print("=" * 70)

    n_test_typical = 10  # tamaño de fold típico
    print(f"\nBaseline aleatorio (1000 simulaciones × 10 sorteos):")
    for k in [5, 10, 15]:
        mc = monte_carlo_baseline(k=k, n_sorteos=n_test_typical, n_sims=1000)
        exp, _ = random_baseline_expected(k)
        print(f"  K={k}: media={mc['media']:.3f} (teórico {exp:.3f}), 95% CI=[{mc['p2.5']:.3f}, {mc['p97.5']:.3f}]")

    # 3. Test final: entrenar con 80%, evaluar 20%
    print("\n" + "=" * 70)
    print("FASE 3: Test final 80/20")
    print("=" * 70)

    X_train, X_test, y_train, y_test = temporal_split(features_df)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Generar predicción ensemble
    all_probs = {}
    for name, factory in factories.items():
        try:
            model = factory()
            model.fit(X_train, y_train)
            all_probs[name] = model.predict_proba(X_test)
        except Exception as e:
            print(f"  [{name}] error: {e}")

    # Ensemble ponderado
    ensemble = np.zeros_like(list(all_probs.values())[0])
    total_w = sum(weights.get(n, 0) for n in all_probs)
    for name, probs in all_probs.items():
        w = weights.get(name, 0) / total_w
        ensemble += probs * w

    print("\n--- TOP-K RECALL EN TEST FINAL ---")
    rows = []
    for name in list(all_probs.keys()) + ["ENSEMBLE"]:
        probs = ensemble if name == "ENSEMBLE" else all_probs[name]
        row = {"modelo": name}
        for k in [5, 10, 15, 20]:
            recall = top_k_recall(y_test, probs, k=k)
            exp, _ = random_baseline_expected(k)
            row[f"top{k}"] = recall
            row[f"top{k}_vs_random"] = recall / exp
        rows.append(row)
    test_df = pd.DataFrame(rows).set_index("modelo")
    test_df.to_csv(output_dir / "final_test_results.csv")
    print(test_df[["top5", "top10", "top10_vs_random", "top15"]].to_string())

    return weights, perf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    weights, perf = run_backtest(args.features, args.output)
    import json
    with open(Path(args.output) / "ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\n✓ Pesos guardados en {Path(args.output) / 'ensemble_weights.json'}")
