"""Pipeline de entrenamiento con walk-forward CV + simulaciones Monte Carlo.

Para cada modelo, evalúa:
1. Top-K recall promedio en walk-forward CV
2. Comparación con baseline aleatorio Monte Carlo (10,000 simulaciones)
3. Z-score y p-value para evaluar significancia estadística
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, get_feature_cols, get_target_cols
from ml.dataset import temporal_split, walk_forward_splits, load_features
from ml.evaluator import (
    evaluate_model,
    compare_models,
    random_baseline_expected,
    monte_carlo_baseline,
    top_k_recall,
    hit_rate_distribution,
)
from ml.baseline import FrequencyBaseline, RandomBaseline
from ml.xgboost_model import XGBoostMultilabel
from ml.random_forest import RandomForestMultilabel
from ml.lightgbm_model import LightGBMMultilabel
from ml.bayesian import BetaBinomialModel

REPORTS_DIR = Path(__file__).parent.parent / "reports"
K_VALUES = [5, 10, 15, 20]


def make_models(decay: float = 1.0):
    return [
        ("Random", RandomBaseline()),
        ("Frecuencia histórica", FrequencyBaseline()),
        (f"BetaBinomial (decay={decay})", BetaBinomialModel(decay=decay)),
        ("Random Forest", RandomForestMultilabel(n_estimators=200, max_depth=6)),
        ("XGBoost", XGBoostMultilabel(max_depth=4, n_estimators=200, learning_rate=0.05)),
        ("LightGBM", LightGBMMultilabel(max_depth=5, n_estimators=200, learning_rate=0.05)),
    ]


def evaluate_walk_forward(
    features_df: pd.DataFrame,
    initial_train: int = 500,
    step: int = 50,
    k_values: list[int] = None,
) -> pd.DataFrame:
    """
    Walk-forward CV: para cada fold, entrena cada modelo y registra top-k recall.
    Retorna un DataFrame con resultados por fold y por modelo.
    """
    if k_values is None:
        k_values = K_VALUES

    folds = list(walk_forward_splits(features_df, initial_train=initial_train, step=step))
    print(f"Walk-forward CV: {len(folds)} folds, train inicial={initial_train}, step={step}")

    all_results = []
    for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"\n  Fold {fold_idx+1}/{len(folds)} — train={len(X_train)}, test={len(X_test)}")

        for label, model in make_models():
            try:
                t0 = time.time()
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)
                t1 = time.time()
                row = {"fold": fold_idx, "modelo": label, "tiempo": round(t1 - t0, 2)}
                for k in k_values:
                    row[f"top{k}_recall"] = round(top_k_recall(y_test, y_prob, k=k), 4)
                all_results.append(row)
            except Exception as e:
                print(f"    [{label}] error: {e}")

    return pd.DataFrame(all_results)


def evaluate_single_split(
    features_df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """
    Single split clásico + Monte Carlo baseline para comparar.
    Retorna (comparison_df, monte_carlo_dict).
    """
    X_train, X_test, y_train, y_test = temporal_split(features_df, test_fraction=test_fraction)
    n_test = len(X_test)
    print(f"\nSingle split: train={len(X_train)}, test={n_test}")

    print(f"\nMonte Carlo baseline ({n_test} sorteos × 10,000 simulaciones)...")
    mc_results = {}
    for k in K_VALUES:
        mc = monte_carlo_baseline(k=k, n_sorteos=n_test, n_sims=10_000)
        mc_results[k] = mc
        print(f"  K={k}: media={mc['media']:.4f} ± {mc['std']:.4f}, "
              f"95% CI=[{mc['p2.5']:.4f}, {mc['p97.5']:.4f}]")

    rows = []
    for label, model in make_models():
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)
            t1 = time.time()

            row = {"modelo": label, "tiempo_s": round(t1 - t0, 2)}
            for k in K_VALUES:
                recall = top_k_recall(y_test, y_prob, k=k)
                mc = mc_results[k]
                row[f"top{k}_recall"] = round(recall, 4)
                row[f"top{k}_mc_z"] = round(
                    (recall - mc["media"]) / mc["std"] if mc["std"] > 0 else 0, 3
                )
                # P-value empírico unilateral (¿qué tan extremo es?)
                # Aproximación normal usando MC
                from scipy import stats as scistats
                z = (recall - mc["media"]) / mc["std"] if mc["std"] > 0 else 0
                p = 1 - scistats.norm.cdf(z)
                row[f"top{k}_p_unilateral"] = round(p, 4)
            rows.append(row)
        except Exception as e:
            print(f"  [{label}] error: {e}")

    return pd.DataFrame(rows).set_index("modelo"), mc_results


def run(input_csv: str, output_dir: str, walk_forward: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_csv = Path("data/processed/features.csv")
    if features_csv.exists():
        print("Cargando features...")
        features_df = load_features(str(features_csv))
    else:
        print("Construyendo features...")
        df = pd.read_csv(input_csv)
        features_df = build_features(df, min_history=50)
        features_df.to_csv(features_csv)

    feat_cols = get_feature_cols(features_df)
    target_cols = get_target_cols(features_df)
    print(f"Features dataset: {len(features_df)} sorteos, {len(feat_cols)} features, {len(target_cols)} targets")

    # 1. Single split + Monte Carlo
    print("\n" + "=" * 70)
    print("FASE 1: Single split + Monte Carlo baseline")
    print("=" * 70)
    single_results, mc_results = evaluate_single_split(features_df)
    single_results.to_csv(output_dir / "single_split_results.csv")

    print("\n--- RESULTADOS SINGLE SPLIT ---")
    print(single_results.to_string())

    # 2. Walk-forward CV
    if walk_forward:
        print("\n" + "=" * 70)
        print("FASE 2: Walk-forward cross-validation")
        print("=" * 70)
        wf_results = evaluate_walk_forward(features_df, initial_train=500, step=50)
        wf_results.to_csv(output_dir / "walk_forward_results.csv", index=False)

        # Agregado por modelo
        wf_summary = wf_results.groupby("modelo").agg({
            f"top{k}_recall": ["mean", "std"] for k in K_VALUES
        })
        wf_summary.columns = ["_".join(c) for c in wf_summary.columns]
        wf_summary.to_csv(output_dir / "walk_forward_summary.csv")

        print("\n--- RESULTADOS WALK-FORWARD CV (medias por modelo) ---")
        cols_show = [c for c in wf_summary.columns if "mean" in c]
        print(wf_summary[cols_show].to_string())

    # 3. Monte Carlo baseline detallado
    print("\n" + "=" * 70)
    print("FASE 3: Significancia estadística vs aleatoriedad")
    print("=" * 70)
    print("\nDistribución del baseline aleatorio (Monte Carlo):")
    for k in K_VALUES:
        mc = mc_results[k]
        thr = mc["p95"]
        print(f"  K={k}: para ser significativo (p<0.05), recall debe superar {thr:.4f}")

    print("\n¿Algún modelo supera el umbral de significancia?")
    significant = []
    for k in K_VALUES:
        thr = mc_results[k]["p95"]
        col = f"top{k}_recall"
        for modelo in single_results.index:
            if modelo == "Random":
                continue
            recall = single_results.loc[modelo, col]
            if recall > thr:
                significant.append((modelo, k, recall, thr))

    if significant:
        print("\n  ✓ MODELOS SIGNIFICATIVOS (p<0.05 vs aleatorio):")
        for m, k, r, t in significant:
            print(f"    [{m}] K={k}: recall={r:.4f} > umbral={t:.4f}")
    else:
        print("\n  ✗ NINGÚN modelo supera el umbral de significancia 95%.")
        print("    Conclusión: los modelos NO predicen mejor que el azar.")

    # 4. Distribución de hits del mejor modelo
    print("\n" + "=" * 70)
    print("FASE 4: Distribución de hits del mejor modelo (XGBoost)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = temporal_split(features_df)
    xgb = XGBoostMultilabel(max_depth=4, n_estimators=200, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    y_prob = xgb.predict_proba(X_test)
    for k in K_VALUES:
        dist = hit_rate_distribution(y_test, y_prob, k=k)
        print(f"\n  K={k} - distribución de hits por sorteo:")
        for hits, count in dist["distribucion"].items():
            pct = count / len(y_test) * 100
            print(f"    {hits} hits: {count} sorteos ({pct:.1f}%)")

    print(f"\n✓ Reportes guardados en {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--no-walk-forward", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, walk_forward=not args.no_walk_forward)
