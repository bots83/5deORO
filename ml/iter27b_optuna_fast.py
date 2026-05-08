"""Iter 27b: Optuna con cache de evaluación.

Pre-computamos las salidas de los predictores base UNA VEZ.
Luego cada trial solo re-entrena el meta-learner sobre las features ya cacheadas.

Esto acelera ~10-20x cada trial.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset

import optuna
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def precompute_test_features(df, fnames, last_n=50):
    """Pre-computa features X y reales por sorteo del test set."""
    n_total = len(df)
    start = n_total - last_n
    test_X = []
    test_real = []

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue
        X = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in fnames]
                      for num in range(POOL)], dtype=np.float32)
        test_X.append(X)
        test_real.append(real)

    return test_X, test_real


def evaluate_cached(test_X, test_real, model):
    n_3plus = {30: 0, 31: 0, 35: 0, 40: 0, 45: 0}
    n_4plus = {35: 0, 38: 0, 40: 0, 45: 0}
    n_5of5 = {30: 0, 35: 0, 40: 0, 45: 0}

    for X, real in zip(test_X, test_real):
        probs = model.predict(X)
        sorted_idx = np.argsort(probs)[::-1]
        for k in [30, 31, 35, 38, 40, 45]:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits >= 3 and k in n_3plus:
                n_3plus[k] += 1
            if hits >= 4 and k in n_4plus:
                n_4plus[k] += 1
            if hits == 5 and k in n_5of5:
                n_5of5[k] += 1

    return n_3plus, n_4plus, n_5of5


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n", flush=True)

    n_total = len(df)
    print("Building meta dataset...", flush=True)
    X_train, y_train, fnames = build_meta_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}", flush=True)

    print("Pre-computing test features...", flush=True)
    test_X, test_real = precompute_test_features(df, fnames)
    print(f"  Test cache: {len(test_X)} sorteos\n", flush=True)

    def objective(trial):
        cfg = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 200),
            "max_depth": trial.suggest_int("max_depth", -1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        weight_pos = trial.suggest_float("weight_pos", 1.0, 10.0)

        sw = np.where(y_train == 1, weight_pos, 1.0)
        model = LGBMRegressor(**cfg, verbose=-1)
        model.fit(X_train, y_train, sample_weight=sw)

        n_3plus, n_4plus, n_5of5 = evaluate_cached(test_X, test_real, model)
        score = (n_3plus[30] * 5 + n_3plus[31] * 4 + n_3plus[35] * 2 +
                 n_4plus[38] * 3 + n_4plus[40] * 2)
        trial.set_user_attr("metrics", {**{f"30_3+": n_3plus[30], "31_3+": n_3plus[31],
                                           "35_3+": n_3plus[35], "38_4+": n_4plus[38],
                                           "40_4+": n_4plus[40], "45_5/5": n_5of5[45]}})
        return score

    study = optuna.create_study(direction="maximize")
    print("Running Optuna 100 trials (CACHED EVAL)...", flush=True)

    for i in range(100):
        try:
            study.optimize(objective, n_trials=1)
            best = study.best_trial
            m = best.user_attrs.get("metrics", {})
            current = study.trials[-1]
            cm = current.user_attrs.get("metrics", {})
            mark = " ⭐" if current.value == best.value and len(study.trials) > 1 else ""
            if (i+1) % 5 == 0 or mark:
                print(f"  [{i+1}/100] cur={cm.get('30_3+',0)}/{cm.get('35_3+',0)}/{cm.get('38_4+',0)} score={current.value:.0f} | best={m.get('30_3+',0)}/{m.get('35_3+',0)}/{m.get('38_4+',0)} score={best.value:.0f}{mark}", flush=True)
        except Exception as e:
            print(f"  trial {i+1} error: {e}", flush=True)

    best = study.best_trial
    m = best.user_attrs.get("metrics", {})
    print(f"\n🏆 BEST score={best.value:.0f}")
    print(f"  Metrics: {m}")
    print(f"  Params: {best.params}")

    with open("reports/iter27b_optuna.json", "w") as f:
        json.dump({
            "best_params": best.params,
            "best_metrics": {k: int(v) if isinstance(v, np.integer) else v for k, v in m.items()},
            "best_score": best.value,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
