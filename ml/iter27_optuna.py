"""Iter 27: Optimización bayesiana con Optuna.

Optuna busca inteligentemente en el espacio de hyperparams para maximizar
nuestra métrica directamente: # sorteos con ≥3 hits en top-30.
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


def evaluate_model(df, model, fnames, last_n=50):
    n_total = len(df)
    start = n_total - last_n
    n_3plus_30 = 0
    n_3plus_35 = 0
    n_4plus_38 = 0
    n_3plus_31 = 0

    for idx in range(start, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]
        outputs = get_predictor_outputs(history)
        if not outputs:
            continue
        X = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in fnames]
                      for num in range(POOL)], dtype=np.float32)
        probs = model.predict(X)
        sorted_idx = np.argsort(probs)[::-1]
        for k, dic_3, dic_4 in [(30, "n_3plus_30", None), (31, "n_3plus_31", None),
                                  (35, "n_3plus_35", None), (38, None, "n_4plus_38")]:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if dic_3 == "n_3plus_30" and hits >= 3: n_3plus_30 += 1
            if dic_3 == "n_3plus_31" and hits >= 3: n_3plus_31 += 1
            if dic_3 == "n_3plus_35" and hits >= 3: n_3plus_35 += 1
            if dic_4 == "n_4plus_38" and hits >= 4: n_4plus_38 += 1

    return {"30_3+": n_3plus_30, "31_3+": n_3plus_31, "35_3+": n_3plus_35, "38_4+": n_4plus_38}


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n", flush=True)

    n_total = len(df)
    print("Building meta dataset...", flush=True)
    X_train, y_train, fnames = build_meta_dataset(df, 60, n_total - 50)
    print(f"  Train: {X_train.shape}\n", flush=True)

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

        metrics = evaluate_model(df, model, fnames)
        # Score: priorizar top-30 con 3+, luego top-35, luego top-31
        score = metrics["30_3+"] * 5 + metrics["31_3+"] * 4 + metrics["35_3+"] * 2 + metrics["38_4+"]
        trial.set_user_attr("metrics", metrics)
        return score

    study = optuna.create_study(direction="maximize")
    print("Running Optuna 50 trials...", flush=True)

    for i in range(50):
        try:
            study.optimize(objective, n_trials=1)
            if (i+1) % 5 == 0 or len(study.trials) <= 5:
                best = study.best_trial
                m = best.user_attrs.get("metrics", {})
                print(f"  [{i+1}/50] BEST score={best.value:.0f}: t30_3+={m.get('30_3+',0)} t31_3+={m.get('31_3+',0)} t35_3+={m.get('35_3+',0)} t38_4+={m.get('38_4+',0)}", flush=True)
        except Exception as e:
            print(f"  trial {i+1} error: {e}", flush=True)

    best = study.best_trial
    m = best.user_attrs.get("metrics", {})
    print(f"\n🏆 BEST score={best.value:.0f}")
    print(f"  Metrics: {m}")
    print(f"  Params: {best.params}")

    with open("reports/iter27_optuna.json", "w") as f:
        json.dump({
            "best_params": best.params,
            "best_metrics": m,
            "best_score": best.value,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
