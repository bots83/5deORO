"""Mega Ensemble Final: combina los 3 mejores modelos (iter 20 + 21 + 23).

- Iter 20: DeepNN
- Iter 21: LightGBM Weighted (sample_weight=3.0 positivos)
- Iter 23: Per-Position NN

Promediamos las probabilidades.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs, build_meta_dataset
from ml.iter20_deep_nn import DeepNN, get_features_for_sorteo
from ml.iter23_per_position import PerPositionNN

import torch
import torch.nn as nn
import torch.optim as optim
from lightgbm import LGBMRegressor


WEIGHTED_CFG = {
    "num_leaves": 50, "max_depth": 5, "learning_rate": 0.03,
    "n_estimators": 250, "reg_alpha": 0.1, "reg_lambda": 0.1,
}


def train_lgbm_weighted(df, end_idx):
    X, y, fnames = build_meta_dataset(df, 60, end_idx)
    sw = np.where(y == 1, 3.0, 1.0)
    model = LGBMRegressor(**WEIGHTED_CFG, verbose=-1)
    model.fit(X, y, sample_weight=sw)
    return model, fnames


def predict_lgbm_weighted(model, fnames, history):
    outputs = get_predictor_outputs(history)
    if not outputs:
        return np.ones(POOL) / POOL
    X = np.array([[outputs.get(name, np.ones(POOL)/POOL)[num] for name in fnames]
                  for num in range(POOL)], dtype=np.float32)
    probs = model.predict(X)
    # Normalizar a [0, 1]
    probs = np.clip(probs, 0, 1)
    if probs.sum() > 0:
        probs = probs / probs.sum() * 5  # promedio 5
    return probs


def train_deepnn(df, end_idx, mean=None, std=None):
    X_list = []
    y_list = []
    for idx in range(60, end_idx):
        history = df.iloc[:idx]
        result = get_features_for_sorteo(history)
        if result is None:
            continue
        feat_vec, _ = result
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        target = np.zeros(POOL)
        for n in real:
            target[n - 1] = 1
        X_list.append(feat_vec)
        y_list.append(target)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_norm, device=device)
    y_t = torch.tensor(y, device=device)

    model = DeepNN(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCELoss()

    n = X_t.size(0)
    for epoch in range(80):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 16):
            batch = perm[i:i+16]
            optimizer.zero_grad()
            pred = model(X_t[batch])
            loss = criterion(pred, y_t[batch])
            loss.backward()
            optimizer.step()
    return model, (mean, std)


def predict_deepnn(model, mean_std, history):
    feat_vec, _ = get_features_for_sorteo(history)
    if feat_vec is None:
        return np.ones(POOL) / POOL
    mean, std = mean_std
    feat_norm = (feat_vec - mean) / std
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(feat_norm[None, :], device=device, dtype=torch.float32)
        probs = model(x).cpu().numpy()[0]
    return probs


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    n_total = len(df)
    test_size = 50
    train_end = n_total - test_size

    # Entrenar dos modelos
    print("Entrenando LightGBM Weighted...")
    lgbm_model, fnames = train_lgbm_weighted(df, train_end)
    print("Entrenando DeepNN...")
    deepnn_model, mean_std = train_deepnn(df, train_end)

    # Backtest combinando
    print("\nBacktest mega-ensemble...")
    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for idx in range(train_end, n_total):
        sorteo = df.iloc[idx]
        real = {sorteo[f"n{i}"] for i in range(1, 6)}
        history = df.iloc[:idx]

        # Combinar 50/50
        probs_lgbm = predict_lgbm_weighted(lgbm_model, fnames, history)
        probs_dnn = predict_deepnn(deepnn_model, mean_std, history)

        # Normalizar
        if probs_lgbm.sum() > 0:
            probs_lgbm = probs_lgbm / probs_lgbm.sum()
        if probs_dnn.sum() > 0:
            probs_dnn = probs_dnn / probs_dnn.sum()

        ensemble = 0.5 * probs_lgbm + 0.5 * probs_dnn
        sorted_idx = np.argsort(ensemble)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 31, 35, 38, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in top_ks:
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    with open("reports/mega_ensemble.json", "w") as f:
        out = {f"top{k}_5of5": n_5of5[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_4plus": n_4plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_3plus": n_3plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]}
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in out.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
