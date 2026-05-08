"""Iter 20: Neural Network profundo con TODAS las features.

Arquitectura: features de 25 (los predictores + agregadas) → NN profundo → 48 outputs (probs).

A diferencia del meta-learner LightGBM (que predice por número), aquí
predecimos los 48 números simultáneamente con interacciones aprendidas.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs

import torch
import torch.nn as nn
import torch.optim as optim


def get_features_for_sorteo(history):
    """Genera vector de features (concatenando outputs de predictores)."""
    outputs = get_predictor_outputs(history)
    if not outputs:
        return None
    # Para cada predictor, concatenamos sus 48 probabilidades
    feature_names = sorted(outputs.keys())
    features = []
    for name in feature_names:
        features.extend(outputs[name].tolist())
    return np.array(features, dtype=np.float32), feature_names


class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, POOL),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    n_total = len(df)
    test_size = 50

    print("Building train data...")
    X_list = []
    y_list = []
    for idx in range(60, n_total - test_size):
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
    print(f"  X: {X.shape}, y: {y.shape}")

    X_test_list = []
    y_test_list = []
    test_indices = []
    for idx in range(n_total - test_size, n_total):
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
        X_test_list.append(feat_vec)
        y_test_list.append(target)
        test_indices.append(idx)

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    # Normalizar features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std
    X_test_norm = (X_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_t = torch.tensor(X_norm, device=device)
    y_t = torch.tensor(y, device=device)

    model = DeepNN(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCELoss()

    print("\nTraining DeepNN...")
    epochs = 100
    batch_size = 16
    best_loss = float("inf")
    n = X_t.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0
        for i in range(0, n, batch_size):
            batch = perm[i:i+batch_size]
            xb = X_t[batch]
            yb = y_t[batch]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= n
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    print(f"\nFinal loss: {best_loss:.4f}")

    # Evaluación
    model.eval()
    X_test_t = torch.tensor(X_test_norm, device=device)
    with torch.no_grad():
        probs_all = model(X_test_t).cpu().numpy()

    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for i in range(len(probs_all)):
        sorteo = df.iloc[test_indices[i]]
        real = {sorteo[f"n{j}"] for j in range(1, 6)}
        sorted_idx = np.argsort(probs_all[i])[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    print("\nResultados DeepNN:")
    print(f"{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in top_ks:
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    with open("reports/iter20_deepnn.json", "w") as f:
        out = {f"top{k}_5of5": n_5of5[k] for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]} | \
              {f"top{k}_4plus": n_4plus[k] for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]} | \
              {f"top{k}_3plus": n_3plus[k] for k in [10, 15, 20, 25, 30, 31, 35, 40, 45]}
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in out.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
