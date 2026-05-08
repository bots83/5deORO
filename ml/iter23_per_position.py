"""Iter 23: Predicción por POSICIÓN.

En lugar de predecir si cada número saldrá, predecimos:
- Posición 1 (más bajo): qué número será (clasificación 1-44)
- Posición 2: qué número será (1-45)
- Posición 3 (mediano)
- Posición 4
- Posición 5 (más alto)

Cada posición tiene su distribución específica. Esta puede ser ventaja.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import POOL, NUM_COLS
from ml.iter15_iterative_refinement import get_predictor_outputs
from ml.iter20_deep_nn import get_features_for_sorteo

import torch
import torch.nn as nn
import torch.optim as optim


class PerPositionNN(nn.Module):
    """5 cabezas, una por posición."""
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, POOL) for _ in range(5)
        ])

    def forward(self, x):
        z = self.shared(x)
        # 5 outputs, cada uno softmax over POOL
        return [torch.softmax(head(z), dim=-1) for head in self.heads]


def aggregate_per_position(probs_per_pos):
    """Combina las 5 probabilidades por posición en una sola por número."""
    # probs_per_pos: list of 5 arrays of shape (POOL,)
    # Probabilidad de un número = sum sobre posiciones
    combined = np.zeros(POOL)
    for pos_probs in probs_per_pos:
        combined += pos_probs
    return combined / 5


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    n_total = len(df)
    test_size = 50

    print("Building train data...")
    X_list = []
    y_pos_list = [[], [], [], [], []]  # 5 positions

    for idx in range(60, n_total - test_size):
        history = df.iloc[:idx]
        result = get_features_for_sorteo(history)
        if result is None:
            continue
        feat_vec, _ = result
        sorteo = df.iloc[idx]
        nums = sorted([int(sorteo[c]) for c in NUM_COLS])

        X_list.append(feat_vec)
        for pos in range(5):
            target = np.zeros(POOL)
            target[nums[pos] - 1] = 1
            y_pos_list[pos].append(target)

    X = np.array(X_list, dtype=np.float32)
    y_pos = [np.array(yp, dtype=np.float32) for yp in y_pos_list]
    print(f"  X: {X.shape}, y[0]: {y_pos[0].shape}")

    # Test
    X_test_list = []
    test_indices = []
    for idx in range(n_total - test_size, n_total):
        history = df.iloc[:idx]
        result = get_features_for_sorteo(history)
        if result is None:
            continue
        feat_vec, _ = result
        X_test_list.append(feat_vec)
        test_indices.append(idx)
    X_test = np.array(X_test_list, dtype=np.float32)

    # Normalizar
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std
    X_test_norm = (X_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X_norm, device=device)
    y_t_list = [torch.tensor(yp, device=device) for yp in y_pos]

    model = PerPositionNN(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCELoss()

    print("\nTraining PerPositionNN...")
    epochs = 100
    batch_size = 16
    n = X_t.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0
        for i in range(0, n, batch_size):
            batch = perm[i:i+batch_size]
            xb = X_t[batch]
            optimizer.zero_grad()
            preds = model(xb)
            loss = sum(criterion(preds[p], y_t_list[p][batch]) for p in range(5))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n:.4f}")

    # Eval
    model.eval()
    X_test_t = torch.tensor(X_test_norm, device=device)
    with torch.no_grad():
        preds_per_pos = model(X_test_t)  # list of 5, each (n_test, POOL)

    top_ks = list(range(5, 46))
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    for i in range(len(test_indices)):
        sorteo = df.iloc[test_indices[i]]
        real = {sorteo[f"n{j}"] for j in range(1, 6)}
        # Combinar probs de las 5 posiciones
        probs_combined = np.zeros(POOL)
        for p in range(5):
            probs_combined += preds_per_pos[p][i].cpu().numpy()
        probs_combined /= 5
        sorted_idx = np.argsort(probs_combined)[::-1]
        for k in top_ks:
            top_set = set((sorted_idx[:k] + 1).tolist())
            hits = len(real & top_set)
            if hits == 5: n_5of5[k] += 1
            if hits >= 4: n_4plus[k] += 1
            if hits >= 3: n_3plus[k] += 1

    print(f"\n{'Top-K':<8} {'5/5':>5} {'4+':>5} {'3+':>5}")
    for k in [10, 15, 20, 25, 30, 35, 40, 45]:
        print(f"top-{k:<4} {n_5of5[k]:>5} {n_4plus[k]:>5} {n_3plus[k]:>5}")

    print(f"\n🎯 K mínimo para 35/50:")
    for label, dic in [("5/5", n_5of5), ("4+", n_4plus), ("3+", n_3plus)]:
        for k in top_ks:
            if dic[k] >= 35:
                print(f"  {label} ≥ 35/50: Top-{k} ({dic[k]}/50 = {dic[k]*2}%) ✅")
                break

    with open("reports/iter23_per_position.json", "w") as f:
        out = {f"top{k}_5of5": n_5of5[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_4plus": n_4plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]} | \
              {f"top{k}_3plus": n_3plus[k] for k in [10, 15, 20, 25, 30, 35, 40, 45]}
        json.dump({k: int(v) if isinstance(v, np.integer) else v for k, v in out.items()},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
