"""Iter 8: Modelos profundos (LSTM-like + Transformer attention).

Usamos PyTorch si está disponible. Si no, usamos un Multi-Layer Perceptron de sklearn.
La idea: aprender patrones temporales que los modelos lineales no pueden capturar.
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.builder import build_features, _build_features_for_row, POOL, NUM_COLS

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("PyTorch no disponible, usando MLP de sklearn")
    from sklearn.neural_network import MLPClassifier
    from sklearn.multioutput import MultiOutputClassifier


def history_to_sequence(df, idx, window=20):
    """Convierte los últimos `window` sorteos antes de idx en una matriz binaria (window, POOL)."""
    n = idx
    seq = np.zeros((window, POOL))
    start = max(0, n - window)
    for i in range(start, n):
        nums = [int(df.iloc[i][c]) for c in NUM_COLS]
        for num in nums:
            seq[i - start, num - 1] = 1
    return seq


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=POOL, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, POOL)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :]))  # (B, POOL)


def train_lstm(df, last_n_train=150, window=20, epochs=30, batch_size=8, lr=0.001):
    """Entrena LSTM con sliding window."""
    n_total = len(df)
    start_train = max(60 + window, n_total - last_n_train - 50)  # entrenar hasta 50 antes del fin
    end_train = n_total - 50

    X = []
    y = []
    for idx in range(start_train, end_train):
        seq = history_to_sequence(df, idx, window=window)
        sorteo = df.iloc[idx]
        target = np.zeros(POOL)
        for n in [int(sorteo[c]) for c in NUM_COLS]:
            target[n-1] = 1
        X.append(seq)
        y.append(target)

    X = np.stack(X).astype(np.float32)
    y = np.stack(y).astype(np.float32)

    print(f"LSTM training: X shape={X.shape}, y shape={y.shape}")

    if not TORCH_OK:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    model = SimpleLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n_samples = X_t.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i+batch_size]
            xb, yb = X_t[batch_idx], y_t[batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_idx.size(0)
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_samples:.4f}")

    return model


def evaluate_lstm(df, model, last_n=50, window=20):
    if model is None:
        return None
    n_total = len(df)
    start = n_total - last_n

    device = next(model.parameters()).device
    model.eval()
    top_ks = [10, 15, 20, 25, 30, 35, 40]
    n_5of5 = {k: 0 for k in top_ks}
    n_4plus = {k: 0 for k in top_ks}
    n_3plus = {k: 0 for k in top_ks}

    with torch.no_grad():
        for idx in range(start, n_total):
            seq = history_to_sequence(df, idx, window=window)
            x = torch.tensor(seq, device=device).unsqueeze(0).float()
            probs = model(x).cpu().numpy()[0]

            sorteo = df.iloc[idx]
            real = {sorteo[f"n{i}"] for i in range(1, 6)}
            sorted_idx = np.argsort(probs)[::-1]
            for k in top_ks:
                top_set = set((sorted_idx[:k] + 1).tolist())
                hits = len(real & top_set)
                if hits == 5: n_5of5[k] += 1
                if hits >= 4: n_4plus[k] += 1
                if hits >= 3: n_3plus[k] += 1

    return {
        **{f"top{k}_5of5": n_5of5[k] for k in top_ks},
        **{f"top{k}_4plus": n_4plus[k] for k in top_ks},
        **{f"top{k}_3plus": n_3plus[k] for k in top_ks},
    }


def main():
    df = pd.read_csv("data/processed/sorteos.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    print(f"Dataset: {len(df)} sorteos\n")

    # Probar diferentes windows
    results = []
    for window in [10, 20, 30, 50]:
        print(f"\n=== LSTM con window={window} ===")
        model = train_lstm(df, window=window, epochs=30, lr=0.001)
        if model is None:
            print("  No se pudo entrenar (PyTorch no disponible)")
            continue

        r = evaluate_lstm(df, model, window=window)
        r["window"] = window
        results.append(r)

        print(f"\nResultados LSTM(window={window}):")
        for k in [10, 15, 20, 25, 30, 35, 40]:
            n = r[f"top{k}_5of5"]
            check = "✅" if n >= 35 else f"❌"
            print(f"  Top-{k}: {n}/50 con 5/5 hits {check}")
        print(f"  Top-10 con 3+: {r['top10_3plus']}")
        print(f"  Top-15 con 3+: {r['top15_3plus']}")

    if results:
        with open("reports/iter8_lstm.json", "w") as f:
            json.dump([{k: int(v) if isinstance(v, np.integer) else v for k, v in r.items()}
                       for r in results], f, indent=2, default=str)


if __name__ == "__main__":
    main()
