"""Feature engineering con zero leakage para el verdadero 5 de Oro.

Sistema oficial: 5 números del 1-48 sin reemplazo + 1 Bolilla Extra del 1-48.
Para cada sorteo t, todas las features se calculan usando datos hasta t-1.
Target: 48 columnas binarias (1 si el número salió en t, 0 si no).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

POOL = 48
DRAW_SIZE = 5
NUMS = list(range(1, POOL + 1))
NUM_COLS = ["n1", "n2", "n3", "n4", "n5"]


def _build_features_for_row(history: pd.DataFrame) -> dict:
    n = len(history)
    if n == 0:
        return {}

    feats = {}
    all_nums = history[NUM_COLS].values  # (n, 5)

    for num in NUMS:
        presence = (all_nums == num).any(axis=1)

        feats[f"freq_hist_{num}"] = presence.sum() / n

        for window in [10, 20, 50]:
            w = min(window, n)
            feats[f"freq_{window}_{num}"] = presence[-w:].sum() / w

        positions = np.where(presence)[0]
        if len(positions) == 0:
            feats[f"ausencia_{num}"] = float(n)
            feats[f"ciclo_{num}"] = float(n)
        else:
            feats[f"ausencia_{num}"] = float(n - 1 - positions[-1])
            if len(positions) >= 2:
                diffs = np.diff(positions)
                feats[f"ciclo_{num}"] = float(diffs.mean())
            else:
                feats[f"ciclo_{num}"] = float(n)

        feats[f"last_{num}"] = float(presence[-1])
        if n >= 2:
            feats[f"last2_{num}"] = float(presence[-2:].any())
        else:
            feats[f"last2_{num}"] = float(presence[-1])

    # Sorteo anterior
    prev = history.iloc[-1]
    prev_nums = [prev[c] for c in NUM_COLS]
    feats["suma_prev"] = float(sum(prev_nums))
    feats["paridad_prev"] = float(sum(1 for x in prev_nums if x % 2 == 0))
    feats["rango_prev"] = float(max(prev_nums) - min(prev_nums))

    # Decenas en sorteo anterior
    for dec_lo, dec_hi, label in [(1, 12, "d1_12"), (13, 24, "d13_24"),
                                    (25, 36, "d25_36"), (37, 48, "d37_48")]:
        feats[f"prev_{label}"] = float(sum(1 for x in prev_nums if dec_lo <= x <= dec_hi))

    feats["sorteo_num"] = float(n)
    last_date = pd.to_datetime(history["fecha"].iloc[-1])
    feats["mes"] = float(last_date.month)
    feats["dia_mes"] = float(last_date.day)
    feats["dia_semana_num"] = float(last_date.dayofweek)

    return feats


def build_features(df: pd.DataFrame, min_history: int = 30) -> pd.DataFrame:
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    rows = []
    for idx in range(min_history, len(df)):
        history = df.iloc[:idx]
        feats = _build_features_for_row(history)
        if not feats:
            continue

        sorteo_actual = df.iloc[idx]
        nums_actuales = set([sorteo_actual[c] for c in NUM_COLS])
        for num in NUMS:
            feats[f"target_{num}"] = 1 if num in nums_actuales else 0

        feats["fecha"] = sorteo_actual["fecha"]
        rows.append(feats)

    return pd.DataFrame(rows).set_index("fecha")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]


def get_target_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("target_")]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--min-history", type=int, default=30)
    args = parser.parse_args()

    print(f"Construyendo features (min_history={args.min_history})...")
    df = pd.read_csv(args.input)
    features_df = build_features(df, min_history=args.min_history)
    features_df.to_csv(args.output)
    print(f"Features: {features_df.shape} → {args.output}")
    print(f"  {len(get_feature_cols(features_df))} features, {len(get_target_cols(features_df))} targets")
