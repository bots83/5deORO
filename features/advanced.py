"""Features avanzadas para predicción más fina.

Añade sobre las features básicas:
- Lag features: probabilidad por número en lags 1, 2, 3, 5, 7
- Rolling stats: hot/cold streaks, momentum
- Interacciones: combos de números frecuentes en par
- Date features: día de semana, semana del mes, season
- Bolilla extra como feature
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


def _build_advanced_features(history: pd.DataFrame) -> dict:
    n = len(history)
    if n == 0:
        return {}

    feats = {}
    all_nums = history[NUM_COLS].values  # (n, 5)

    for num in NUMS:
        presence = (all_nums == num).any(axis=1)

        # Frecuencias en múltiples ventanas
        feats[f"freq_hist_{num}"] = presence.sum() / n
        for window in [5, 10, 15, 20, 30, 50, 75]:
            w = min(window, n)
            feats[f"freq_{window}_{num}"] = presence[-w:].sum() / w

        # Lag features: ¿salió hace exactamente N sorteos?
        for lag in [1, 2, 3, 5, 7, 10]:
            if n >= lag:
                feats[f"lag{lag}_{num}"] = float(presence[-lag])
            else:
                feats[f"lag{lag}_{num}"] = 0.0

        # Ausencia y ciclo
        positions = np.where(presence)[0]
        if len(positions) == 0:
            feats[f"ausencia_{num}"] = float(n)
            feats[f"ciclo_avg_{num}"] = float(n)
            feats[f"ciclo_std_{num}"] = 0.0
            feats[f"ciclo_recent_{num}"] = float(n)
        else:
            feats[f"ausencia_{num}"] = float(n - 1 - positions[-1])
            if len(positions) >= 2:
                diffs = np.diff(positions)
                feats[f"ciclo_avg_{num}"] = float(diffs.mean())
                feats[f"ciclo_std_{num}"] = float(diffs.std())
                # Ciclo reciente: media de los últimos 3 gaps
                feats[f"ciclo_recent_{num}"] = float(diffs[-3:].mean()) if len(diffs) >= 3 else float(diffs.mean())
            else:
                feats[f"ciclo_avg_{num}"] = float(n)
                feats[f"ciclo_std_{num}"] = 0.0
                feats[f"ciclo_recent_{num}"] = float(n)

        # Hot/cold streak: cuántos sorteos consecutivos saliendo (o sin salir)
        streak_active = 0
        streak_out = 0
        for j in range(n - 1, -1, -1):
            if presence[j]:
                if streak_active >= 0:
                    streak_active += 1
                else:
                    break
            else:
                if streak_active > 0:
                    break
                streak_out += 1
        feats[f"streak_in_{num}"] = float(streak_active)
        feats[f"streak_out_{num}"] = float(streak_out)

        # Momentum: diff de freq window 10 vs 30
        if n >= 30:
            f10 = presence[-10:].sum() / 10
            f30 = presence[-30:].sum() / 30
            feats[f"momentum_{num}"] = f10 - f30
        else:
            feats[f"momentum_{num}"] = 0.0

        # Sliding ratio: freq últimos 20 / freq histórica
        if n >= 20:
            f20 = presence[-20:].sum() / 20
            f_hist = feats[f"freq_hist_{num}"]
            feats[f"ratio20_hist_{num}"] = f20 / f_hist if f_hist > 0 else 1.0
        else:
            feats[f"ratio20_hist_{num}"] = 1.0

    # Stats del último sorteo
    prev = history.iloc[-1]
    prev_nums = [prev[c] for c in NUM_COLS]
    feats["suma_prev"] = float(sum(prev_nums))
    feats["paridad_prev"] = float(sum(1 for x in prev_nums if x % 2 == 0))
    feats["rango_prev"] = float(max(prev_nums) - min(prev_nums))
    feats["min_prev"] = float(min(prev_nums))
    feats["max_prev"] = float(max(prev_nums))
    feats["std_prev"] = float(np.std(prev_nums))

    # Decenas en el sorteo anterior
    for dec_lo, dec_hi, label in [(1, 12, "d1_12"), (13, 24, "d13_24"),
                                    (25, 36, "d25_36"), (37, 48, "d37_48")]:
        feats[f"prev_{label}"] = float(sum(1 for x in prev_nums if dec_lo <= x <= dec_hi))

    # Stats últimos 5 sorteos
    if n >= 5:
        last5_nums = history.tail(5)[NUM_COLS].values.flatten()
        feats["suma_5_avg"] = float(history.tail(5)[NUM_COLS].sum(axis=1).mean())
        feats["paridad_5_avg"] = float(np.mean([sum(1 for x in row if x % 2 == 0)
                                                 for row in history.tail(5)[NUM_COLS].values]))
    else:
        feats["suma_5_avg"] = float(sum(prev_nums))
        feats["paridad_5_avg"] = feats["paridad_prev"]

    # Bolilla extra del sorteo anterior (si existe)
    if "bolilla_extra" in history.columns and pd.notna(prev.get("bolilla_extra")):
        be = int(prev["bolilla_extra"])
        feats["prev_bolilla_extra"] = float(be)
        feats["prev_be_par"] = float(be % 2 == 0)
        # Decena de la bolilla extra
        feats["prev_be_dec"] = float((be - 1) // 12)
    else:
        feats["prev_bolilla_extra"] = 0.0
        feats["prev_be_par"] = 0.5
        feats["prev_be_dec"] = -1.0

    # Contexto temporal
    feats["sorteo_num"] = float(n)
    last_date = pd.to_datetime(history["fecha"].iloc[-1])
    feats["mes"] = float(last_date.month)
    feats["dia_mes"] = float(last_date.day)
    feats["dia_semana_num"] = float(last_date.dayofweek)
    feats["semana_mes"] = float((last_date.day - 1) // 7 + 1)
    feats["trimestre"] = float((last_date.month - 1) // 3 + 1)

    return feats


def build_advanced_features(df: pd.DataFrame, min_history: int = 50) -> pd.DataFrame:
    """Construye features avanzadas. Más lento que builder.py pero más rico."""
    df = df.sort_values("fecha").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"])

    rows = []
    for idx in range(min_history, len(df)):
        history = df.iloc[:idx]
        feats = _build_advanced_features(history)
        if not feats:
            continue
        sorteo_actual = df.iloc[idx]
        nums_actuales = set([sorteo_actual[c] for c in NUM_COLS])
        for num in NUMS:
            feats[f"target_{num}"] = 1 if num in nums_actuales else 0
        feats["fecha"] = sorteo_actual["fecha"]
        rows.append(feats)
    return pd.DataFrame(rows).set_index("fecha")
