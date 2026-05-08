"""Análisis de coocurrencia de pares y tríos."""
from itertools import combinations

import numpy as np
import pandas as pd


def matriz_coocurrencia(df: pd.DataFrame) -> np.ndarray:
    """Matriz 48×48 simétrica: cuántas veces dos números salieron juntos."""
    mat = np.zeros((49, 49), dtype=int)
    for _, row in df.iterrows():
        nums = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        for a, b in combinations(nums, 2):
            mat[a][b] += 1
            mat[b][a] += 1
    return mat[1:, 1:]  # índices 1-48 → array 48×48


def pares_frecuentes(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    mat = np.zeros((49, 49), dtype=int)
    for _, row in df.iterrows():
        nums = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        for a, b in combinations(nums, 2):
            mat[a][b] += 1
            mat[b][a] += 1

    pares = []
    for i in range(1, 49):
        for j in range(i + 1, 49):
            if mat[i][j] > 0:
                pares.append({"n1": i, "n2": j, "coocurrencia": mat[i][j]})

    return (
        pd.DataFrame(pares)
        .sort_values("coocurrencia", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def trios_frecuentes(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    conteo: dict = {}
    for _, row in df.iterrows():
        nums = sorted([row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]])
        for trio in combinations(nums, 3):
            conteo[trio] = conteo.get(trio, 0) + 1

    trios = [
        {"n1": t[0], "n2": t[1], "n3": t[2], "coocurrencia": v}
        for t, v in conteo.items()
    ]
    return (
        pd.DataFrame(trios)
        .sort_values("coocurrencia", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
