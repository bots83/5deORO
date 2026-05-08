"""Análisis temporal: ausencias, rachas, ciclos, autocorrelación."""
import pandas as pd
import numpy as np


def ausencia_actual(df: pd.DataFrame) -> pd.Series:
    """Sorteos transcurridos desde la última aparición de cada número."""
    df = df.sort_values("fecha").reset_index(drop=True)
    total = len(df)
    ausencias = {}
    for num in range(1, 49):
        apariciones = df[(df[["n1","n2","n3","n4","n5"]] == num).any(axis=1)].index
        if len(apariciones) == 0:
            ausencias[num] = total
        else:
            ausencias[num] = total - 1 - apariciones[-1]
    return pd.Series(ausencias, name="ausencia_actual")


def rachas_maximas(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada número: racha máxima de apariciones y ausencias consecutivas."""
    df = df.sort_values("fecha").reset_index(drop=True)
    results = []
    for num in range(1, 49):
        present = (df[["n1","n2","n3","n4","n5"]] == num).any(axis=1).astype(int)
        max_racha_presente = 0
        max_racha_ausente = 0
        cur_p = cur_a = 0
        for val in present:
            if val == 1:
                cur_p += 1
                cur_a = 0
            else:
                cur_a += 1
                cur_p = 0
            max_racha_presente = max(max_racha_presente, cur_p)
            max_racha_ausente = max(max_racha_ausente, cur_a)
        results.append({
            "numero": num,
            "racha_max_presente": max_racha_presente,
            "racha_max_ausente": max_racha_ausente,
        })
    return pd.DataFrame(results).set_index("numero")


def ciclo_retorno_promedio(df: pd.DataFrame) -> pd.Series:
    """Promedio de sorteos entre apariciones consecutivas de cada número."""
    df = df.sort_values("fecha").reset_index(drop=True)
    ciclos = {}
    for num in range(1, 49):
        apariciones = df[(df[["n1","n2","n3","n4","n5"]] == num).any(axis=1)].index.tolist()
        if len(apariciones) < 2:
            ciclos[num] = np.nan
        else:
            diffs = [apariciones[i+1] - apariciones[i] for i in range(len(apariciones)-1)]
            ciclos[num] = np.mean(diffs)
    return pd.Series(ciclos, name="ciclo_retorno_prom")


def suma_por_sorteo(df: pd.DataFrame) -> pd.Series:
    """Suma de los 5 números por sorteo."""
    return (df["n1"] + df["n2"] + df["n3"] + df["n4"] + df["n5"]).rename("suma_total")


def autocorrelacion_suma(df: pd.DataFrame, lags: int = 20) -> pd.DataFrame:
    """ACF de la suma total de los sorteos."""
    from statsmodels.tsa.stattools import acf
    sumas = suma_por_sorteo(df).values
    acf_vals, confint = acf(sumas, nlags=lags, alpha=0.05)
    return pd.DataFrame({
        "lag": range(lags + 1),
        "acf": acf_vals,
        "lower": confint[:, 0] - acf_vals,
        "upper": confint[:, 1] - acf_vals,
    })
