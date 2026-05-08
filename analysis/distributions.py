"""Distribuciones estadísticas de los sorteos."""
import pandas as pd
import numpy as np
from scipy import stats


def paridad(df: pd.DataFrame) -> pd.Series:
    """Cuenta de números pares en cada sorteo (0-5)."""
    nums = df[["n1", "n2", "n3", "n4", "n5"]]
    return (nums % 2 == 0).sum(axis=1).rename("pares")


def suma_total(df: pd.DataFrame) -> pd.Series:
    return (df["n1"] + df["n2"] + df["n3"] + df["n4"] + df["n5"]).rename("suma_total")


def rango(df: pd.DataFrame) -> pd.Series:
    return (df["n5"] - df["n1"]).rename("rango")


def distribucion_decenas(df: pd.DataFrame) -> pd.DataFrame:
    """Cuántos números caen en cada decena (1-10, 11-20, ..., 41-48) por sorteo."""
    decenas = {}
    for dec, label in [(1, "d1_10"), (11, "d11_20"), (21, "d21_30"), (31, "d31_40"), (41, "d41_48")]:
        hi = dec + 9 if dec < 41 else 48
        nums = df[["n1", "n2", "n3", "n4", "n5"]]
        decenas[label] = ((nums >= dec) & (nums <= hi)).sum(axis=1)
    return pd.DataFrame(decenas)


def estadisticas_suma(df: pd.DataFrame) -> dict:
    s = suma_total(df)
    return {
        "media": float(s.mean()),
        "mediana": float(s.median()),
        "std": float(s.std()),
        "min": int(s.min()),
        "max": int(s.max()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
        "esperado_teorico": 5 * 49 / 2,  # E[X] para uniforme 1-48: (1+48)/2 * 5 = 122.5
    }


def test_uniformidad(df: pd.DataFrame) -> dict:
    """Chi-cuadrado para verificar si las frecuencias son uniformes."""
    from analysis.frequencies import frecuencia_absoluta
    obs = frecuencia_absoluta(df).values
    n = len(df) * 5
    exp = np.full(48, n / 48)
    chi2, p = stats.chisquare(obs, exp)
    return {"chi2": float(chi2), "p_value": float(p), "uniforme": p > 0.05}
