"""Análisis de frecuencias de números 1-48."""
import pandas as pd
import numpy as np


def _all_numbers(df: pd.DataFrame) -> pd.Series:
    return pd.concat([df["n1"], df["n2"], df["n3"], df["n4"], df["n5"]])


def frecuencia_absoluta(df: pd.DataFrame) -> pd.Series:
    counts = _all_numbers(df).value_counts().reindex(range(1, 49), fill_value=0)
    counts.index.name = "numero"
    counts.name = "freq_abs"
    return counts.sort_index()


def frecuencia_relativa(df: pd.DataFrame) -> pd.Series:
    total = len(df) * 5
    freq = frecuencia_absoluta(df) / total
    freq.name = "freq_rel"
    return freq


def frecuencia_ventana(df: pd.DataFrame, ventana: int = 100) -> pd.DataFrame:
    """Frecuencia de cada número en los últimos `ventana` sorteos."""
    subset = df.tail(ventana)
    return frecuencia_absoluta(subset).rename(f"freq_{ventana}").to_frame()


def numeros_calientes_frios(df: pd.DataFrame, top_n: int = 10) -> dict:
    freq = frecuencia_absoluta(df).sort_values(ascending=False)
    return {
        "calientes": freq.head(top_n).index.tolist(),
        "frios": freq.tail(top_n).index.tolist(),
        "calientes_freq": freq.head(top_n).values.tolist(),
        "frios_freq": freq.tail(top_n).values.tolist(),
    }


def tabla_frecuencias_completa(df: pd.DataFrame, ventana: int = 100) -> pd.DataFrame:
    """Tabla consolidada con frecuencias históricas y recientes."""
    hist = frecuencia_absoluta(df).rename("freq_hist")
    rel = frecuencia_relativa(df).rename("freq_rel")
    vent = frecuencia_ventana(df, ventana)[f"freq_{ventana}"]

    result = pd.concat([hist, rel, vent], axis=1)
    result.index.name = "numero"
    result["esperado"] = len(df) * 5 / 48
    result["desviacion"] = (result["freq_hist"] - result["esperado"]) / result["esperado"] * 100
    return result
