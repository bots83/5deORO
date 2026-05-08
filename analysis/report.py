"""Genera reporte completo de análisis estadístico."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.frequencies import tabla_frecuencias_completa, numeros_calientes_frios
from analysis.temporal import ausencia_actual, rachas_maximas, ciclo_retorno_promedio, autocorrelacion_suma
from analysis.cooccurrence import pares_frecuentes, trios_frecuentes
from analysis.distributions import paridad, suma_total, rango, distribucion_decenas, estadisticas_suma, test_uniformidad

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def plot_frecuencias(df: pd.DataFrame, output_dir: Path):
    tabla = tabla_frecuencias_completa(df)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax1 = axes[0]
    x = tabla.index
    ax1.bar(x, tabla["freq_hist"], alpha=0.7, label="Frecuencia histórica", color="steelblue")
    ax1.axhline(tabla["esperado"].iloc[0], color="red", linestyle="--", label=f"Esperado ({tabla['esperado'].iloc[0]:.1f})")
    ax1.set_xlabel("Número")
    ax1.set_ylabel("Apariciones")
    ax1.set_title("Frecuencia histórica de cada número (1-48)")
    ax1.legend()
    ax1.set_xticks(range(1, 49))
    ax1.tick_params(axis="x", labelsize=7)

    ax2 = axes[1]
    ausencias = ausencia_actual(df.sort_values("fecha"))
    colors = ["red" if a > 15 else "orange" if a > 8 else "steelblue" for a in ausencias.values]
    ax2.bar(ausencias.index, ausencias.values, color=colors, alpha=0.8)
    ax2.set_xlabel("Número")
    ax2.set_ylabel("Sorteos sin aparecer")
    ax2.set_title("Ausencia actual de cada número")
    ax2.set_xticks(range(1, 49))
    ax2.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / "frecuencias_ausencias.png", dpi=150)
    plt.close(fig)
    print(f"  Gráfico: {output_dir / 'frecuencias_ausencias.png'}")


def plot_suma_distribucion(df: pd.DataFrame, output_dir: Path):
    sumas = suma_total(df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(sumas, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(sumas.mean(), color="red", linestyle="--", label=f"Media: {sumas.mean():.1f}")
    ax.axvline(122.5, color="green", linestyle=":", label="Esperado teórico: 122.5")
    ax.set_xlabel("Suma total del sorteo")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de la suma de los 5 números por sorteo")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "distribucion_suma.png", dpi=150)
    plt.close(fig)


def plot_coocurrencia_heatmap(df: pd.DataFrame, output_dir: Path):
    from analysis.cooccurrence import matriz_coocurrencia
    mat = matriz_coocurrencia(df)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax)
    ax.set_title("Matriz de coocurrencia (veces que dos números salieron juntos)")
    ax.set_xlabel("Número")
    ax.set_ylabel("Número")
    ticks = list(range(0, 48, 5))
    labels = [str(i+1) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    fig.savefig(output_dir / "coocurrencia_heatmap.png", dpi=150)
    plt.close(fig)


def run(csv_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    print(f"Dataset: {len(df)} sorteos ({df['fecha'].min().date()} → {df['fecha'].max().date()})")

    # 1. Frecuencias
    print("\n[1/6] Análisis de frecuencias...")
    tabla = tabla_frecuencias_completa(df)
    tabla.to_csv(output_dir / "frecuencias.csv")
    calientes_frios = numeros_calientes_frios(df)

    # 2. Temporal
    print("[2/6] Análisis temporal...")
    ausencias = ausencia_actual(df)
    rachas = rachas_maximas(df)
    ciclos = ciclo_retorno_promedio(df)
    temporal_df = pd.concat([ausencias, rachas, ciclos], axis=1)
    temporal_df.to_csv(output_dir / "temporal.csv")
    acf_df = autocorrelacion_suma(df)
    acf_df.to_csv(output_dir / "acf_suma.csv", index=False)

    # 3. Coocurrencias
    print("[3/6] Análisis de coocurrencias...")
    pares = pares_frecuentes(df, top_n=30)
    pares.to_csv(output_dir / "pares_frecuentes.csv", index=False)
    trios = trios_frecuentes(df, top_n=30)
    trios.to_csv(output_dir / "trios_frecuentes.csv", index=False)

    # 4. Distribuciones
    print("[4/6] Distribuciones...")
    stats_suma = estadisticas_suma(df)
    uniformidad = test_uniformidad(df)
    paridades = paridad(df).value_counts().sort_index()

    uniformidad_serializable = {k: bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, float) else v for k, v in uniformidad.items()}
    dist_report = {
        "suma": stats_suma,
        "uniformidad_chi2": uniformidad_serializable,
        "paridad_distribucion": {int(k): int(v) for k, v in paridades.to_dict().items()},
    }
    with open(output_dir / "distribuciones.json", "w") as f:
        json.dump(dist_report, f, indent=2)

    # 5. Gráficos
    print("[5/6] Generando gráficos...")
    plot_frecuencias(df, output_dir)
    plot_suma_distribucion(df, output_dir)
    plot_coocurrencia_heatmap(df, output_dir)

    # 6. Resumen en consola
    print("\n[6/6] Resumen:")
    print(f"  Total sorteos: {len(df)}")
    print(f"  Números más frecuentes: {calientes_frios['calientes']}")
    print(f"  Números menos frecuentes: {calientes_frios['frios']}")
    print(f"  Números con mayor ausencia actual: {ausencias.nlargest(5).to_dict()}")
    print(f"  Test de uniformidad — chi²={uniformidad['chi2']:.2f}, p={uniformidad['p_value']:.4f} → {'UNIFORME' if uniformidad['uniforme'] else 'NO UNIFORME'}")
    print(f"  Suma promedio por sorteo: {stats_suma['media']:.1f} (esperado teórico: 122.5)")
    print(f"\nArchivos generados en: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sorteos.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    run(args.input, args.output)
