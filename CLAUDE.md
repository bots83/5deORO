# 5deORO — Sistema de análisis y predicción del 5 de Oro (La Banca Uruguay)

## Propósito

Investigación estadística + ML sobre el juego "5 de Oro" de La Banca Uruguay (5 números del 1-48). El objetivo es analizar si existen patrones explotables, como base para herramientas de prevención/educación.

## Estructura

```
data/raw/          # HTML crudo descargado por los scrapers
data/processed/    # sorteos.csv (dataset limpio), features.csv
data/db/           # 5deoro.db (SQLite)
scrapers/          # Scrapers de fuentes externas
db/                # Schema SQL y capa de acceso a datos
analysis/          # Análisis estadístico (frecuencias, coocurrencia, temporal)
features/          # Feature engineering para ML
ml/                # Modelos y evaluación
reports/           # CSVs y gráficos generados
notebooks/         # Exploración interactiva con Jupyter
```

## Flujo de ejecución

```bash
# 1. Instalar dependencias faltantes
pip install lxml tqdm

# 2. Inicializar DB
python -m db.repository --init

# 3. Scraping histórico (fuente principal: LotteryTexts.com)
python -m scrapers.lotteryTexts_scraper --save-db

# 4. Actualización con sorteos recientes (Loteria.Guru)
python -m scrapers.loteriaGuru_scraper --save-db

# 5. Exportar CSV limpio
python -c "from db.repository import Repository; Repository().export_csv('data/processed/sorteos.csv')"

# 6. Análisis estadístico
python -m analysis.report --output reports/

# 7. Build features + entrenar modelos
python -m ml.trainer

# 8. Exploración interactiva
jupyter lab notebooks/
```

## Fuentes de datos

- **LotteryTexts.com** — ~1,003 sorteos desde enero 2016 (fuente principal histórica)
- **Loteria.Guru** — resultados recientes para actualización incremental
- **La Banca oficial** (labanca.com.uy) — devuelve 403, no accesible directamente

## Estructura de un sorteo

```
fecha: YYYY-MM-DD
n1, n2, n3, n4, n5: enteros 1-48 ordenados (n1 < n2 < n3 < n4 < n5)
bonus: entero 1-48 (bola adicional)
```

## Métrica clave del modelo

**Top-K recall**: de los 5 números sorteados, cuántos estaban en los K predichos como más probables.
- K=10, baseline aleatorio esperado: 5×(10/48) ≈ 1.04 números correctos/sorteo
- Si el modelo supera 1.3+ sostenidamente → hay señal estadística real

## Notas técnicas

- SQLite (no Postgres): 1,000 filas no necesitan servidor
- Split temporal 80/20 — nunca shuffle (viola causalidad)
- XGBoost multilabel como modelo primario (más robusto con ~800 sorteos de train)
- Zero leakage en features: cada feature en sorteo t usa solo datos hasta t-1
