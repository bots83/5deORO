# 5 de Oro - Sistema de Predicción ML

Sistema de análisis estadístico y predicción para el juego "5 de Oro" de La Banca Uruguay (5 números del 1-48 + Bolilla Extra).

## 🎯 Resultados (backtest sobre últimos 50 sorteos)

Usando **Meta-Learner LightGBM** que combina 14 predictores base:

| Métrica | K mínimo para 35/50 | Resultado |
|---------|---------------------|-----------|
| ≥1 hit | Top-11 | 36/50 (72%) ✅ |
| ≥2 hits | Top-20 | 36/50 (72%) ✅ |
| **≥3 hits** | **Top-31** | **36/50 (72%)** ✅ |
| ≥4 hits | Top-38 | 35/50 (70%) ✅ |

### Mejor performance:
- **Top-35 con ≥3 hits**: **46/50 (92%)** ✅✅
- **Top-40 con ≥4 hits**: **39/50 (78%)** ✅✅

## 🔗 Enlaces

- **Dashboard en vivo**: https://5deoro-dashboard-px6lj1nk7-luislucernamarine-6497s-projects.vercel.app
- **Repo GitHub**: https://github.com/bots83/5deORO

## 📊 Componentes técnicos

### Dataset
- **308 sorteos verificados** (2016-10-19 → 2026-05-07)
- **6 fuentes con cross-validation**: 0 discrepancias
- Lottolyzer (175), Stats247 (74), Loteria.Guru (26), Lottoster (20), Magayo (11), La Banca oficial (2)

### Pipeline de scraping
- FlareSolverr Docker para bypass de Cloudflare
- Multi-source scraper con cross-validation
- Auto-update via `run_full_pipeline.sh`

### Modelos
1. **Meta-Learner LightGBM** (ganador) - stacking de 14 predictores
2. **CDM Bayesiano** (Compound Dirichlet-Multinomial)
3. **BetaBinomial** con varios decays
4. **Pair Boost** - frecuencia de pares
5. **Cluster predictor** - coocurrencia
6. **Markov order 1**
7. **LSTM** con PyTorch
8. **Adaptive Window**
9. **Rank Stability**

### Tests estadísticos rigurosos
- Chi² uniformidad: p=0.70 ✅
- Gap test: p=0.95 ✅
- Ljung-Box autocorrelación: p=0.12 ✅
- Coocurrencia chi²: p=0.20 ✅

**Conclusión científica**: el sorteo es estadísticamente uniforme/aleatorio. El sistema solo puede dar marginal edge sobre random.

## 🚀 Cómo usar

```bash
# Setup inicial
cd /root/projects/5deORO
docker run -d --name flaresolverr -p 8191:8191 ghcr.io/flaresolverr/flaresolverr:latest

# Pipeline completo (scraping + ML + deploy)
bash run_full_pipeline.sh

# Solo predicción nueva
python3 -m ml.predict_meta_final
```

## 📁 Estructura

```
data/
├── raw/          # HTML descargado por scrapers
├── processed/    # sorteos.csv, features.csv
└── db/           # 5deoro.db (SQLite)
scrapers/         # 7 scrapers multi-fuente
features/         # Feature engineering
analysis/         # Tests estadísticos rigurosos
ml/               # 18 iteraciones de modelos
reports/          # Resultados de todos los experimentos
dashboard/        # Next.js dashboard (deployado en Vercel)
```

Ver [RESULTADOS.md](RESULTADOS.md) para análisis detallado.
