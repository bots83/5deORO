#!/bin/bash
# Pipeline completo end-to-end:
# 1. Re-build features con dataset actualizado
# 2. Calibrar pesos con backtest
# 3. Generar predicción final
# 4. Copiar datos al dashboard
# 5. Re-deploy a Vercel
set -e

cd "$(dirname "$0")"

echo "==> 1. Exportando CSV desde DB..."
python3 -c "from db.repository import Repository; Repository().export_csv('data/processed/sorteos.csv')"

echo "==> 2. Construyendo features..."
rm -f data/processed/features.csv
python3 -m features.builder --input data/processed/sorteos.csv --output data/processed/features.csv --min-history 50

echo "==> 3. Tests de aleatoriedad..."
python3 -m analysis.randomness_tests --input data/processed/sorteos.csv --output reports/randomness_tests.json

echo "==> 4. Backtest + calibración..."
python3 -W ignore -m ml.backtest --features data/processed/features.csv --output reports/ 2>&1 | grep -v "UserWarning\|warnings.warn\|valid feature names" | tail -50

echo "==> 5. Predicción final..."
python3 -W ignore -m ml.predict_final --input data/processed/sorteos.csv --weights reports/ensemble_weights.json --output reports/prediction_final.json 2>&1 | grep -v "UserWarning\|warnings.warn\|valid feature names" | tail -40

echo "==> 6. Copiando datos al dashboard..."
mkdir -p dashboard/public/data
cp data/processed/sorteos.csv dashboard/public/data/sorteos.csv
cp reports/prediction_final.json dashboard/public/data/prediction.json
cp reports/randomness_tests.json dashboard/public/data/randomness.json
cp reports/model_calibration.csv dashboard/public/data/calibration.csv 2>/dev/null || true

echo "==> 7. Build dashboard..."
(cd dashboard && npm run build 2>&1 | tail -5)

echo "==> 8. Commit y push..."
git add -A
git commit -m "Update: dataset extendido + nueva predicción

🤖 Generated with Claude Code
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>" || echo "No hay cambios para commitear"
git push 2>&1 | tail -3

echo "==> 9. Deploy a Vercel..."
(cd dashboard && vercel deploy --prod --yes 2>&1 | tail -5)

echo "✓ Pipeline completo"
