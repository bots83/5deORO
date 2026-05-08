"use client";
import { Prediction } from "@/lib/data";
import { Sparkles, TrendingUp } from "lucide-react";

export default function PredictionCard({ prediction }: { prediction: Prediction }) {
  const baselineProb = 5 / 48;

  return (
    <div className="glass rounded-2xl p-8 mb-6 relative overflow-hidden">
      <div className="absolute inset-0 gold-gradient opacity-10" />
      <div className="relative">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="text-gold-400" size={24} />
          <h2 className="text-2xl font-bold">Predicción para el próximo sorteo</h2>
        </div>

        <div className="flex flex-wrap gap-3 justify-center my-6">
          {prediction.top5.map((num) => (
            <div key={num} className="ball ball-gold w-20 h-20 text-3xl shadow-2xl">
              {String(num).padStart(2, "0")}
            </div>
          ))}
        </div>

        <div className="text-center text-sm text-slate-400 mb-4">
          5 números más probables — ensemble de {Object.keys(prediction.weights_used).length} modelos
        </div>

        <div className="grid md:grid-cols-3 gap-4 mt-6">
          <Stat label="Sorteos analizados" value={prediction.n_sorteos_dataset.toString()} />
          <Stat label="Último sorteo" value={prediction.fecha_dataset} />
          <Stat label="Mejor número" value={`${prediction.top5[0]} (${(prediction.ensemble_probs[`num_${prediction.top5[0]}`] / baselineProb).toFixed(2)}x random)`} />
        </div>

        <div className="mt-6">
          <h3 className="text-sm text-slate-300 mb-3 flex items-center gap-2">
            <TrendingUp size={16} /> Top-10 (con margen)
          </h3>
          <div className="flex flex-wrap gap-2 justify-center">
            {prediction.top10.map((num) => (
              <div
                key={num}
                className={`ball w-12 h-12 text-lg ${prediction.top5.includes(num) ? "ball-gold" : "ball-normal"}`}
              >
                {String(num).padStart(2, "0")}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-black/30 rounded-lg p-4 text-center">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className="text-lg font-semibold gold-text">{value}</div>
    </div>
  );
}
