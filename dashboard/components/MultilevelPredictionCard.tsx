"use client";
import { Sparkles, Target, Shield, ShieldCheck } from "lucide-react";

type MultilevelData = {
  fecha_dataset: string;
  n_sorteos_dataset: number;
  predictions: Record<string, { numbers: number[]; label: string; prob_total: number }>;
  weights_used: Record<string, number>;
};

const LEVELS = [
  { key: "top5", title: "🎯 Top-5", subtitle: "Apuesta puntual", icon: Target, color: "from-amber-500 to-yellow-600" },
  { key: "top10", title: "📊 Top-10", subtitle: "Recomendado", icon: Sparkles, color: "from-blue-500 to-cyan-600" },
  { key: "top15", title: "🛡 Top-15", subtitle: "Cobertura amplia", icon: Shield, color: "from-violet-500 to-purple-600" },
  { key: "top20", title: "🛡 Top-20", subtitle: "Seguro", icon: ShieldCheck, color: "from-emerald-500 to-teal-600" },
  { key: "top25", title: "💎 Top-25", subtitle: "Muy seguro", icon: ShieldCheck, color: "from-pink-500 to-rose-600" },
];

export default function MultilevelPredictionCard({ data }: { data: MultilevelData }) {
  return (
    <div className="glass rounded-2xl p-8 mb-6 relative overflow-hidden">
      <div className="absolute inset-0 gold-gradient opacity-5" />
      <div className="relative">
        <h2 className="text-2xl font-bold mb-2 gold-text">Predicciones multinivel</h2>
        <p className="text-sm text-slate-400 mb-6">
          Diferentes niveles según tu tolerancia a riesgo
        </p>

        <div className="space-y-4">
          {LEVELS.map(({ key, title, subtitle, icon: Icon, color }) => {
            const pred = data.predictions[key];
            if (!pred) return null;
            return (
              <div key={key} className="bg-black/30 rounded-xl p-4 border border-white/5">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="text-lg font-bold flex items-center gap-2">
                      <Icon size={18} />
                      {title}
                    </div>
                    <div className="text-xs text-slate-400">{subtitle}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-slate-400">Cobertura</div>
                    <div className="text-lg font-bold gold-text">
                      {(pred.prob_total * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {pred.numbers.map((num) => (
                    <div
                      key={num}
                      className={`ball w-9 h-9 text-sm bg-gradient-to-br ${color} text-white shadow-lg`}
                    >
                      {String(num).padStart(2, "0")}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 p-4 bg-amber-500/5 border border-amber-500/20 rounded-lg text-xs text-amber-200">
          ⚠️ El sorteo es estadísticamente aleatorio. Estos rankings reflejan patrones recientes
          y NO garantizan ningún resultado. Úsalo como referencia de investigación.
        </div>
      </div>
    </div>
  );
}
