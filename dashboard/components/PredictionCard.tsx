"use client";
import { Prediction } from "@/lib/data";
import { Sparkles, TrendingUp, Target, Shield, ShieldCheck, Award, Zap } from "lucide-react";

const LEVEL_CONFIG: Record<string, { title: string; icon: any; color: string; gradient: string }> = {
  top5:  { title: "Top-5",  icon: Target,      color: "amber",   gradient: "from-amber-500 to-yellow-600" },
  top10: { title: "Top-10", icon: Sparkles,    color: "blue",    gradient: "from-blue-500 to-cyan-600" },
  top12: { title: "Top-12", icon: Sparkles,    color: "indigo",  gradient: "from-indigo-500 to-blue-600" },
  top15: { title: "Top-15", icon: Shield,      color: "violet",  gradient: "from-violet-500 to-purple-600" },
  top20: { title: "Top-20", icon: ShieldCheck, color: "emerald", gradient: "from-emerald-500 to-teal-600" },
  top25: { title: "Top-25", icon: Award,       color: "pink",    gradient: "from-pink-500 to-rose-600" },
  top29: { title: "Top-29", icon: Award,       color: "rose",    gradient: "from-rose-500 to-pink-600" },
  top30: { title: "Top-30", icon: Award,       color: "fuchsia", gradient: "from-fuchsia-500 to-pink-600" },
  top37: { title: "Top-37", icon: Zap,         color: "purple",  gradient: "from-purple-500 to-violet-600" },
  top45: { title: "Top-45", icon: Zap,         color: "slate",   gradient: "from-slate-500 to-zinc-600" },
};

function levelLabel(key: string, pred: any): string {
  if (pred?.label) return pred.label;
  return LEVEL_CONFIG[key]?.title || key;
}

export default function PredictionCard({ prediction }: { prediction: Prediction }) {
  const baselineProb = 5 / 48;

  // Mostrar solo niveles más útiles si hay muchos
  const displayKeys = ["top5", "top10", "top15", "top20", "top25", "top29", "top37", "top45"];
  const availableKeys = displayKeys.filter(k => prediction.predictions[k]);

  return (
    <div className="space-y-4">
      {/* Top-10 destacado */}
      {prediction.predictions["top10"] && (
        <div className="glass rounded-2xl p-8 relative overflow-hidden">
          <div className="absolute inset-0 gold-gradient opacity-10" />
          <div className="relative">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="text-gold-400" size={24} />
              <h2 className="text-2xl font-bold">Predicción Top-10 (recomendada)</h2>
            </div>
            <div className="flex flex-wrap gap-3 justify-center my-6">
              {prediction.predictions["top10"].numbers.map((num) => (
                <div key={num} className="ball ball-gold w-16 h-16 text-2xl shadow-2xl">
                  {String(num).padStart(2, "0")}
                </div>
              ))}
            </div>
            <p className="text-center text-sm text-slate-400 mb-2">
              {prediction.predictions["top10"].note}
            </p>
            {prediction.predictions["top10"].backtest && (
              <div className="grid md:grid-cols-3 gap-3 mt-4 text-sm">
                <Stat label="≥1 hit" value={`${prediction.predictions["top10"].backtest["1plus_hits"]}/50`} />
                <Stat label="≥3 hits" value={`${prediction.predictions["top10"].backtest["3plus_hits"]}/50`} />
                <Stat label="Cobertura prob" value={`${(prediction.predictions["top10"].prob_total * 100).toFixed(0)}%`} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Otros niveles */}
      <div className="grid md:grid-cols-2 gap-4">
        {availableKeys.filter(k => k !== "top10").map((key) => {
          const pred = prediction.predictions[key];
          const cfg = LEVEL_CONFIG[key];
          if (!pred) return null;
          const Icon = cfg?.icon || Sparkles;
          return (
            <div key={key} className="glass rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Icon size={18} className="text-slate-300" />
                  <div className="font-bold">{levelLabel(key, pred)}</div>
                </div>
                {pred.backtest && (
                  <div className="text-xs text-slate-400">
                    {pred.backtest["3plus_hits"]}/50 con ≥3 hits
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {pred.numbers.slice(0, 30).map((num) => (
                  <div
                    key={num}
                    className={`ball w-8 h-8 text-xs bg-gradient-to-br ${cfg?.gradient || "from-slate-500 to-slate-600"} text-white shadow`}
                  >
                    {String(num).padStart(2, "0")}
                  </div>
                ))}
                {pred.numbers.length > 30 && (
                  <div className="text-xs text-slate-400 self-center ml-2">+{pred.numbers.length - 30} más</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-black/30 rounded-lg p-3 text-center">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className="text-lg font-semibold gold-text">{value}</div>
    </div>
  );
}
