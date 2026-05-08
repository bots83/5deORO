"use client";
import { Sorteo, Prediction } from "@/lib/data";
import { useMemo } from "react";

type Props = {
  sorteos: Sorteo[];
  prediction: Prediction | null;
};

export default function SimulationView({ sorteos, prediction }: Props) {
  const stats = useMemo(() => {
    if (sorteos.length === 0) return null;

    // Calcular estadísticas básicas
    const allNums = sorteos.flatMap((s) => [s.n1, s.n2, s.n3, s.n4, s.n5]);
    const sumas = sorteos.map((s) => s.n1 + s.n2 + s.n3 + s.n4 + s.n5);
    const sumaAvg = sumas.reduce((a, b) => a + b, 0) / sumas.length;
    const sumaMin = Math.min(...sumas);
    const sumaMax = Math.max(...sumas);

    // Frecuencias
    const freq: Record<number, number> = {};
    for (let i = 1; i <= 48; i++) freq[i] = 0;
    allNums.forEach((n) => freq[n]++);

    const freqArr = Object.entries(freq).map(([n, c]) => ({ num: +n, count: c }));
    freqArr.sort((a, b) => b.count - a.count);
    const hot = freqArr.slice(0, 5);
    const cold = freqArr.slice(-5);

    // Pares
    const evenCount = allNums.filter((n) => n % 2 === 0).length;
    const evenPct = (evenCount / allNums.length) * 100;

    // Decenas
    const decenas = [0, 0, 0, 0];
    allNums.forEach((n) => {
      if (n <= 12) decenas[0]++;
      else if (n <= 24) decenas[1]++;
      else if (n <= 36) decenas[2]++;
      else decenas[3]++;
    });

    // Si tenemos predicción, evaluar contra últimos N
    let backtest = null;
    if (prediction) {
      const top10 = new Set(prediction.top10);
      const last20 = sorteos.slice(-20);
      let hits = 0;
      last20.forEach((s) => {
        const nums = [s.n1, s.n2, s.n3, s.n4, s.n5];
        nums.forEach((n) => {
          if (top10.has(n)) hits++;
        });
      });
      const expectedHits = (last20.length * 5 * 10) / 48;
      backtest = {
        last_n: last20.length,
        hits,
        expected: expectedHits.toFixed(2),
        ratio: (hits / expectedHits).toFixed(2),
      };
    }

    return {
      sumaAvg: sumaAvg.toFixed(2),
      sumaMin,
      sumaMax,
      hot,
      cold,
      evenPct: evenPct.toFixed(1),
      decenas,
      backtest,
    };
  }, [sorteos, prediction]);

  if (!stats) return null;

  return (
    <div className="glass rounded-2xl p-6 mb-6">
      <h2 className="text-xl font-bold mb-4">📊 Simulaciones y estadísticas</h2>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div>
          <h3 className="text-sm text-slate-400 mb-2">🔥 5 números más frecuentes (hot)</h3>
          <div className="flex gap-2 flex-wrap">
            {stats.hot.map((h) => (
              <div key={h.num} className="bg-red-500/20 border border-red-500/30 rounded-lg px-3 py-2">
                <span className="font-bold">{String(h.num).padStart(2, "0")}</span>
                <span className="text-xs text-slate-400 ml-2">{h.count}x</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm text-slate-400 mb-2">❄ 5 números menos frecuentes (cold)</h3>
          <div className="flex gap-2 flex-wrap">
            {stats.cold.map((c) => (
              <div key={c.num} className="bg-blue-500/20 border border-blue-500/30 rounded-lg px-3 py-2">
                <span className="font-bold">{String(c.num).padStart(2, "0")}</span>
                <span className="text-xs text-slate-400 ml-2">{c.count}x</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-3 mb-6">
        <Stat label="Suma promedio" value={stats.sumaAvg} hint={`Esperado: 122.5 (5×49/2)`} />
        <Stat label="% Pares" value={`${stats.evenPct}%`} hint="Esperado: ~50% (24/48)" />
        <Stat label="Rango sumas" value={`${stats.sumaMin}-${stats.sumaMax}`} />
      </div>

      <div className="mb-6">
        <h3 className="text-sm text-slate-400 mb-2">Distribución por decenas</h3>
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: "1-12", value: stats.decenas[0] },
            { label: "13-24", value: stats.decenas[1] },
            { label: "25-36", value: stats.decenas[2] },
            { label: "37-48", value: stats.decenas[3] },
          ].map((d) => {
            const total = stats.decenas.reduce((a, b) => a + b, 0);
            const pct = ((d.value / total) * 100).toFixed(1);
            return (
              <div key={d.label} className="bg-black/30 rounded-lg p-3 text-center">
                <div className="text-xs text-slate-400">{d.label}</div>
                <div className="text-lg font-bold gold-text">{pct}%</div>
                <div className="text-xs text-slate-500">{d.value} apariciones</div>
              </div>
            );
          })}
        </div>
      </div>

      {stats.backtest && (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4">
          <h3 className="text-sm text-amber-300 mb-2">🎯 Back-test: Top-10 vs últimos {stats.backtest.last_n} sorteos</h3>
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div>
              <div className="text-xs text-slate-400">Hits totales</div>
              <div className="text-lg font-bold">{stats.backtest.hits}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Esperado (random)</div>
              <div className="text-lg font-bold text-slate-300">{stats.backtest.expected}</div>
            </div>
            <div>
              <div className="text-xs text-slate-400">Ratio vs random</div>
              <div className="text-lg font-bold text-amber-300">{stats.backtest.ratio}x</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-black/30 rounded-lg p-3 text-center">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-xl font-bold gold-text">{value}</div>
      {hint && <div className="text-xs text-slate-500 mt-1">{hint}</div>}
    </div>
  );
}
