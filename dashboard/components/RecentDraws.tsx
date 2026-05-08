"use client";
import { Sorteo } from "@/lib/data";

export default function RecentDraws({ sorteos }: { sorteos: Sorteo[] }) {
  const recent = sorteos.slice(-10).reverse();

  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="text-xl font-bold mb-4">Últimos 10 sorteos</h2>
      <div className="space-y-2">
        {recent.map((s) => (
          <div key={s.fecha} className="flex items-center justify-between bg-black/20 rounded-lg p-3">
            <div className="text-xs text-slate-400 w-24">{s.fecha}</div>
            <div className="flex gap-1.5">
              {[s.n1, s.n2, s.n3, s.n4, s.n5].map((n, i) => (
                <div key={i} className="ball ball-normal w-8 h-8 text-xs">
                  {String(n).padStart(2, "0")}
                </div>
              ))}
              {s.bolilla_extra !== null && (
                <div className="ball ball-bonus w-8 h-8 text-xs ml-1" title="Bolilla Extra">
                  {String(s.bolilla_extra).padStart(2, "0")}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
