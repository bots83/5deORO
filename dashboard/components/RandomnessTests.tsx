"use client";
import { RandomnessTests } from "@/lib/data";
import { Check, X } from "lucide-react";

export default function RandomnessTestsCard({ tests }: { tests: RandomnessTests }) {
  const items = [
    { name: "Uniformidad (chi²)", p: tests.chi_squared_uniform.p_value, pass: tests.chi_squared_uniform.uniforme },
    { name: "Gaps geométricos", p: tests.gap_test_geometric.p_value, pass: tests.gap_test_geometric.geometrico },
    { name: "IID (Ljung-Box)", p: tests.autocorrelation_test.ljung_box_lag10_p, pass: tests.autocorrelation_test.iid },
    { name: "Coocurrencia indep.", p: tests.cooccurrence_chi2.p_value, pass: tests.cooccurrence_chi2.independientes },
  ];

  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="text-xl font-bold mb-1">Tests de aleatoriedad</h2>
      <p className="text-xs text-slate-400 mb-4">Verificación de que el sorteo NO tiene sesgos explotables</p>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item.name} className="flex items-center justify-between bg-black/20 rounded-lg p-3">
            <div>
              <div className="text-sm font-medium">{item.name}</div>
              <div className="text-xs text-slate-400">p = {item.p.toFixed(4)}</div>
            </div>
            {item.pass ? (
              <div className="flex items-center gap-1 text-emerald-400 text-sm">
                <Check size={16} /> Aleatorio
              </div>
            ) : (
              <div className="flex items-center gap-1 text-red-400 text-sm">
                <X size={16} /> Sesgo
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-xs text-amber-200">
        ⚠ Si todos los tests confirman aleatoriedad, no hay forma matemática de predecir mejor que el azar a largo plazo.
      </div>
    </div>
  );
}
