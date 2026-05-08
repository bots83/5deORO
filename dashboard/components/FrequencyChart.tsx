"use client";
import { Sorteo, computeFrequencies } from "@/lib/data";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from "recharts";

export default function FrequencyChart({ sorteos, highlight }: { sorteos: Sorteo[]; highlight: number[] }) {
  const freq = computeFrequencies(sorteos);
  const expected = (sorteos.length * 5) / 48;
  const data = Array.from({ length: 48 }, (_, i) => ({
    num: i + 1,
    count: freq.get(i + 1) || 0,
    highlighted: highlight.includes(i + 1),
  }));

  return (
    <div className="glass rounded-2xl p-6 mb-6">
      <h2 className="text-xl font-bold mb-1">Frecuencia histórica por número</h2>
      <p className="text-xs text-slate-400 mb-4">
        Línea naranja = frecuencia esperada bajo aleatoriedad uniforme ({expected.toFixed(1)}). Barras doradas = números predichos.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="num" stroke="#94a3b8" fontSize={11} />
          <YAxis stroke="#94a3b8" fontSize={11} />
          <Tooltip
            contentStyle={{ background: "#14141c", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8 }}
            labelStyle={{ color: "#fbbf24" }}
            formatter={(value: number) => [`${value} apariciones`, "Frecuencia"]}
            labelFormatter={(num) => `Número ${num}`}
          />
          <ReferenceLine y={expected} stroke="#f59e0b" strokeDasharray="3 3" />
          <Bar dataKey="count">
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.highlighted ? "#fbbf24" : "rgba(255,255,255,0.2)"}
                stroke={entry.highlighted ? "#f59e0b" : "transparent"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
