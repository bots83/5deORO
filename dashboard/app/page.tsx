"use client";
import { useEffect, useState } from "react";
import { Sorteo, Prediction, RandomnessTests, loadSorteos, loadPrediction, loadRandomness } from "@/lib/data";
import PredictionCard from "@/components/PredictionCard";
import FrequencyChart from "@/components/FrequencyChart";
import RandomnessTestsCard from "@/components/RandomnessTests";
import RecentDraws from "@/components/RecentDraws";
import SimulationView from "@/components/SimulationView";

export default function Home() {
  const [sorteos, setSorteos] = useState<Sorteo[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [tests, setTests] = useState<RandomnessTests | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([loadSorteos(), loadPrediction(), loadRandomness()])
      .then(([s, p, t]) => {
        setSorteos(s);
        setPrediction(p);
        setTests(t);
        setLoading(false);
      })
      .catch(err => { console.error(err); setLoading(false); });
  }, []);

  if (loading) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <div className="text-slate-400">Cargando...</div>
      </main>
    );
  }

  return (
    <main className="min-h-screen p-6 max-w-7xl mx-auto">
      <header className="mb-8 text-center pt-8">
        <h1 className="text-5xl font-bold mb-2">
          <span className="gold-text">5 de Oro</span>
        </h1>
        <p className="text-slate-400">Análisis estadístico y predicción ML — La Banca Uruguay</p>
        <p className="text-xs text-slate-500 mt-2">
          ⚠ Sistema de investigación. El juego es estadísticamente aleatorio. Juega responsablemente.
        </p>
      </header>

      {prediction && <PredictionCard prediction={prediction} />}

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <RecentDraws sorteos={sorteos} />
        {tests && <RandomnessTestsCard tests={tests} />}
      </div>

      <FrequencyChart sorteos={sorteos} highlight={prediction?.top5 || []} />

      <SimulationView sorteos={sorteos} prediction={prediction} />

      <footer className="mt-12 mb-6 text-center text-xs text-slate-500">
        <p>Dataset: {sorteos.length} sorteos verificados ({sorteos[0]?.fecha} → {sorteos[sorteos.length - 1]?.fecha})</p>
        <p className="mt-1">Cross-validation entre 4 fuentes: 0 discrepancias.</p>
      </footer>
    </main>
  );
}
