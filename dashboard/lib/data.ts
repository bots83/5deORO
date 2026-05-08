export type Sorteo = {
  fecha: string;
  dia_semana: string;
  n1: number; n2: number; n3: number; n4: number; n5: number;
  bolilla_extra: number | null;
  fuente: string;
};

export type Prediction = {
  fecha_dataset: string;
  n_sorteos_dataset: number;
  top5: number[];
  top10: number[];
  top15: number[];
  ensemble_probs: Record<string, number>;
  models_predictions: Record<string, number[]>;
  weights_used: Record<string, number>;
};

export type RandomnessTests = {
  chi_squared_uniform: { chi2: number; p_value: number; uniforme: boolean; min_count: number; max_count: number; esperado: number; df: number };
  gap_test_geometric: { chi2: number; p_value: number; geometrico: boolean; media_observada: number; media_esperada: number; n_gaps: number };
  autocorrelation_test: { max_acf_excluding_lag0: number; ljung_box_lag10_p: number; ljung_box_lag20_p: number; lags_significativos: number[]; iid: boolean };
  cooccurrence_chi2: { chi2: number; p_value: number; independientes: boolean; media_observada: number; media_esperada: number };
};

export async function loadSorteos(): Promise<Sorteo[]> {
  const r = await fetch("/data/sorteos.csv");
  const text = await r.text();
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map(line => {
    const values = line.split(",");
    const obj: any = {};
    headers.forEach((h, i) => { obj[h] = values[i]; });
    return {
      fecha: obj.fecha,
      dia_semana: obj.dia_semana,
      n1: +obj.n1, n2: +obj.n2, n3: +obj.n3, n4: +obj.n4, n5: +obj.n5,
      bolilla_extra: obj.bolilla_extra ? +obj.bolilla_extra : null,
      fuente: obj.fuente,
    } as Sorteo;
  });
}

export async function loadPrediction(): Promise<Prediction> {
  const r = await fetch("/data/prediction.json");
  return r.json();
}

export async function loadRandomness(): Promise<RandomnessTests> {
  const r = await fetch("/data/randomness.json");
  return r.json();
}

export function computeFrequencies(sorteos: Sorteo[]): Map<number, number> {
  const freq = new Map<number, number>();
  for (let i = 1; i <= 48; i++) freq.set(i, 0);
  for (const s of sorteos) {
    [s.n1, s.n2, s.n3, s.n4, s.n5].forEach(n => freq.set(n, (freq.get(n) || 0) + 1));
  }
  return freq;
}
