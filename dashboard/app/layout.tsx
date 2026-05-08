import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "5 de Oro — Análisis Predictivo",
  description: "Sistema de análisis estadístico y predicción para el 5 de Oro de La Banca Uruguay",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
