"""Scraper para Loteria.Guru — actualización incremental de sorteos recientes."""
import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from scrapers.base_scraper import BaseScraper, ScraperError
from db.repository import Repository

URLS = [
    "https://loteria.guru/resultados-loteria-uruguay/uy-5-de-oro",
    "https://loteria.guru/resultados/uruguay/5-de-oro",
    "https://lotteryguru.com/uruguay-lottery-results/5-de-oro",
]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def parse_date_es(text: str) -> date | None:
    text = text.strip().lower()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    # "4 de mayo de 2026"
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", text)
    if m:
        day, mes, year = int(m.group(1)), MESES_ES.get(m.group(2)), int(m.group(3))
        if mes:
            return date(year, mes, day)
    return None


def parse_numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\b(\d{1,2})\b", text) if 1 <= int(x) <= 48]


class LoteriaGuruScraper(BaseScraper):
    def __init__(self, use_playwright: bool = False, **kwargs):
        super().__init__(rate_limit=2.0, **kwargs)
        self.use_playwright = use_playwright

    def _fetch_html(self, url: str) -> str | None:
        try:
            if self.use_playwright:
                return self.fetch_with_playwright(url)
            return self.fetch(url)
        except ScraperError as e:
            print(f"  Error HTTP en {url}: {e}")
            return None

    def _parse(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        sorteos = []
        seen = set()

        # Buscar divs con resultados (estructura común en loteria.guru)
        result_containers = soup.select(
            ".result, .draw, .lottery-result, .draw-result, "
            "[class*='result'], [class*='draw'], article"
        )

        for container in result_containers:
            text = container.get_text(separator=" ", strip=True)
            fecha = parse_date_es(text)
            if not fecha:
                continue
            nums = parse_numbers(text)
            if len(nums) < 5:
                continue
            if fecha not in seen:
                seen.add(fecha)
                sorteos.append({
                    "fecha": fecha,
                    "numeros": sorted(nums[:5]),
                    "bonus": nums[5] if len(nums) > 5 else None,
                    "fuente": "loteriaGuru",
                })

        # Fallback: buscar en tablas
        if not sorteos:
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if not cells:
                        continue
                    fecha = parse_date_es(cells[0]) if cells else None
                    if not fecha:
                        continue
                    nums = parse_numbers(" ".join(cells[1:]))
                    if len(nums) >= 5 and fecha not in seen:
                        seen.add(fecha)
                        sorteos.append({
                            "fecha": fecha,
                            "numeros": sorted(nums[:5]),
                            "bonus": nums[5] if len(nums) > 5 else None,
                            "fuente": "loteriaGuru",
                        })

        # Fallback: línea por línea en texto plano
        if not sorteos:
            for line in soup.get_text(separator="\n").split("\n"):
                fecha = parse_date_es(line)
                if not fecha:
                    continue
                nums = parse_numbers(line)
                if len(nums) >= 5 and fecha not in seen:
                    seen.add(fecha)
                    sorteos.append({
                        "fecha": fecha,
                        "numeros": sorted(nums[:5]),
                        "bonus": nums[5] if len(nums) > 5 else None,
                        "fuente": "loteriaGuru",
                    })

        return sorteos

    def run(self) -> list[dict]:
        print("Iniciando scraper Loteria.Guru...")
        for url in URLS:
            print(f"  Probando: {url}")
            html = self._fetch_html(url)
            if not html or len(html) < 500:
                continue
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / "loteriaGuru_raw.html").write_text(html, encoding="utf-8")
            sorteos = self._parse(html)
            if sorteos:
                sorteos.sort(key=lambda x: x["fecha"])
                print(f"  Parseados: {len(sorteos)} sorteos de {url}")
                return sorteos
            else:
                print(f"  Sin datos parseados en {url}, activando Playwright...")
                html = self.fetch_with_playwright(url)
                if html:
                    sorteos = self._parse(html)
                    if sorteos:
                        sorteos.sort(key=lambda x: x["fecha"])
                        print(f"  Parseados via Playwright: {len(sorteos)} sorteos")
                        return sorteos
        print("No se pudo obtener datos de Loteria.Guru")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Loteria.Guru — sorteos recientes")
    parser.add_argument("--save-db", action="store_true", help="Guardar en SQLite")
    parser.add_argument("--playwright", action="store_true", help="Forzar uso de Playwright")
    args = parser.parse_args()

    scraper = LoteriaGuruScraper(use_playwright=args.playwright)
    sorteos = scraper.run()

    if args.save_db and sorteos:
        repo = Repository()
        repo.init()
        inserted = 0
        for s in sorteos:
            try:
                if repo.insert_sorteo(s["fecha"], s["numeros"], s["bonus"], s["fuente"]):
                    inserted += 1
            except ValueError as e:
                print(f"  Skipping {s['fecha']}: {e}")
        print(f"DB: {inserted} nuevos, total: {repo.count()}")
    elif not sorteos:
        print("Sin datos obtenidos.")
        sys.exit(1)
