"""Scraper para LotteryTexts.com — historial completo 5 de Oro Uruguay."""
import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from scrapers.base_scraper import BaseScraper, ScraperError
from db.repository import Repository

BASE_URL = "https://lotterytexts.com/lottery/uruguay-5-de-oro/"
RESULTS_URL = "https://lotterytexts.com/lottery/uruguay-5-de-oro/results/"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def parse_date(text: str) -> date | None:
    text = text.strip()
    for fmt in ("%d %B %Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_numbers(text: str) -> list[int] | None:
    nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", text) if 1 <= int(x) <= 48]
    return nums if len(nums) >= 5 else None


class LotteryTextsScraper(BaseScraper):
    def __init__(self, save_raw: bool = True, **kwargs):
        super().__init__(rate_limit=2.5, **kwargs)
        self.save_raw = save_raw

    def _get_all_pages(self) -> list[str]:
        """Descarga todas las páginas de resultados."""
        pages_html = []
        # Intenta primero la página de resultados completa
        urls_to_try = [
            RESULTS_URL,
            BASE_URL,
            "https://lotterytexts.com/lottery/5-de-oro-uruguay/",
            "https://lotterytexts.com/lottery/5-de-oro-uruguay/results/",
        ]
        for url in urls_to_try:
            try:
                print(f"  Probando: {url}")
                html = self.fetch(url)
                if html and len(html) > 1000:
                    pages_html.append(html)
                    # Buscar paginación
                    soup = BeautifulSoup(html, "lxml")
                    pagination = soup.select("a[href*='page'], .pagination a, a.next, a[rel='next']")
                    visited = {url}
                    for link in pagination:
                        href = link.get("href", "")
                        if href and href not in visited:
                            abs_url = href if href.startswith("http") else f"https://lotterytexts.com{href}"
                            visited.add(abs_url)
                            try:
                                page_html = self.fetch(abs_url)
                                pages_html.append(page_html)
                            except ScraperError:
                                pass
                    break
            except ScraperError as e:
                print(f"  Error: {e}")
                continue
        return pages_html

    def _parse_sorteos(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        sorteos = []

        # Buscar tablas de resultados
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue
                # Intentar parsear fecha en la primera celda
                fecha = parse_date(cells[0])
                if not fecha:
                    continue
                # Buscar números en las celdas restantes
                nums_text = " ".join(cells[1:])
                nums = parse_numbers(nums_text)
                if not nums or len(nums) < 5:
                    continue
                sorteos.append({
                    "fecha": fecha,
                    "numeros": sorted(nums[:5]),
                    "bonus": nums[5] if len(nums) > 5 else None,
                    "fuente": "lotteryTexts",
                })

        # Si no hay tablas, buscar patrones de texto
        if not sorteos:
            text = soup.get_text(separator="\n")
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Patrón: fecha seguida de números
                date_match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4})\b", line)
                if not date_match:
                    continue
                fecha = parse_date(date_match.group(1))
                if not fecha:
                    continue
                remainder = line[date_match.end():]
                nums = parse_numbers(remainder)
                if nums and len(nums) >= 5:
                    sorteos.append({
                        "fecha": fecha,
                        "numeros": sorted(nums[:5]),
                        "bonus": nums[5] if len(nums) > 5 else None,
                        "fuente": "lotteryTexts",
                    })
        return sorteos

    def run(self) -> list[dict]:
        print("Iniciando scraper LotteryTexts.com...")
        pages = self._get_all_pages()
        if not pages:
            print("No se pudo obtener datos de LotteryTexts.com")
            return []

        all_sorteos = []
        seen_fechas = set()
        for i, html in enumerate(pages):
            if self.save_raw:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                (RAW_DIR / f"lotteryTexts_page{i+1}.html").write_text(html, encoding="utf-8")
            sorteos = self._parse_sorteos(html)
            for s in sorteos:
                if s["fecha"] not in seen_fechas:
                    seen_fechas.add(s["fecha"])
                    all_sorteos.append(s)

        all_sorteos.sort(key=lambda x: x["fecha"])
        print(f"  Parseados: {len(all_sorteos)} sorteos únicos")
        return all_sorteos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper LotteryTexts.com — 5 de Oro Uruguay")
    parser.add_argument("--save-db", action="store_true", help="Guardar resultados en SQLite")
    parser.add_argument("--no-raw", action="store_true", help="No guardar HTML crudo")
    args = parser.parse_args()

    scraper = LotteryTextsScraper(save_raw=not args.no_raw)
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
        print(f"DB: {inserted} nuevos insertados, {len(sorteos)-inserted} ya existían. Total: {repo.count()}")
    elif not sorteos:
        print("No se obtuvieron datos.")
        sys.exit(1)
