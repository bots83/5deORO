"""Scraper para Lottolyzer.com — historial completo paginado del 5 de Oro.

Estructura real:
- 6 números principales (Winning No.) en rango 1-48
- 2 números supplementary (Supp No.)
- Sum, From Last, etc. son derivados
"""
import argparse
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.repository import Repository

BASE_URL = "https://en.lottolyzer.com/history/uruguay/5-de-oro/page/{page}/per-page/50/summary-view"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://en.lottolyzer.com/",
}


def fetch(url: str, max_retries: int = 3) -> str | None:
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code} en {url}")
            return None
        except requests.RequestException as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  Error: {e}")
    return None


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return []

    sorteos = []
    rows = table.find_all("tr")
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue

        try:
            draw_num = int(cells[0])
        except (ValueError, IndexError):
            continue

        try:
            fecha = datetime.strptime(cells[1].strip(), "%Y-%m-%d").date()
        except ValueError:
            continue

        # Winning No.: 6 números separados por coma
        winning_str = cells[2]
        winning_nums = [int(x) for x in re.findall(r"\d+", winning_str) if 1 <= int(x) <= 48]
        if len(winning_nums) != 6:
            continue
        if len(set(winning_nums)) != 6:
            continue

        # Supplementary: 2 números
        supp_str = cells[3]
        supp_nums = [int(x) for x in re.findall(r"\d+", supp_str) if 1 <= int(x) <= 48]

        # Validación con la columna Sum (cells[5] si existe)
        if len(cells) > 5:
            try:
                expected_sum = int(cells[5])
                if expected_sum != sum(winning_nums):
                    print(f"  ⚠ Sum mismatch en draw {draw_num}: {expected_sum} vs {sum(winning_nums)}")
                    continue
            except (ValueError, IndexError):
                pass

        sorteos.append({
            "draw": draw_num,
            "fecha": fecha,
            "numeros": sorted(winning_nums),
            "supplementary": sorted(supp_nums[:2]),
            "fuente": "lottolyzer",
        })
    return sorteos


def scrape_all(max_pages: int = 100, delay: float = 2.0, save_raw: bool = True) -> list[dict]:
    all_sorteos = []
    seen_draws = set()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scraping Lottolyzer — hasta {max_pages} páginas...")
    for page in tqdm(range(1, max_pages + 1), desc="Páginas"):
        url = BASE_URL.format(page=page)
        html = fetch(url)
        if not html:
            print(f"  Página {page}: sin respuesta, deteniendo.")
            break

        if save_raw:
            (RAW_DIR / f"lottolyzer_page{page:03d}.html").write_text(html, encoding="utf-8")

        sorteos = parse_page(html)
        if not sorteos:
            print(f"  Página {page}: sin datos, deteniendo.")
            break

        nuevos = 0
        for s in sorteos:
            if s["draw"] not in seen_draws:
                seen_draws.add(s["draw"])
                all_sorteos.append(s)
                nuevos += 1

        if nuevos == 0:
            print(f"  Página {page}: todos duplicados, deteniendo.")
            break

        time.sleep(delay)

    all_sorteos.sort(key=lambda x: x["fecha"])
    return all_sorteos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--no-raw", action="store_true")
    args = parser.parse_args()

    sorteos = scrape_all(max_pages=args.max_pages, delay=args.delay, save_raw=not args.no_raw)
    print(f"\nTotal sorteos: {len(sorteos)}")
    if sorteos:
        print(f"Rango: {sorteos[0]['fecha']} → {sorteos[-1]['fecha']}")
        # Verificación
        all_nums = []
        for s in sorteos:
            all_nums.extend(s["numeros"])
        import numpy as np
        nums_arr = np.array(all_nums)
        print(f"Rango de números: {nums_arr.min()}-{nums_arr.max()}")
        print(f"Total apariciones por número:")
        for n in range(1, 49):
            cnt = int((nums_arr == n).sum())
            print(f"  {n:2d}: {cnt}")

    if args.save_db and sorteos:
        repo = Repository()
        repo.init()
        inserted = 0
        skipped = 0
        for s in sorteos:
            try:
                if repo.insert_sorteo(
                    s["fecha"], s["numeros"], s["supplementary"],
                    s["fuente"], draw_num=s["draw"],
                ):
                    inserted += 1
                else:
                    skipped += 1
            except ValueError as e:
                print(f"  Skip {s['fecha']}: {e}")
                skipped += 1
        print(f"DB: {inserted} nuevos, {skipped} duplicados/skipped. Total: {repo.count()}")
