"""Scraper de LotteryTexts via FlareSolverr — historial completo desde 2016.

Estructura HTML: cada sorteo es un <section class="lottery-section lottery-logo">
con <p class="lottery-date"> y <ul class="lottery-balls"> con 5 nums + 1 ball-extra.

Iteramos por año via URLs /past-results/YYYY/ que devuelven los 20 más recientes.
También probamos /past-results/YYYY/MM/ por si hay más detalle.
"""
import argparse
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.repository import Repository

FLARESOLVERR_URL = "http://localhost:8191/v1"

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def fetch(url: str) -> str | None:
    try:
        r = requests.post(FLARESOLVERR_URL, json={
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 60000,
        }, timeout=120)
        d = r.json()
        if d.get("status") == "ok" and d["solution"].get("status") == 200:
            return d["solution"]["response"]
    except Exception as e:
        print(f"  Error: {e}")
    return None


def parse_results(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    sorteos = []

    sections = soup.select('section.lottery-section.lottery-logo')
    for sec in sections:
        # Solo sorteos del 5 de Oro
        heading = sec.select_one('.lottery-heading')
        if not heading or '5 De Oro' not in heading.get_text():
            continue

        date_p = sec.select_one('.lottery-date')
        if not date_p:
            continue

        # "Monday, May 04, 2026" or "Lunes, 04 May 2026" - parsear robusto
        date_text = date_p.get_text(strip=True)
        # Buscar el span con la fecha
        spans = date_p.find_all('span')
        for sp in spans:
            txt = sp.get_text(strip=True)
            # "May 04, 2026" or "4 may 2026"
            m = re.search(r'(\w{3,9})\s+(\d{1,2}),?\s+(\d{4})', txt) or \
                re.search(r'(\d{1,2})\s+(\w{3,9})\s+(\d{4})', txt)
            if m:
                groups = m.groups()
                if groups[0].isalpha():
                    mes_str, day, year = groups
                else:
                    day, mes_str, year = groups
                mes = MONTHS.get(mes_str.lower()[:3])
                if mes:
                    try:
                        fecha = date(int(year), mes, int(day))
                        break
                    except ValueError:
                        continue
        else:
            continue

        balls_ul = sec.select_one('ul.lottery-balls')
        if not balls_ul:
            continue

        nums = []
        bolilla = None
        for li in balls_ul.find_all('li'):
            txt = li.get_text(strip=True)
            try:
                n = int(txt)
            except ValueError:
                continue
            if not (1 <= n <= 48):
                continue
            classes = li.get('class', [])
            if any('ball-' in c or 'extra' in c.lower() for c in classes):
                bolilla = n
            else:
                nums.append(n)

        if len(nums) == 5 and len(set(nums)) == 5:
            sorteos.append({
                "fecha": fecha,
                "numeros": sorted(nums),
                "bolilla_extra": bolilla,
                "fuente": "lotterytexts",
            })

    return sorteos


def scrape_all_years(year_start: int = 2016, year_end: int = 2026, save_db: bool = True):
    repo = Repository() if save_db else None
    if repo:
        repo.init()

    all_sorteos = {}
    new_count = 0

    # Iterar por año (de más reciente a más antiguo)
    for year in range(year_end, year_start - 1, -1):
        url = f"https://lotterytexts.com/uruguay/5-de-oro/past-results/{year}/"
        print(f"\n[{year}] {url}")
        html = fetch(url)
        if not html:
            print(f"  Sin respuesta")
            continue

        sorteos = parse_results(html)
        print(f"  Sorteos encontrados en página: {len(sorteos)}")

        # También iterar por mes para agarrar más
        # Si el año tiene exactamente 20, probablemente hay más
        if len(sorteos) >= 18:  # cerca del límite, probar por mes
            for month in range(1, 13):
                month_url = f"https://lotterytexts.com/uruguay/5-de-oro/past-results/{year}/{month:02d}/"
                month_html = fetch(month_url)
                if month_html:
                    month_sorteos = parse_results(month_html)
                    if month_sorteos:
                        sorteos.extend(month_sorteos)
                time.sleep(1)

        # Insertar en DB
        for s in sorteos:
            if s["fecha"] in all_sorteos:
                continue
            all_sorteos[s["fecha"]] = s
            if repo:
                try:
                    if repo.insert_sorteo(
                        s["fecha"], s["numeros"], s["bolilla_extra"], s["fuente"]
                    ):
                        new_count += 1
                except ValueError:
                    pass

        print(f"  Total único acumulado: {len(all_sorteos)}, nuevos en DB: {new_count}")
        time.sleep(2)

    if repo:
        print(f"\nDB final: {repo.count()} sorteos")

    return list(all_sorteos.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--year-start", type=int, default=2016)
    parser.add_argument("--year-end", type=int, default=2026)
    args = parser.parse_args()

    sorteos = scrape_all_years(args.year_start, args.year_end, save_db=args.save_db)
    print(f"\nTotal: {len(sorteos)} sorteos únicos")
    if sorteos:
        sorteos.sort(key=lambda x: x["fecha"])
        print(f"Rango: {sorteos[0]['fecha']} → {sorteos[-1]['fecha']}")
