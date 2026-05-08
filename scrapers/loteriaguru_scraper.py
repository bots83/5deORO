"""Scraper de Loteria.Guru — 13 páginas × 25 sorteos ≈ 325 sorteos.

Estructura:
- URL paginada: /resultados-loteria-uruguay/uy-5-de-oro/resultados-anteriores-5-de-oro-uy?page={1..13}
- Cada sorteo tiene .lg-date con la fecha y números en algún elemento.
"""
import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.repository import Repository

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "es-UY,es;q=0.9",
}

MESES_ES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}


def fetch(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.text
        print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    return None


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    sorteos = []

    # Cada sorteo: <div class="columns ... lg-line"> con
    #   .lg-date.has-text-right con "DD mes" en strong y año en texto
    #   ul.lg-numbers-small con li.lg-number (5x) + li.lg-reversed (1x)
    rows = soup.select("div.lg-line")
    for row in rows:
        date_div = row.select_one(".lg-date.has-text-right")
        if not date_div:
            continue
        strong = date_div.find("strong")
        if not strong:
            continue
        day_month = strong.get_text(strip=True).split()
        if len(day_month) != 2:
            continue
        try:
            day = int(day_month[0])
            mes = MESES_ES.get(day_month[1].lower()[:3])
        except ValueError:
            continue
        if not mes:
            continue
        # Año: texto del div sin el strong
        full_text = date_div.get_text(separator=" ", strip=True)
        m = re.search(r"\b(20\d{2})\b", full_text)
        if not m:
            continue
        year = int(m.group(1))
        try:
            fecha = date(year, mes, day)
        except ValueError:
            continue

        # Números: ul.lg-numbers-small dentro del row
        ul = row.select_one("ul.lg-numbers-small")
        if not ul:
            continue

        nums = []
        bolilla = None
        for li in ul.find_all("li"):
            txt = li.get_text(strip=True)
            try:
                n = int(txt)
            except ValueError:
                continue
            if not (1 <= n <= 48):
                continue
            classes = li.get("class", [])
            if "lg-reversed" in classes:
                bolilla = n
            else:
                nums.append(n)

        if len(nums) == 5 and len(set(nums)) == 5:
            sorteos.append({
                "fecha": fecha,
                "numeros": sorted(nums),
                "bolilla_extra": bolilla,
                "fuente": "loteriaguru",
            })

    return sorteos


def scrape_all_pages(max_pages: int = 20, save_db: bool = True):
    repo = Repository() if save_db else None
    if repo:
        repo.init()

    seen = set()
    new_count = 0

    for page in range(1, max_pages + 1):
        url = f"https://loteria.guru/resultados-loteria-uruguay/uy-5-de-oro/resultados-anteriores-5-de-oro-uy?page={page}"
        print(f"\n[Page {page}] {url}")
        html = fetch(url)
        if not html:
            print("  Sin respuesta, deteniendo.")
            break

        sorteos = parse_page(html)
        new_in_page = 0
        for s in sorteos:
            if s["fecha"] in seen:
                continue
            seen.add(s["fecha"])
            if repo:
                try:
                    if repo.insert_sorteo(
                        s["fecha"], s["numeros"], s["bolilla_extra"], s["fuente"]
                    ):
                        new_count += 1
                        new_in_page += 1
                except ValueError:
                    pass

        print(f"  {len(sorteos)} sorteos en página, {new_in_page} nuevos. Total únicos: {len(seen)}, nuevos en DB: {new_count}")

        if not sorteos:
            print("  Sin sorteos, deteniendo.")
            break
        time.sleep(1.5)

    if repo:
        print(f"\nDB final: {repo.count()} sorteos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()
    scrape_all_pages(args.max_pages, save_db=args.save_db)
