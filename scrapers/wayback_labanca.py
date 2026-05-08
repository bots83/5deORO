"""Scraper de Wayback Machine para extraer historial completo de labanca.

Cada snapshot tiene la página principal con un sorteo + 50 fechas en el select.
Iteramos por todos los snapshots y para cada uno, extraemos el sorteo de su fecha h2.
"""
import json
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.repository import Repository

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def parse_spanish_date(text: str):
    text = text.lower()
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", text)
    if m:
        d = int(m.group(1))
        mes = MESES_ES.get(m.group(2))
        y = int(m.group(3))
        if mes:
            try:
                return date(y, mes, d)
            except ValueError:
                pass
    return None


def parse_snapshot(html: str):
    """Extrae sorteos de un snapshot."""
    soup = BeautifulSoup(html, "lxml")
    sorteos = []

    panel = soup.select_one("#panel-izquierdo")
    if not panel:
        return sorteos

    # Sorteo principal mostrado
    h2 = panel.find("h2")
    if not h2:
        return sorteos
    fecha = parse_spanish_date(h2.get_text(strip=True))
    if not fecha:
        return sorteos

    ul = panel.select_one("ul.bolillas")
    if not ul:
        return sorteos

    nums = []
    bolilla = None
    for li in ul.find_all("li"):
        img = li.find("img")
        if not img:
            continue
        alt = img.get("alt", "").strip()
        if not alt.isdigit():
            continue
        n = int(alt)
        if "extra" in (li.get("class") or []):
            bolilla = n
        elif 1 <= n <= 48:
            nums.append(n)

    if len(nums) == 5 and len(set(nums)) == 5:
        sorteos.append({
            "fecha": fecha,
            "numeros": sorted(nums),
            "bolilla_extra": bolilla,
            "fuente": "labanca_wayback",
        })

    return sorteos


def main():
    print("Listando snapshots de Wayback...")
    r = requests.get(
        "https://web.archive.org/cdx/search/cdx?url=labanca.com.uy/resultados/cincodeoro&output=json&limit=500&matchType=prefix",
        timeout=60
    )
    data = json.loads(r.text)
    snapshots = [d for d in data[1:] if d[4] == "200"]
    print(f"Total snapshots 200 OK: {len(snapshots)}")

    repo = Repository()
    all_sorteos = {}
    inserted = 0

    for i, snap in enumerate(snapshots):
        timestamp = snap[1]
        original = snap[2]
        wayback_url = f"https://web.archive.org/web/{timestamp}/{original}"

        # Reintentar hasta 3 veces
        success = False
        for attempt in range(3):
            try:
                r = requests.get(wayback_url, timeout=120,
                                headers={"User-Agent": "Mozilla/5.0 Chrome/124"})
                if r.status_code == 200:
                    sorteos = parse_snapshot(r.text)
                    for s in sorteos:
                        if s["fecha"] not in all_sorteos:
                            all_sorteos[s["fecha"]] = s
                            try:
                                if repo.insert_sorteo(s["fecha"], s["numeros"], s["bolilla_extra"], s["fuente"]):
                                    inserted += 1
                            except ValueError:
                                pass
                    success = True
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 + attempt * 5)

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(snapshots)}] {timestamp}: total únicos={len(all_sorteos)}, nuevos en DB={inserted}", flush=True)
        time.sleep(1.0)

    print(f"\nTotal sorteos únicos extraídos: {len(all_sorteos)}")
    print(f"Nuevos en DB: {inserted}")
    print(f"DB total ahora: {repo.count()}")


if __name__ == "__main__":
    main()
