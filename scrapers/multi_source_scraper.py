"""Scraper multifuente para el verdadero 5 de Oro de La Banca Uruguay.

Combina varias fuentes públicas (que no requieren bypass de Cloudflare):
- Lottoster.com — últimos ~20 sorteos con detalle
- Combinacionganadora.com — últimos ~20 sorteos
- Magayo.com — últimos ~30 sorteos
- Stats247.com — últimos 100 sorteos

Todas las fuentes muestran: 5 números del 1-48 + Bolilla Extra del 1-48.
"""
import argparse
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.repository import Repository

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "es-UY,es;q=0.9,en;q=0.7",
}

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
MESES_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_date(text: str) -> date | None:
    if not text:
        return None
    text = text.strip()
    # Formatos numéricos
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    # "4 de mayo de 2026" / "4 mayo 2026"
    m = re.match(r"(\d{1,2})\s+(?:de\s+)?(\w+)\s+(?:de\s+)?(\d{4})", text.lower())
    if m:
        d, mes, y = int(m.group(1)), m.group(2), int(m.group(3))
        if mes in MESES_ES:
            return date(y, MESES_ES[mes], d)
        if mes in MESES_EN:
            return date(y, MESES_EN[mes], d)
    # "29 April 2026"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text)
    if m:
        d, mes, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        if mes in MESES_EN:
            return date(y, MESES_EN[mes], d)
    return None


def fetch(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code} en {url}")
            return None
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Error: {e}")
    return None


# =============================================================================
# Lottoster.com
# =============================================================================

def scrape_lottoster() -> list[dict]:
    """Scrapea Lottoster — los últimos sorteos con detalle (5 nums + bolilla extra)."""
    print("[Lottoster] Scrapeando...")
    url = "https://www.lottoster.com/uy/5-de-oro/results/"
    html = fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    sorteos = []

    # Cada <ul class="numbers"> tiene 5 nums + 1 con prefijo E (extra)
    # Y cada uno está cerca de un texto con la fecha
    uls = soup.select("ul.numbers")

    # Buscar fechas asociadas. Lottoster muestra "X days ago" pero también enlaces /check/YYYY-MM-DD
    check_links = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"/check/(\d{4}-\d{2}-\d{2})/", a["href"])
        if m:
            try:
                check_links.append((datetime.strptime(m.group(1), "%Y-%m-%d").date(), a["href"]))
            except ValueError:
                continue
    # Únicos en orden
    check_dates = []
    seen = set()
    for d, _ in check_links:
        if d not in seen:
            seen.add(d)
            check_dates.append(d)

    # Mapear ULs con fechas — en orden de aparición ambos deberían corresponder
    # Si hay menos check_dates que uls, asumir que las uls extras son del display principal
    n = min(len(uls), len(check_dates))
    for i in range(n):
        ul = uls[i]
        fecha = check_dates[i]
        nums_normales = []
        bolilla = None
        for li in ul.find_all("li"):
            txt = li.get_text(strip=True)
            if txt.startswith("E"):
                try:
                    bolilla = int(txt[1:])
                except ValueError:
                    pass
            else:
                try:
                    nums_normales.append(int(txt))
                except ValueError:
                    pass
        if len(nums_normales) == 5 and all(1 <= n <= 48 for n in nums_normales):
            sorteos.append({
                "fecha": fecha,
                "numeros": sorted(nums_normales),
                "bolilla_extra": bolilla,
                "fuente": "lottoster",
            })

    print(f"  Lottoster: {len(sorteos)} sorteos")
    return sorteos


# =============================================================================
# Combinacionganadora.com
# =============================================================================

def scrape_combinacionganadora() -> list[dict]:
    print("[Combinacionganadora] Scrapeando...")
    url = "https://www.combinacionganadora.com/uy/5-de-oro/resultados/"
    html = fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    sorteos = []

    # Misma estructura que lottoster
    uls = soup.select("ul.numbers")
    check_dates = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"/check/(\d{4}-\d{2}-\d{2})/", a["href"])
        if m:
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                if d not in seen:
                    seen.add(d)
                    check_dates.append(d)
            except ValueError:
                continue

    n = min(len(uls), len(check_dates))
    for i in range(n):
        ul = uls[i]
        fecha = check_dates[i]
        nums = []
        bolilla = None
        for li in ul.find_all("li"):
            txt = li.get_text(strip=True)
            if txt.startswith("E"):
                try:
                    bolilla = int(txt[1:])
                except ValueError:
                    pass
            else:
                try:
                    nums.append(int(txt))
                except ValueError:
                    pass
        if len(nums) == 5 and all(1 <= n <= 48 for n in nums):
            sorteos.append({
                "fecha": fecha,
                "numeros": sorted(nums),
                "bolilla_extra": bolilla,
                "fuente": "combinacionganadora",
            })

    print(f"  Combinacionganadora: {len(sorteos)} sorteos")
    return sorteos


# =============================================================================
# Magayo.com
# =============================================================================

def scrape_magayo() -> list[dict]:
    print("[Magayo] Scrapeando...")
    url = "https://www.magayo.com/lotto/uruguay/5-de-oro-results/"
    html = fetch(url)
    if not html:
        return []

    sorteos = []
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="|")

    # Patrón: "29 April 2026|Wednesday|01 08 18 19 26|48"
    pattern = r"(\d{1,2}\s+\w+\s+\d{4})\|(\w+)\|((?:\d{2}\s*){5})Bolilla\s+Extra\s+(\d{1,2})"
    matches = re.findall(pattern, text)
    if not matches:
        # Pattern alternativo (con ​ separators o sin Bolilla en el texto entre numeros)
        pattern2 = r"(\d{1,2}\s+\w+\s+\d{4})\|(\w+)\|(\d{2}[\s​ ]+\d{2}[\s​ ]+\d{2}[\s​ ]+\d{2}[\s​ ]+\d{2})Bolilla\s+Extra\s+(\d{1,2})"
        matches = re.findall(pattern2, text)

    seen = set()
    for fecha_str, dia, nums_str, bolilla_str in matches:
        fecha = parse_date(fecha_str)
        if not fecha or fecha in seen:
            continue
        seen.add(fecha)
        nums = [int(x) for x in re.findall(r"\d+", nums_str) if 1 <= int(x) <= 48]
        if len(nums) != 5:
            continue
        try:
            bolilla = int(bolilla_str)
        except ValueError:
            bolilla = None
        sorteos.append({
            "fecha": fecha,
            "numeros": sorted(nums),
            "bolilla_extra": bolilla,
            "fuente": "magayo",
        })

    print(f"  Magayo: {len(sorteos)} sorteos")
    return sorteos


# =============================================================================
# Stats247.com
# =============================================================================

def scrape_stats247() -> list[dict]:
    print("[Stats247] Scrapeando...")
    url = "https://stats247.com/es/loto/uruguay-5-de-oro"
    html = fetch(url)
    if not html:
        return []

    sorteos = []
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    # La tabla 1 tiene la fecha + ul con números
    if len(tables) > 1:
        table = tables[1]
        rows = table.find_all("tr")
        for row in rows[1:]:  # saltar header
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            fecha_str = cells[0].get_text(strip=True)
            fecha = parse_date(fecha_str)
            if not fecha:
                continue

            # Buscar los <li> en la celda de números
            lis = cells[1].find_all("li")
            if not lis:
                continue

            nums = []
            bolilla = None
            for li in lis:
                txt = li.get_text(strip=True)
                # En stats247, el último li tiene class lg-reversed
                classes = li.get("class", [])
                try:
                    val = int(txt)
                except ValueError:
                    continue
                if not (1 <= val <= 48):
                    continue
                if "lg-reversed" in classes or "extra" in str(classes).lower():
                    bolilla = val
                else:
                    nums.append(val)

            if len(nums) == 5:
                sorteos.append({
                    "fecha": fecha,
                    "numeros": sorted(nums),
                    "bolilla_extra": bolilla,
                    "fuente": "stats247",
                })

    print(f"  Stats247: {len(sorteos)} sorteos")
    return sorteos


# =============================================================================
# Scraper principal
# =============================================================================

def scrape_all() -> list[dict]:
    """Combina todas las fuentes y deduplica por fecha."""
    all_sorteos: dict[date, dict] = {}
    # Prioridad: stats247 (más sorteos) > magayo > lottoster > combinacionganadora
    sources = [
        scrape_stats247,
        scrape_magayo,
        scrape_lottoster,
        scrape_combinacionganadora,
    ]

    for source_fn in sources:
        try:
            sorteos = source_fn()
        except Exception as e:
            print(f"  Error: {e}")
            continue
        for s in sorteos:
            if s["fecha"] not in all_sorteos:
                all_sorteos[s["fecha"]] = s

    result = sorted(all_sorteos.values(), key=lambda x: x["fecha"])
    return result


def cross_validate(sorteos_by_source: dict) -> None:
    """Compara fechas comunes entre fuentes para detectar discrepancias."""
    sources = list(sorteos_by_source.keys())
    if len(sources) < 2:
        return

    print("\n=== CROSS-VALIDATION ENTRE FUENTES ===")
    for i, s1 in enumerate(sources):
        for s2 in sources[i + 1:]:
            fechas_comunes = set(d["fecha"] for d in sorteos_by_source[s1]) & \
                            set(d["fecha"] for d in sorteos_by_source[s2])
            discrepancias = []
            for fecha in fechas_comunes:
                d1 = next(d for d in sorteos_by_source[s1] if d["fecha"] == fecha)
                d2 = next(d for d in sorteos_by_source[s2] if d["fecha"] == fecha)
                if d1["numeros"] != d2["numeros"]:
                    discrepancias.append((fecha, d1["numeros"], d2["numeros"]))
            print(f"  {s1} vs {s2}: {len(fechas_comunes)} fechas comunes, {len(discrepancias)} discrepancias")
            for d, n1, n2 in discrepancias[:3]:
                print(f"    {d}: {s1}={n1}, {s2}={n2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--cross-validate", action="store_true")
    args = parser.parse_args()

    if args.cross_validate:
        results_by_source = {
            "stats247": scrape_stats247(),
            "magayo": scrape_magayo(),
            "lottoster": scrape_lottoster(),
            "combinacionganadora": scrape_combinacionganadora(),
        }
        cross_validate(results_by_source)
        sorteos = sorted(
            {s["fecha"]: s for source in results_by_source.values() for s in source}.values(),
            key=lambda x: x["fecha"],
        )
    else:
        sorteos = scrape_all()

    print(f"\nTotal sorteos únicos: {len(sorteos)}")
    if sorteos:
        print(f"Rango fechas: {sorteos[0]['fecha']} → {sorteos[-1]['fecha']}")
        # Verificación
        all_nums = []
        for s in sorteos:
            all_nums.extend(s["numeros"])
        import numpy as np
        nums_arr = np.array(all_nums)
        print(f"Rango números: {nums_arr.min()}-{nums_arr.max()}")
        print(f"Total apariciones: {len(nums_arr)}")
        print(f"Sorteos x 5 nums: {len(sorteos)*5}")

    if args.save_db and sorteos:
        repo = Repository()
        repo.init()
        inserted = 0
        for s in sorteos:
            try:
                if repo.insert_sorteo(s["fecha"], s["numeros"], s["bolilla_extra"], s["fuente"]):
                    inserted += 1
            except ValueError as e:
                print(f"  Skip {s['fecha']}: {e}")
        print(f"DB: {inserted} insertados, total: {repo.count()}")
