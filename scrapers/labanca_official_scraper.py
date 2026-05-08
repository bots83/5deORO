"""Scraper oficial de La Banca de Quinielas (Uruguay) usando FlareSolverr.

Estructura de la respuesta:
- Form POST a /resultados/cincodeoro con fecha_sorteo=YYYY-MM-DD-22:00
- Cada sorteo tiene <ul class="bolillas"> con 5 <li><img alt="NN"/></li>
  + 1 <li class="extra"><img alt="NN"/></li> = 5 números + Bolilla Extra
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.repository import Repository

FLARESOLVERR_URL = "http://localhost:8191/v1"
TARGET_URL = "https://www3.labanca.com.uy/resultados/cincodeoro"


def flaresolverr_get(url: str, timeout_ms: int = 60000) -> dict | None:
    try:
        r = requests.post(FLARESOLVERR_URL, json={
            "cmd": "request.get",
            "url": url,
            "maxTimeout": timeout_ms,
        }, timeout=(timeout_ms / 1000) + 30)
        d = r.json()
        if d.get("status") == "ok":
            return d["solution"]
    except Exception as e:
        print(f"  FlareSolverr error: {e}")
    return None


def flaresolverr_post(url: str, data: dict, session: str | None = None,
                       timeout_ms: int = 60000) -> dict | None:
    payload = {
        "cmd": "request.post",
        "url": url,
        "postData": "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in data.items()),
        "maxTimeout": timeout_ms,
    }
    if session:
        payload["session"] = session
    try:
        r = requests.post(FLARESOLVERR_URL, json=payload, timeout=(timeout_ms / 1000) + 30)
        d = r.json()
        if d.get("status") == "ok":
            return d["solution"]
    except Exception as e:
        print(f"  FlareSolverr POST error: {e}")
    return None


def create_session() -> str | None:
    try:
        r = requests.post(FLARESOLVERR_URL, json={"cmd": "sessions.create"}, timeout=60)
        return r.json().get("session")
    except Exception:
        return None


def destroy_session(session: str):
    try:
        requests.post(FLARESOLVERR_URL, json={
            "cmd": "sessions.destroy",
            "session": session,
        }, timeout=30)
    except Exception:
        pass


def parse_sorteo_html(html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    panel = soup.select_one("#panel-izquierdo")
    if not panel:
        return None

    # Fecha
    h2 = panel.find("h2")
    if not h2:
        return None
    fecha_text = h2.get_text(strip=True)
    # "Jueves 7 de Mayo de 2026"
    fecha = parse_spanish_date(fecha_text)
    if not fecha:
        return None

    # Bolillas: primer <ul class="bolillas"> tiene los 5 números + extra
    ul = panel.select_one("ul.bolillas")
    if not ul:
        return None

    nums = []
    bolilla_extra = None
    for li in ul.find_all("li"):
        img = li.find("img")
        if not img:
            continue
        alt = img.get("alt", "").strip()
        if not alt.isdigit():
            continue
        n = int(alt)
        if "extra" in (li.get("class") or []):
            bolilla_extra = n
        elif 1 <= n <= 48:
            nums.append(n)

    if len(nums) != 5 or len(set(nums)) != 5:
        return None

    return {
        "fecha": fecha,
        "numeros": sorted(nums),
        "bolilla_extra": bolilla_extra,
        "fuente": "labanca_official",
    }


def get_authenticity_token(html: str) -> str | None:
    # name puede aparecer antes o después de value, con espacios variables
    m = re.search(r'name="authenticity_token"[^>]*value="([^"]+)"', html)
    if m:
        return m.group(1)
    m = re.search(r'value="([^"]+)"[^>]*name="authenticity_token"', html)
    if m:
        return m.group(1)
    # CSRF meta
    m = re.search(r'<meta\s+name="csrf-token"\s+content="([^"]+)"', html)
    return m.group(1) if m else None


def parse_spanish_date(text: str) -> date | None:
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
        "octubre": 10, "noviembre": 11, "diciembre": 12,
    }
    text = text.lower()
    m = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", text)
    if m:
        d = int(m.group(1))
        mes = meses.get(m.group(2))
        y = int(m.group(3))
        if mes:
            return date(y, mes, d)
    return None


def get_initial_options() -> tuple[list[str], str | None]:
    """Obtiene fechas disponibles del select y authenticity_token."""
    sol = flaresolverr_get(TARGET_URL)
    if not sol:
        return [], None
    html = sol["response"]
    soup = BeautifulSoup(html, "lxml")
    options = []
    for o in soup.select("select#fecha_sorteo option"):
        v = o.get("value", "").strip()
        if v:
            options.append(v)
    token = get_authenticity_token(html)
    return options, token


def fetch_sorteo_by_date(fecha_value: str, session: str, token: str) -> dict | None:
    sol = flaresolverr_post(TARGET_URL, {
        "utf8": "✓",
        "authenticity_token": token,
        "fecha_sorteo": fecha_value,
        "commit": "Mostrar",
    }, session=session)
    if not sol:
        return None
    return parse_sorteo_html(sol["response"])


def scrape_all_dates(year_start: int = 2008, year_end: int = 2026, save_db: bool = False):
    """Scrapeа hacia atrás generando fechas posibles del juego.

    El 5 de Oro juega: lunes, miércoles, jueves, sábado, domingo (~5 veces/sem)
    Generamos todos esos días en el rango y probamos cada uno.
    """
    # Primero obtenemos las fechas iniciales del select
    print("Obteniendo lista inicial de fechas disponibles...")
    initial_options, token = get_initial_options()
    print(f"  Fechas en select: {len(initial_options)}")

    # Crear sesión persistente para reusar Cloudflare bypass
    session = create_session()
    if not session:
        print("  Error: no pudimos crear sesión FlareSolverr")
        return []

    print(f"  Sesión FlareSolverr creada: {session}")

    sorteos = []
    repo = Repository() if save_db else None
    if repo:
        repo.init()

    try:
        # Fase 1: scrappear las 50 fechas del select inicial
        print(f"\nFase 1: scrappeando {len(initial_options)} fechas iniciales...")
        for i, opt in enumerate(initial_options):
            # opt es algo como "2026-05-07-22:00"
            sorteo = fetch_sorteo_by_date(opt, session, token)
            if sorteo:
                sorteos.append(sorteo)
                if repo:
                    try:
                        repo.insert_sorteo(
                            sorteo["fecha"], sorteo["numeros"],
                            sorteo["bolilla_extra"], sorteo["fuente"],
                        )
                    except ValueError:
                        pass
                print(f"  [{i+1}/{len(initial_options)}] {sorteo['fecha']}: {sorteo['numeros']} + {sorteo['bolilla_extra']}")
            else:
                print(f"  [{i+1}/{len(initial_options)}] {opt}: no encontrado")
            time.sleep(0.3)

        # Fase 2: probar fechas hacia atrás (días que el juego se sortea)
        # Inferimos del patrón: los sorteos parecen ser lunes(0), miercoles(2), jueves(3), sabado(5), domingo(6)
        # Pero del select vimos: Jue, Lun, Mie, Dom, Jue, Lun, Mie, Dom, Jue, Lun
        # Entonces: Lun, Mie, Jue, Dom (4 sorteos por semana en promedio)

        if sorteos:
            oldest = min(s["fecha"] for s in sorteos)
            print(f"\nFase 2: scrappeando hacia atrás desde {oldest} hasta {year_start}-01-01")

            # Generar fechas candidatas (mar, mie, jue, dom = días con sorteo)
            current = oldest - timedelta(days=1)
            target_weekdays = {0, 2, 3, 5, 6}  # lun(0), mie(2), jue(3), sab(5), dom(6)
            consecutive_failures = 0
            max_consecutive = 30

            while current.year >= year_start:
                if current.weekday() in target_weekdays:
                    fecha_value = f"{current.isoformat()}-22:00"
                    sorteo = fetch_sorteo_by_date(fecha_value, session, token)
                    if sorteo:
                        sorteos.append(sorteo)
                        if repo:
                            try:
                                repo.insert_sorteo(
                                    sorteo["fecha"], sorteo["numeros"],
                                    sorteo["bolilla_extra"], sorteo["fuente"],
                                )
                            except ValueError:
                                pass
                        consecutive_failures = 0
                        if len(sorteos) % 10 == 0:
                            print(f"  Acumulado: {len(sorteos)} sorteos. Última fecha: {sorteo['fecha']}")
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive:
                            print(f"  {max_consecutive} fallos consecutivos en {current}. Deteniendo.")
                            break
                    time.sleep(0.2)
                current -= timedelta(days=1)

    finally:
        destroy_session(session)
        print(f"\nSesión destruída: {session}")

    return sorteos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--year-start", type=int, default=2008)
    parser.add_argument("--year-end", type=int, default=2026)
    args = parser.parse_args()

    sorteos = scrape_all_dates(year_start=args.year_start, year_end=args.year_end,
                                 save_db=args.save_db)
    print(f"\nTotal sorteos: {len(sorteos)}")
    if sorteos:
        sorteos.sort(key=lambda x: x["fecha"])
        print(f"Rango: {sorteos[0]['fecha']} → {sorteos[-1]['fecha']}")
