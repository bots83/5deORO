"""Bypass agresivo de Cloudflare en labanca.com.uy con varias estrategias.

Estrategias en orden:
1. FlareSolverr local (si Docker disponible)
2. Wayback Machine snapshots con HTML completo
3. curl_cffi con sesiones persistentes y delays
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


def fetch_via_flaresolverr(url: str, flaresolverr_url: str = "http://localhost:8191/v1") -> dict | None:
    """FlareSolverr ejecuta un browser real headless para resolver Cloudflare."""
    try:
        r = requests.post(flaresolverr_url, json={
            "cmd": "request.get",
            "url": url,
            "maxTimeout": 60000,
        }, timeout=80)
        if r.status_code == 200:
            data = r.json()
            if data.get("solution", {}).get("status") == 200:
                return data["solution"]
    except Exception as e:
        print(f"  FlareSolverr no disponible: {e}")
    return None


def fetch_wayback_snapshots(url_pattern: str, max_snapshots: int = 50) -> list[dict]:
    """Lista snapshots de Wayback con HTML completo (no solo el challenge)."""
    cdx_url = f"https://web.archive.org/cdx/search/cdx?url={url_pattern}&output=json&limit={max_snapshots}"
    try:
        r = requests.get(cdx_url, timeout=30)
        data = json.loads(r.text)
        snapshots = []
        for row in data[1:]:
            timestamp = row[1]
            original = row[2]
            statuscode = row[4]
            length = int(row[6]) if row[6].isdigit() else 0
            if statuscode == "200" and length > 5000:  # Filtrar challenges (suelen ser ~3KB)
                snapshots.append({
                    "timestamp": timestamp,
                    "original": original,
                    "length": length,
                    "wayback_url": f"https://web.archive.org/web/{timestamp}/{original}",
                })
        return snapshots
    except Exception as e:
        print(f"  Error CDX: {e}")
        return []


def parse_labanca_html(html: str) -> list[dict]:
    """Parsea HTML de labanca.com.uy para extraer sorteos del 5 de Oro."""
    soup = BeautifulSoup(html, "lxml")
    sorteos = []

    # La estructura HTML puede variar entre snapshots históricos
    # Buscar tablas con números 1-48 y fechas
    text = soup.get_text(separator="\n")

    # Patrón típico: fecha + 5 numeros + bolilla
    patterns = [
        r"(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+\D*(\d{1,2})",
        r"(\d{1,2}-\d{1,2}-\d{2,4}).*?(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            try:
                fecha_str = m.group(1)
                fecha = None
                for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y"):
                    try:
                        fecha = datetime.strptime(fecha_str, fmt).date()
                        break
                    except ValueError:
                        pass
                if not fecha:
                    continue

                nums = [int(m.group(i)) for i in range(2, 7)]
                if not all(1 <= n <= 48 for n in nums) or len(set(nums)) != 5:
                    continue

                bolilla = None
                if len(m.groups()) >= 7:
                    try:
                        b = int(m.group(7))
                        if 1 <= b <= 48 and b not in nums:
                            bolilla = b
                    except ValueError:
                        pass

                sorteos.append({
                    "fecha": fecha,
                    "numeros": sorted(nums),
                    "bolilla_extra": bolilla,
                    "fuente": "labanca_wayback",
                })
            except Exception:
                continue
    return sorteos


def try_all_methods():
    target_url = "https://www3.labanca.com.uy/resultados/cincodeoro"

    print("=" * 70)
    print("BYPASS CLOUDFLARE — labanca.com.uy/resultados/cincodeoro")
    print("=" * 70)

    # Método 1: FlareSolverr
    print("\n[Método 1] FlareSolverr local...")
    sol = fetch_via_flaresolverr(target_url)
    if sol:
        sorteos = parse_labanca_html(sol["response"])
        print(f"  ✓ FlareSolverr: {len(sorteos)} sorteos extraídos")
        if sorteos:
            return sorteos

    # Método 2: Wayback Machine snapshots
    print("\n[Método 2] Wayback Machine snapshots...")
    patterns = [
        "labanca.com.uy/resultados/cincodeoro*",
        "labanca.com.uy/cincodeoro*",
        "www3.labanca.com.uy/resultados/cincodeoro*",
        "www.labanca.com.uy/resultados/cincodeoro*",
    ]
    all_snapshots = []
    for pattern in patterns:
        snaps = fetch_wayback_snapshots(pattern, max_snapshots=200)
        all_snapshots.extend(snaps)
        print(f"  Pattern '{pattern}': {len(snaps)} snapshots")

    # Snapshots únicos por timestamp
    unique = {s["timestamp"]: s for s in all_snapshots}
    print(f"\n  Total snapshots únicos: {len(unique)}")

    # Descargar y parsear
    all_sorteos = {}
    for ts, snap in sorted(unique.items())[:50]:  # primeros 50
        try:
            r = requests.get(snap["wayback_url"], timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 Chrome/124.0.0.0"})
            if r.status_code == 200:
                sorteos = parse_labanca_html(r.text)
                for s in sorteos:
                    if s["fecha"] not in all_sorteos:
                        all_sorteos[s["fecha"]] = s
                print(f"    [{ts}] {len(sorteos)} sorteos parseados (acumulado: {len(all_sorteos)})")
            time.sleep(1)
        except Exception as e:
            print(f"    [{ts}] error: {e}")

    return list(all_sorteos.values())


if __name__ == "__main__":
    sorteos = try_all_methods()
    print(f"\nTotal sorteos rescatados: {len(sorteos)}")
    if sorteos:
        sorteos.sort(key=lambda x: x["fecha"])
        print(f"Rango: {sorteos[0]['fecha']} → {sorteos[-1]['fecha']}")
