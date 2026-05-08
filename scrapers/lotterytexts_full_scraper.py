"""Scraper FULL de LotteryTexts iterando por mes (year/MM/)."""
import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from scrapers.lotterytexts_scraper import parse_results, fetch
from db.repository import Repository


def scrape_full(year_start=2016, year_end=2026, save_db=True):
    repo = Repository() if save_db else None
    if repo:
        repo.init()

    seen = set()
    new_count = 0

    # Iterar todos los años + meses
    for year in range(year_end, year_start - 1, -1):
        for month in range(12, 0, -1):
            # No fetchear meses futuros
            from datetime import date as Date
            today = Date.today()
            if year > today.year or (year == today.year and month > today.month):
                continue

            url = f"https://lotterytexts.com/uruguay/5-de-oro/past-results/{year}/{month:02d}/"
            html = fetch(url)
            if not html:
                continue

            sorteos = parse_results(html)
            new_in_month = 0
            for s in sorteos:
                key = s["fecha"]
                if key in seen:
                    continue
                seen.add(key)
                if repo:
                    try:
                        if repo.insert_sorteo(
                            s["fecha"], s["numeros"], s["bolilla_extra"], s["fuente"]
                        ):
                            new_count += 1
                            new_in_month += 1
                    except ValueError:
                        pass

            print(f"[{year}-{month:02d}] {len(sorteos)} sorteos en página, {new_in_month} nuevos. Total únicos: {len(seen)}, nuevos en DB: {new_count}")
            time.sleep(1.0)

    if repo:
        print(f"\nDB final: {repo.count()} sorteos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--year-start", type=int, default=2016)
    parser.add_argument("--year-end", type=int, default=2026)
    args = parser.parse_args()
    scrape_full(args.year_start, args.year_end, save_db=args.save_db)
