"""SQLite repository para el 5 de Oro de La Banca Uruguay.

Reglas oficiales: elegir 5 números del 1-48 + 1 Bolilla Extra del 1-48.
"""
import argparse
import csv
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "db" / "5deoro.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

DIAS = {0: "lunes", 1: "martes", 2: "miercoles", 3: "jueves",
        4: "viernes", 5: "sabado", 6: "domingo"}


@contextmanager
def get_conn(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class Repository:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init(self):
        schema = SCHEMA_PATH.read_text()
        with get_conn(self.db_path) as conn:
            conn.executescript(schema)
        print(f"DB inicializada en {self.db_path}")

    def insert_sorteo(
        self,
        fecha: date,
        numeros: list[int],
        bolilla_extra: Optional[int],
        fuente: str,
    ) -> bool:
        """Inserta un sorteo (5 números + bolilla extra)."""
        nums = sorted(numeros)
        if len(nums) != 5 or not all(1 <= n <= 48 for n in nums):
            raise ValueError(f"Esperados 5 números 1-48, recibido: {numeros}")
        if len(set(nums)) != 5:
            raise ValueError(f"Números duplicados: {numeros}")
        if bolilla_extra is not None and not (1 <= bolilla_extra <= 48):
            raise ValueError(f"Bolilla extra fuera de rango: {bolilla_extra}")

        dia = DIAS[fecha.weekday()]
        try:
            with get_conn(self.db_path) as conn:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO sorteos
                       (fecha, dia_semana, n1, n2, n3, n4, n5, bolilla_extra, fuente)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fecha.isoformat(), dia, *nums, bolilla_extra, fuente),
                )
                if cur.lastrowid and cur.rowcount > 0:
                    sorteo_id = cur.lastrowid
                    conn.executemany(
                        "INSERT INTO numeros_planos (sorteo_id, numero, posicion) VALUES (?, ?, ?)",
                        [(sorteo_id, n, i + 1) for i, n in enumerate(nums)],
                    )
                    return True
        except sqlite3.IntegrityError as e:
            print(f"  IntegrityError {fecha}: {e}")
        return False

    def count(self) -> int:
        with get_conn(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM sorteos").fetchone()[0]

    def get_all(self) -> list[dict]:
        with get_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT fecha, dia_semana, n1, n2, n3, n4, n5, bolilla_extra, fuente "
                "FROM sorteos ORDER BY fecha ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, output_path: str | Path):
        rows = self.get_all()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if not rows:
                print("No hay datos.")
                return
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exportados {len(rows)} sorteos a {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--export", type=str)
    args = parser.parse_args()

    repo = Repository()
    if args.init:
        repo.init()
    if args.count:
        print(f"Total sorteos: {repo.count()}")
    if args.export:
        repo.export_csv(args.export)
