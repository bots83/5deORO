-- 5 de Oro de La Banca Uruguay
-- Reglas oficiales: elegir 5 números del 1 al 48 + 1 Bolilla Extra
CREATE TABLE IF NOT EXISTS sorteos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha       DATE NOT NULL UNIQUE,
    dia_semana  TEXT NOT NULL,
    n1          INTEGER NOT NULL CHECK(n1 BETWEEN 1 AND 48),
    n2          INTEGER NOT NULL CHECK(n2 BETWEEN 1 AND 48),
    n3          INTEGER NOT NULL CHECK(n3 BETWEEN 1 AND 48),
    n4          INTEGER NOT NULL CHECK(n4 BETWEEN 1 AND 48),
    n5          INTEGER NOT NULL CHECK(n5 BETWEEN 1 AND 48),
    bolilla_extra INTEGER CHECK(bolilla_extra BETWEEN 1 AND 48),
    suma_total  INTEGER GENERATED ALWAYS AS (n1+n2+n3+n4+n5) VIRTUAL,
    fuente      TEXT NOT NULL DEFAULT 'unknown',
    scraped_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK(n1 < n2 AND n2 < n3 AND n3 < n4 AND n4 < n5)
);

CREATE TABLE IF NOT EXISTS numeros_planos (
    sorteo_id   INTEGER NOT NULL REFERENCES sorteos(id) ON DELETE CASCADE,
    numero      INTEGER NOT NULL CHECK(numero BETWEEN 1 AND 48),
    posicion    INTEGER NOT NULL CHECK(posicion BETWEEN 1 AND 5),
    PRIMARY KEY (sorteo_id, posicion)
);

CREATE INDEX IF NOT EXISTS idx_sorteos_fecha ON sorteos(fecha);
CREATE INDEX IF NOT EXISTS idx_numeros_numero ON numeros_planos(numero);
CREATE INDEX IF NOT EXISTS idx_numeros_sorteo ON numeros_planos(sorteo_id);
