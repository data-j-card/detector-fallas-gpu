"""Genera los JSON de ejemplo para /predecir con la primera ventana de un episodio de test por clase.

Son episodios que el modelo no vio al entrenar (models/metricas.json -> episodios_test).
Uso: python ejemplos/generar.py
"""

import json
from pathlib import Path

import polars as pl

from schema import TAMANO_VENTANA, leer_csv

RAIZ = Path(__file__).resolve().parents[1]
SENALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]
EPISODIOS = {"1_normal": 0, "2_sobrecalentamiento": 13, "3_degradacion_memoria": 24, "4_falla_alimentacion": 38}


def ventana(df: pl.DataFrame, ep: int) -> list[dict]:
    return df.filter(pl.col("episodio_id") == ep).sort("segundo").head(TAMANO_VENTANA).select(SENALES).to_dicts()


def guardar(nombre: str, lecturas: list[dict]):
    # una lectura por linea, como en el enunciado
    filas = ",\n".join(f"    {json.dumps(l)}" for l in lecturas)
    (RAIZ / "ejemplos" / f"{nombre}.json").write_text('{\n  "lecturas": [\n' + filas + "\n  ]\n}\n")


if __name__ == "__main__":
    df = leer_csv(RAIZ / "data" / "telemetria_publica.csv")
    for nombre, ep in EPISODIOS.items():
        guardar(nombre, ventana(df, ep))
    rota = ventana(df, 13)
    rota[10]["temp_c"] = 150.0  # fuera del rango fisico 0-120 C: la API responde 422
    guardar("5_fuera_de_rango", rota)
