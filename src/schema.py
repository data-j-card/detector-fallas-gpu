"""Contrato de datos: la telemetria que no lo cumple no entra al pipeline.

Columna faltante o tipo incorrecto -> se rechaza el lote completo.
Fila nula o fuera de rango fisico -> pandera la descarta y su ventana queda incompleta.
"""

import pandera.polars as pa
import polars as pl

ESTADOS = ["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"]
TAMANO_VENTANA = 30

TIPOS = {"episodio_id": pl.Int64, "segundo": pl.Int64, "temp_c": pl.Float64, "power_w": pl.Float64,
         "util_pct": pl.Float64, "clock_mhz": pl.Float64, "ecc_errors": pl.Int64, "estado": pl.String}


class Senales(pa.DataFrameModel):
    temp_c: float = pa.Field(in_range={"min_value": 0, "max_value": 120})
    power_w: float = pa.Field(gt=0)
    util_pct: float = pa.Field(in_range={"min_value": 0, "max_value": 100})
    clock_mhz: float = pa.Field(gt=0)
    ecc_errors: int = pa.Field(ge=0)

    @pa.check("clock_mhz")
    def mhz_entero(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(pl.col(data.key) % 1 == 0)

    class Config:
        drop_invalid_rows = True


class Telemetria(Senales):
    episodio_id: int = pa.Field(ge=0)
    segundo: int = pa.Field(ge=0)
    estado: str = pa.Field(isin=ESTADOS)

    class Config:
        unique = ["episodio_id", "segundo"]
        drop_invalid_rows = True


def leer_csv(ruta) -> pl.DataFrame:
    return pl.read_csv(ruta, schema_overrides=TIPOS)

