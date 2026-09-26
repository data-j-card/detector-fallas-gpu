"""Contrato de datos: la telemetria que no lo cumple no entra al pipeline.

Columna faltante o tipo incorrecto -> se rechaza el lote completo.
Fila nula o fuera de rango fisico -> pandera la descarta y su ventana queda incompleta.
"""

from copy import copy

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field, PrivateAttr, model_validator

ESTADOS = ["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"]
TAMANO_VENTANA = 30
TOLERANCIA_W = 50.0  # negativos hasta -50 W son ruido del sensor en un colapso; se recortan a 0

TIPOS = {"episodio_id": pl.Int64, "segundo": pl.Int64, "temp_c": pl.Float64, "power_w": pl.Float64,
         "util_pct": pl.Float64, "clock_mhz": pl.Float64, "ecc_errors": pl.Int64, "estado": pl.String}


class Senales(pa.DataFrameModel):
    temp_c: float = pa.Field(in_range={"min_value": 0, "max_value": 120})
    power_w: float = pa.Field(ge=-TOLERANCIA_W)
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

    @pa.dataframe_check
    def par_unico(cls, data: pa.PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(~pl.struct("episodio_id", "segundo").is_duplicated())

    class Config:
        drop_invalid_rows = True


def validar(df: pl.DataFrame, modelo=Telemetria) -> pl.DataFrame:
    # drop_invalid_rows tambien deja pasar columnas faltantes y tipos malos:
    # la estructura se valida aparte, sin filas, con una copia que no descarta.
    estructura = copy(modelo.to_schema())
    estructura.drop_invalid_rows = False
    estructura.validate(df.head(0), lazy=True)
    return modelo.validate(df, lazy=True)


def validar_ventana(df: pl.DataFrame) -> pl.DataFrame:
    """Contrato para predecir: exactamente TAMANO_VENTANA lecturas y ninguna invalida.

    Primer paso del pipeline serializado; aqui no se descarta nada, se rechaza la ventana.
    """
    if df.height != TAMANO_VENTANA:
        raise ValueError(f"la ventana tiene {df.height} lecturas; se esperan {TAMANO_VENTANA}")
    valido = validar(df, Senales)
    if valido.height < df.height:
        raise ValueError(f"{df.height - valido.height} lecturas fuera del rango fisico")
    return valido


def leer_csv(ruta) -> pl.DataFrame:
    return pl.read_csv(ruta, schema_overrides=TIPOS)


class Lectura(BaseModel):
    temp_c: float
    power_w: float
    util_pct: float
    clock_mhz: float
    ecc_errors: int


class Ventana(BaseModel):
    """Entrada de la API: una ventana de lecturas consecutivas, en orden."""

    lecturas: list[Lectura] = Field(min_length=TAMANO_VENTANA, max_length=TAMANO_VENTANA)
    _df: pl.DataFrame = PrivateAttr()

    @model_validator(mode="after")
    def _contrato(self):
        self._df = validar_ventana(pl.DataFrame([l.model_dump() for l in self.lecturas]))
        return self

    @property
    def df(self) -> pl.DataFrame:
        return self._df
