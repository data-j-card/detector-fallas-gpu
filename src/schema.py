"""Contrato de datos: la telemetria que no lo cumple no entra al pipeline.

Columna faltante o tipo incorrecto -> se rechaza el lote completo.
Fila nula o fuera de rango fisico -> pandera la descarta y su ventana queda incompleta.
"""

import math
from copy import copy

import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, PrivateAttr, model_validator

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


def posiciones_invalidas(df: pl.DataFrame) -> list[int]:
    """Posiciones de las lecturas que violan el contrato: fuera de rango o vacias."""
    valido = validar(df.with_row_index("posicion"), Senales)
    return sorted(set(range(df.height)) - set(valido["posicion"]))


def validar_ventana(df: pl.DataFrame) -> pl.DataFrame:
    """Contrato para predecir: exactamente TAMANO_VENTANA lecturas y ninguna invalida.

    Primer paso del pipeline serializado; aqui no se descarta nada, se rechaza la ventana.
    """
    if df.height != TAMANO_VENTANA:
        raise ValueError(f"la ventana tiene {df.height} lecturas; se esperan {TAMANO_VENTANA}")
    if malas := posiciones_invalidas(df):
        raise ValueError(f"lecturas invalidas en las posiciones {malas}")
    return df


def leer_csv(ruta) -> pl.DataFrame:
    return pl.read_csv(ruta, schema_overrides=TIPOS)


SENALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]
MAX_DETALLE = 10  # lecturas invalidas que se detallan en el mensaje


def _valor(lectura: dict, campo: str):
    """(valor, motivo): el numero del campo, o None y la razon por la que rompe el contrato."""
    if campo not in lectura:
        return None, f"falta {campo}"
    v = lectura[campo]
    if v is None:
        return None, f"{campo} vacio"
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None, f"{campo} no es un numero"
    if campo == "ecc_errors":
        return (int(v), None) if v == int(v) else (None, f"{campo} no es entero")
    return float(v), None


def _detalle(motivos: dict[int, str]) -> str:
    items = [f"posicion {p}: {m}" for p, m in sorted(motivos.items())]
    resto = len(items) - MAX_DETALLE
    return "; ".join(items[:MAX_DETALLE]) + (f"; y {resto} mas" if resto > 0 else "")


class Lectura(BaseModel):
    """Una lectura de un segundo. Un valor que rompe el contrato llega aqui como null."""

    temp_c: float | None
    power_w: float | None
    util_pct: float | None
    clock_mhz: float | None
    ecc_errors: int | None


class Peticion(BaseModel):
    """Entrada de la API: lecturas consecutivas de una GPU, una por segundo, en orden.

    Se cortan en ventanas de TAMANO_VENTANA. Una lectura que rompe el contrato (falta un campo,
    viene vacio, no es un numero o esta fuera del rango fisico) descarta su ventana, igual que en
    el entrenamiento, y queda en el aviso. Solo se rechaza la peticion si no hay ventana valida.
    """

    lecturas: list[Lectura]
    _ventanas: list[pl.DataFrame] = PrivateAttr()
    _aviso: str | None = PrivateAttr()

    @model_validator(mode="wrap")
    @classmethod
    def _cortar_en_ventanas(cls, data, handler):
        crudas = data.get("lecturas") if isinstance(data, dict) else None
        if not isinstance(crudas, list):
            return handler(data)  # sin lista de lecturas: pydantic responde el error de estructura

        n = len(crudas)
        if n < TAMANO_VENTANA:
            raise ValueError(f"llegaron {n} lecturas; se necesitan al menos {TAMANO_VENTANA}")

        limpias, motivos, ignorados = [], {}, set()
        for pos, lectura in enumerate(crudas):
            if not isinstance(lectura, dict):
                limpias.append(dict.fromkeys(SENALES))
                motivos[pos] = "no es un objeto"
                continue
            ignorados |= lectura.keys() - set(SENALES)
            valores = {c: _valor(lectura, c) for c in SENALES}
            limpias.append({c: v for c, (v, _) in valores.items()})
            if errores := [m for _, m in valores.values() if m]:
                motivos[pos] = ", ".join(errores)
        self = handler({"lecturas": limpias})

        df = pl.DataFrame(limpias, schema={c: TIPOS[c] for c in SENALES})
        for pos in posiciones_invalidas(df):
            motivos.setdefault(pos, "fuera del rango fisico")
        inicios = range(0, n - n % TAMANO_VENTANA, TAMANO_VENTANA)
        rotas = {i for i in inicios for p in motivos if i <= p < i + TAMANO_VENTANA}
        self._ventanas = [df.slice(i, TAMANO_VENTANA) for i in inicios if i not in rotas]
        en_rotas = {p: m for p, m in motivos.items() if p < inicios[-1] + TAMANO_VENTANA}
        if not self._ventanas:
            raise ValueError(f"ninguna ventana de {TAMANO_VENTANA} lecturas cumple el contrato: {_detalle(en_rotas)}")

        avisos = []
        if rotas:
            avisos.append(f"{len(rotas)} de {len(inicios)} ventanas descartadas: {_detalle(en_rotas)}")
        if n % TAMANO_VENTANA:
            avisos.append(f"las ultimas {n % TAMANO_VENTANA} lecturas no completan una ventana de "
                          f"{TAMANO_VENTANA} y no se usaron")
        if ignorados:
            avisos.append(f"campos ignorados: {sorted(ignorados)}")
        self._aviso = " | ".join(avisos) or None
        return self

    @property
    def ventanas(self) -> list[pl.DataFrame]:
        return self._ventanas

    @property
    def aviso(self) -> str | None:
        return self._aviso
