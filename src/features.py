"""Ventana -> features -> etiqueta.

`features` es la unica funcion que calcula features: la usan el entrenamiento
y la API. Cada feature sale solo de las filas de su ventana, sin estadisticos
del dataset, asi que no hay ajuste que filtre informacion de test.
"""

import logging

import polars as pl

from schema import TAMANO_VENTANA, validar

UMBRAL_TEMP_C = 83.0  # zona en que las GPU NVIDIA empiezan a bajar el reloj
UMBRAL_UTIL_ALTA = 90.0

log = logging.getLogger(__name__)
t = pl.int_range(pl.len())


def _div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b != 0).then(a / b).otherwise(0.0)


def _corr(a: str, b: str) -> pl.Expr:
    return pl.corr(a, b).fill_nan(0.0)  # serie constante -> correlacion indefinida


EXPRESIONES = [
    e
    for c in ["temp_c", "power_w", "util_pct", "clock_mhz"]
    for e in (
        pl.col(c).mean().alias(f"{c}_mean"),
        pl.col(c).std().alias(f"{c}_std"),
        pl.col(c).min().alias(f"{c}_min"),
        pl.col(c).max().alias(f"{c}_max"),
        (pl.cov(t, c) / t.var()).alias(f"{c}_pendiente"),
        pl.col(c).diff().abs().mean().alias(f"{c}_dif_abs"),
    )
] + [
    _div(pl.col("power_w").std(), pl.col("power_w").mean()).alias("power_w_cv"),
    _div(pl.col("clock_mhz").std(), pl.col("clock_mhz").mean()).alias("clock_mhz_cv"),
    pl.col("ecc_errors").sum().alias("ecc_total"),
    (pl.col("ecc_errors") > 0).mean().alias("ecc_frac_con_error"),
    pl.col("ecc_errors").max().alias("ecc_max"),
    (pl.cov(t, "ecc_errors") / t.var()).alias("ecc_pendiente"),
    (pl.col("temp_c") >= UMBRAL_TEMP_C).mean().alias("temp_frac_sobre_umbral"),
    (pl.col("util_pct") >= UMBRAL_UTIL_ALTA).mean().alias("util_frac_alta"),
    _div(pl.col("temp_c").mean(), pl.col("power_w").mean()).alias("temp_por_watt"),
    _div(pl.col("power_w").mean(), pl.col("util_pct").mean()).alias("watt_por_util"),
    _div(pl.col("clock_mhz").mean(), pl.col("util_pct").mean()).alias("clock_por_util"),
    _corr("temp_c", "clock_mhz").alias("corr_temp_clock"),
    _corr("power_w", "clock_mhz").alias("corr_power_clock"),
    _corr("util_pct", "power_w").alias("corr_util_power"),
    _corr("power_w", "temp_c").alias("corr_power_temp"),
]


def features(df: pl.DataFrame, por: list[str] | None = None) -> pl.DataFrame:
    """Una fila de features por grupo `por`; sin `por`, el df entero es una ventana (API)."""
    df = df.with_columns(pl.col("power_w").clip(lower_bound=0))
    return df.group_by(por, maintain_order=True).agg(EXPRESIONES) if por else df.select(EXPRESIONES)


def construir_dataset(df: pl.DataFrame, tamano: int = TAMANO_VENTANA):
    """X, y, grupos con una fila por ventana completa y valida de cada episodio.

    X no lleva episodio_id, segundo ni estado; `grupos` es para StratifiedGroupKFold.
    """
    valido = validar(df)
    if valido.group_by("episodio_id").agg(pl.col("estado").n_unique()).get_column("estado").max() > 1:
        raise ValueError("hay episodios con mas de un estado")

    claves = ["episodio_id", "ventana", "estado"]
    v = (valido.sort("episodio_id", "segundo")
         .with_columns(ventana=pl.col("segundo") // tamano)
         .filter(pl.len().over("episodio_id", "ventana") == tamano))
    f = features(v, claves)

    esperadas = df.select(pl.struct("episodio_id", pl.col("segundo") // tamano).n_unique()).item()
    log.info("filas descartadas: %d | ventanas descartadas: %d | lecturas de potencia recortadas a 0: %d",
             df.height - valido.height, esperadas - f.height, v.filter(pl.col("power_w") < 0).height)
    return f.drop(claves), f["estado"], f["episodio_id"]
