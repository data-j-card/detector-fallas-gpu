from pathlib import Path

import polars as pl
import pytest

from schema import leer_csv

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def df():
    return leer_csv(RAIZ / "data" / "telemetria_publica.csv")


@pytest.fixture
def romper():
    def _romper(df, ep, seg, **valores):
        fila = (pl.col("episodio_id") == ep) & (pl.col("segundo") == seg)
        return df.with_columns(pl.when(fila).then(pl.lit(v)).otherwise(pl.col(c)).alias(c) for c, v in valores.items())
    return _romper
