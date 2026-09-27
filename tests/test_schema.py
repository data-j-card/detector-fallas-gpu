import polars as pl
import pytest

from pandera.errors import SchemaErrors
from pydantic import ValidationError

from schema import Peticion, validar

CORRUPTAS = [{"power_w": -80.0}, {"temp_c": 150.0}, {"util_pct": 101.0}, {"ecc_errors": -1},
             {"clock_mhz": 2412.5}, {"estado": "otro"}, {"temp_c": None}]


def test_datos_reales_cumplen_el_contrato(df):
    assert validar(df).height == df.height


@pytest.mark.parametrize("valores", CORRUPTAS)
def test_fila_fuera_de_contrato_se_descarta(df, romper, valores):
    assert validar(romper(df, 5, 42, **valores)).height == validar(df).height - 1



@pytest.mark.parametrize("roto", [lambda d: d.drop("ecc_errors"),
                                  lambda d: d.with_columns(pl.col("ecc_errors").cast(pl.String))])
def test_estructura_invalida_rechaza_el_lote(df, roto):
    with pytest.raises(SchemaErrors):
        validar(roto(df))


def test_peticion_corta_ventanas_y_descarta_la_que_tiene_una_lectura_invalida(df):
    lecturas = df.head(95).drop("episodio_id", "segundo", "estado").to_dicts()
    lecturas[41] = {**lecturas[41], "temp_c": 150.0}
    p = Peticion(lecturas=lecturas)
    assert [v.height for v in p.ventanas] == [30, 30]  # 3 ventanas completas, la segunda descartada
    assert "posicion 41: fuera del rango fisico" in p.aviso and "ultimas 5 lecturas" in p.aviso


def test_peticion_rechaza_menos_de_30_lecturas(df):
    with pytest.raises(ValidationError, match="llegaron 29 lecturas"):
        Peticion(lecturas=df.head(29).drop("episodio_id", "segundo", "estado").to_dicts())


def test_segundo_duplicado_descarta_ambas_copias(df):
    dup = df.filter((pl.col("episodio_id") == 5) & (pl.col("segundo") == 42))
    assert validar(pl.concat([df, dup])).height == validar(df).height - 1
