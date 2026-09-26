import polars as pl
import pytest

from pandera.errors import SchemaErrors
from pydantic import ValidationError

from schema import Ventana, validar

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


def test_api_acepta_ventana_valida_y_rechaza_fuera_de_rango_o_incompleta(df):
    lecturas = df.head(30).drop("episodio_id", "segundo", "estado").to_dicts()
    assert Ventana(lecturas=lecturas).df.height == 30
    with pytest.raises(ValidationError):
        Ventana(lecturas=[{**lecturas[0], "temp_c": 150.0}] + lecturas[1:])
    with pytest.raises(ValidationError):
        Ventana(lecturas=lecturas[:29])


def test_segundo_duplicado_descarta_ambas_copias(df):
    dup = df.filter((pl.col("episodio_id") == 5) & (pl.col("segundo") == 42))
    assert validar(pl.concat([df, dup])).height == validar(df).height - 1
