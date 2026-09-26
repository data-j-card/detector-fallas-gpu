import polars as pl
import pytest

from schema import Telemetria

CORRUPTAS = [{"power_w": -80.0}, {"temp_c": 150.0}, {"util_pct": 101.0}, {"ecc_errors": -1},
             {"clock_mhz": 2412.5}, {"estado": "otro"}, {"temp_c": None}]


def validar(df):
    return Telemetria.validate(df, lazy=True)


def test_datos_reales_solo_fallan_en_potencia_negativa(df):
    descartadas = df.join(validar(df), on=["episodio_id", "segundo"], how="anti")
    assert descartadas.height == 3 and (descartadas["power_w"] < 0).all()


@pytest.mark.parametrize("valores", CORRUPTAS)
def test_fila_fuera_de_contrato_se_descarta(df, romper, valores):
    assert validar(romper(df, 5, 42, **valores)).height == validar(df).height - 1

