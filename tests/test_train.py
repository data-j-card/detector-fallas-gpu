import json

import joblib
import polars as pl
import pytest

from schema import Ventana
from train import entrenar


@pytest.fixture(scope="module")
def salida(tmp_path_factory):
    ruta = tmp_path_factory.mktemp("models")
    entrenar(salida=ruta)
    return ruta


def test_pipeline_serializado_diagnostica_ventana_cruda(df, salida):
    modelo = joblib.load(salida / "modelo.joblib")
    for (estado,), g in df.sort("episodio_id", "segundo").group_by("estado", maintain_order=True):
        v = Ventana(lecturas=g.head(30).drop("episodio_id", "segundo", "estado").to_dicts())
        assert modelo.predict(v.df)[0] == estado


def test_pipeline_rechaza_features_en_otro_orden(df, salida):
    modelo = joblib.load(salida / "modelo.joblib")
    clasificador = modelo[1:]
    X = modelo[0].transform(df.head(30).drop("episodio_id", "segundo", "estado"))
    with pytest.raises(ValueError, match="feature names"):
        clasificador.predict(X.select(reversed(X.columns)))


def test_reporte_de_metricas(salida):
    r = json.loads((salida / "metricas.json").read_text(encoding="utf-8"))
    assert r["ventanas"] == 480 and r["episodios"] == 48 and len(r["features"]) == 39
    assert set(r["metricas"]) == {"accuracy_ventana", "f1_macro_ventana", "log_loss_ventana", "accuracy_episodio"}
