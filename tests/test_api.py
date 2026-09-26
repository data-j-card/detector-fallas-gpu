import json

import polars as pl
import pytest
from fastapi.testclient import TestClient

from api import RAIZ, app

SENALES = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]
TEST = json.loads((RAIZ / "models" / "metricas.json").read_text())["episodios_test"]


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:  # el with ejecuta el lifespan: carga el modelo
        yield c


def ventana(df, ep):
    """Primeras 30 lecturas del episodio, en el formato que recibe /predecir."""
    return df.filter(pl.col("episodio_id") == ep).sort("segundo").head(30).select(SENALES).to_dicts()


def test_health_confirma_modelo_cargado(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "clases": ["degradacion_memoria", "falla_alimentacion", "normal",
                                                   "sobrecalentamiento"]}


@pytest.mark.parametrize("ep", TEST)
def test_predecir_diagnostica_episodios_de_test(cliente, df, ep):
    estado = df.filter(pl.col("episodio_id") == ep)["estado"][0]
    r = cliente.post("/predecir", json={"lecturas": ventana(df, ep)})
    assert r.status_code == 200
    assert r.json()["estado_predicho"] == estado and 0.9 < r.json()["confianza"] <= 1


class ModeloIntocable:
    def predict_proba(self, X):
        raise AssertionError("una peticion invalida llego al modelo")


def con(df, **cambios):
    """Una ventana normal con la lectura 3 modificada."""
    lecturas = ventana(df, 0)
    lecturas[3] = {**lecturas[3], **cambios}
    return lecturas


INVALIDAS = {
    "falta un campo": (lambda df: [{k: v for k, v in l.items() if k != "ecc_errors"} for l in ventana(df, 0)],
                       "Field required"),
    "campo desconocido": (lambda df: con(df, tempC=50.0), "Extra inputs are not permitted"),
    "numero como texto": (lambda df: con(df, temp_c="88.3"), "Input should be a valid number"),
    "conteo decimal": (lambda df: con(df, ecc_errors=1.0), "Input should be a valid integer"),
    "conteo booleano": (lambda df: con(df, ecc_errors=True), "Input should be a valid integer"),
    "valor nulo": (lambda df: con(df, power_w=None), "Input should be a valid number"),
    "2 lecturas": (lambda df: ventana(df, 0)[:2], "at least 30 items"),
    "10 lecturas": (lambda df: ventana(df, 0)[:10], "at least 30 items"),
    "31 lecturas": (lambda df: ventana(df, 0) * 2, "at most 30 items"),
    "fuera de rango": (lambda df: con(df, temp_c=150.0), "fuera del rango fisico en las posiciones [3]"),
}


@pytest.mark.parametrize("lecturas, mensaje", INVALIDAS.values(), ids=INVALIDAS.keys())
def test_peticion_invalida_responde_422_claro_sin_tocar_el_modelo(cliente, df, monkeypatch, lecturas, mensaje):
    monkeypatch.setattr(app.state, "modelo", ModeloIntocable())
    r = cliente.post("/predecir", json={"lecturas": lecturas(df)})
    assert r.status_code == 422
    assert any(mensaje in e["msg"] for e in r.json()["detail"]), r.json()
