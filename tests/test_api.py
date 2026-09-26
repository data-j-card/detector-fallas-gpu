import json

import pytest
from fastapi.testclient import TestClient

from api import RAIZ, app

EJEMPLOS = RAIZ / "ejemplos"


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:  # el with ejecuta el lifespan: carga el modelo
        yield c


def cargar(nombre):
    return json.loads((EJEMPLOS / f"{nombre}.json").read_text())


def test_health_confirma_modelo_cargado(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "clases": ["degradacion_memoria", "falla_alimentacion", "normal",
                                                   "sobrecalentamiento"]}


@pytest.mark.parametrize("estado", ["normal", "sobrecalentamiento", "degradacion_memoria", "falla_alimentacion"])
def test_predecir_diagnostica_episodio_de_test(cliente, estado):
    nombre = next(p.stem for p in EJEMPLOS.glob(f"*_{estado}.json"))
    r = cliente.post("/predecir", json=cargar(nombre))
    assert r.status_code == 200
    assert r.json()["estado_predicho"] == estado and 0.9 < r.json()["confianza"] <= 1


def test_predecir_rechaza_lectura_fuera_de_rango_con_422(cliente):
    r = cliente.post("/predecir", json=cargar("5_fuera_de_rango"))
    assert r.status_code == 422 and "fuera del rango fisico" in r.json()["detail"][0]["msg"]


@pytest.mark.parametrize("romper", [lambda l: l[:29], lambda l: [{**x, "ecc_errors": "a"} for x in l]])
def test_predecir_rechaza_ventana_incompleta_o_mal_tipada_con_422(cliente, romper):
    lecturas = romper(cargar("1_normal")["lecturas"])
    assert cliente.post("/predecir", json={"lecturas": lecturas}).status_code == 422
