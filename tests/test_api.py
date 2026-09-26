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


class ModeloIntocable:
    def predict_proba(self, X):
        raise AssertionError("una peticion invalida llego al modelo")


def con(**cambios):
    """La ventana normal con la lectura 3 modificada."""
    lecturas = cargar("1_normal")["lecturas"]
    lecturas[3] = {**lecturas[3], **cambios}
    return lecturas


def sin_ecc():
    return [{k: v for k, v in l.items() if k != "ecc_errors"} for l in cargar("1_normal")["lecturas"]]


INVALIDAS = {
    "falta un campo": (sin_ecc, "Field required"),
    "campo desconocido": (lambda: con(tempC=50.0), "Extra inputs are not permitted"),
    "numero como texto": (lambda: con(temp_c="88.3"), "Input should be a valid number"),
    "conteo decimal": (lambda: con(ecc_errors=1.0), "Input should be a valid integer"),
    "conteo booleano": (lambda: con(ecc_errors=True), "Input should be a valid integer"),
    "valor nulo": (lambda: con(power_w=None), "Input should be a valid number"),
    "2 lecturas": (lambda: cargar("1_normal")["lecturas"][:2], "at least 30 items"),
    "10 lecturas": (lambda: cargar("1_normal")["lecturas"][:10], "at least 30 items"),
    "31 lecturas": (lambda: cargar("1_normal")["lecturas"] * 2, "at most 30 items"),
    "fuera de rango": (lambda: cargar("5_fuera_de_rango")["lecturas"], "fuera del rango fisico en las posiciones [10]"),
}


@pytest.mark.parametrize("lecturas, mensaje", INVALIDAS.values(), ids=INVALIDAS.keys())
def test_peticion_invalida_responde_422_claro_sin_tocar_el_modelo(cliente, monkeypatch, lecturas, mensaje):
    monkeypatch.setattr(app.state, "modelo", ModeloIntocable())
    r = cliente.post("/predecir", json={"lecturas": lecturas()})
    assert r.status_code == 422
    assert any(mensaje in e["msg"] for e in r.json()["detail"]), r.json()
