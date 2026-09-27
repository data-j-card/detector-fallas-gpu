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


def episodio(df, ep, n=300):
    """Primeras n lecturas del episodio, en el formato que recibe /predecir."""
    return df.filter(pl.col("episodio_id") == ep).sort("segundo").head(n).select(SENALES).to_dicts()


def estado(df, ep):
    return df.filter(pl.col("episodio_id") == ep)["estado"][0]


def test_health_confirma_modelo_cargado(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "clases": ["degradacion_memoria", "falla_alimentacion", "normal",
                                                   "sobrecalentamiento"]}


@pytest.mark.parametrize("ep", TEST)
def test_una_ventana_de_30_se_predice_sin_aviso(cliente, df, ep):
    r = cliente.post("/predecir", json={"lecturas": episodio(df, ep, 30)})
    assert r.status_code == 200
    assert r.json()["estado_predicho"] == estado(df, ep) and r.json()["aviso"] is None


@pytest.mark.parametrize("ep", TEST)
def test_episodio_completo_de_300_se_predice_sin_aviso(cliente, df, ep):
    r = cliente.post("/predecir", json={"lecturas": episodio(df, ep)})
    assert r.status_code == 200
    assert r.json()["estado_predicho"] == estado(df, ep) and r.json()["aviso"] is None


ROMPEN_CONTRATO = {
    "fuera de rango": ({"temp_c": 150.0}, "fuera del rango fisico"),
    "reloj no entero": ({"clock_mhz": 2412.5}, "fuera del rango fisico"),
    "vacio": ({"power_w": None}, "power_w vacio"),
    "texto": ({"util_pct": "88"}, "util_pct no es un numero"),
    "booleano": ({"clock_mhz": True}, "clock_mhz no es un numero"),
    "conteo decimal": ({"ecc_errors": 1.5}, "ecc_errors no es entero"),
}


@pytest.mark.parametrize("cambio, motivo", ROMPEN_CONTRATO.values(), ids=ROMPEN_CONTRATO.keys())
def test_lectura_que_rompe_el_contrato_descarta_su_ventana_con_aviso(cliente, df, cambio, motivo):
    lecturas = episodio(df, 13, 60)
    lecturas[41].update(cambio)
    r = cliente.post("/predecir", json={"lecturas": lecturas})
    assert r.status_code == 200 and r.json()["estado_predicho"] == "sobrecalentamiento"
    assert r.json()["aviso"] == f"1 de 2 ventanas descartadas: posicion 41: {motivo}"


def test_falta_un_campo_descarta_su_ventana_con_aviso(cliente, df):
    lecturas = episodio(df, 13, 60)
    del lecturas[5]["ecc_errors"]
    r = cliente.post("/predecir", json={"lecturas": lecturas})
    assert r.status_code == 200 and r.json()["aviso"] == "1 de 2 ventanas descartadas: posicion 5: falta ecc_errors"


def test_lecturas_sobrantes_y_campos_extra_quedan_en_el_aviso(cliente, df):
    lecturas = [{**l, "episodio_id": 24} for l in episodio(df, 24, 95)]
    r = cliente.post("/predecir", json={"lecturas": lecturas})
    assert r.status_code == 200 and r.json()["estado_predicho"] == "degradacion_memoria"
    assert r.json()["aviso"] == ("las ultimas 5 lecturas no completan una ventana de 30 y no se usaron"
                                 " | campos ignorados: ['episodio_id']")


def test_conteo_entero_escrito_como_decimal_se_acepta(cliente, df):
    lecturas = [{**l, "ecc_errors": float(l["ecc_errors"])} for l in episodio(df, 24, 30)]
    r = cliente.post("/predecir", json={"lecturas": lecturas})
    assert r.status_code == 200 and r.json()["aviso"] is None


class ModeloIntocable:
    def predict_proba(self, X):
        raise AssertionError("una peticion invalida llego al modelo")


def con(df, **cambios):
    """30 lecturas normales con la lectura 3 modificada."""
    lecturas = episodio(df, 0, 30)
    lecturas[3] = {**lecturas[3], **cambios}
    return lecturas


INVALIDAS = {
    "menos de 30 lecturas": (lambda df: {"lecturas": episodio(df, 0, 10)},
                             "llegaron 10 lecturas; se necesitan al menos 30"),
    "ninguna ventana valida": (lambda df: {"lecturas": con(df, temp_c=150.0)},
                               "ninguna ventana de 30 lecturas cumple el contrato: posicion 3: fuera del rango fisico"),
    "sin lista de lecturas": (lambda df: {"datos": episodio(df, 0, 30)}, "Field required"),
}


@pytest.mark.parametrize("cuerpo, mensaje", INVALIDAS.values(), ids=INVALIDAS.keys())
def test_sin_ventana_valida_responde_422_claro_sin_tocar_el_modelo(cliente, df, monkeypatch, cuerpo, mensaje):
    monkeypatch.setattr(app.state, "modelo", ModeloIntocable())
    r = cliente.post("/predecir", json=cuerpo(df))
    assert r.status_code == 422
    assert any(mensaje in e["msg"] for e in r.json()["detail"]), r.json()
