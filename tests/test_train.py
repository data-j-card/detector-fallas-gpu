import json

import joblib
import polars as pl
import pytest

from schema import Ventana
from features import construir_dataset
from train import entrenar, separar_test


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


def test_test_son_12_episodios_no_vistos_3_por_clase(df, salida):
    r = json.loads((salida / "metricas.json").read_text(encoding="utf-8"))
    train, test = set(r["episodios_entrenamiento"]), set(r["episodios_test"])
    assert len(train) == 36 and len(test) == 12 and not train & test
    clase = dict(df.select("episodio_id", "estado").unique().iter_rows())
    assert pl.Series([clase[e] for e in test]).value_counts()["count"].to_list() == [3, 3, 3, 3]


def test_test_reporta_ic95_por_episodio(salida):
    t = json.loads((salida / "metricas.json").read_text(encoding="utf-8"))["test"]
    bajo, alto = t["ic95_accuracy_episodio"]
    assert bajo <= t["accuracy_episodio"] <= alto and t["episodios_acertados"].endswith("/12")


def test_split_no_parte_episodios(df):
    _, y, grupos = construir_dataset(df)
    g = grupos.to_numpy()
    tr, te = separar_test(y.to_numpy(), g)
    assert not set(g[tr]) & set(g[te])
    assert all((g[te] == e).sum() == 10 for e in set(g[te]))  # cada episodio de test entra con sus 10 ventanas
