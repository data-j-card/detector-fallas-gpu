import json

import joblib
import polars as pl
import pytest
from polars.testing import assert_frame_equal

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


def test_pipeline_calcula_las_mismas_features_que_el_entrenamiento(df, salida):
    X, _, grupos = construir_dataset(df)
    preparar = joblib.load(salida / "modelo.joblib")[:2]  # contrato -> features, como en la API
    crudo = df.sort("episodio_id", "segundo")
    for ep in range(48):  # primera ventana de cada episodio
        ventana = crudo.filter(pl.col("episodio_id") == ep).head(30).drop("episodio_id", "segundo", "estado")
        assert_frame_equal(preparar.transform(ventana), X[grupos.to_list().index(ep)])


@pytest.mark.parametrize("romper", [
    lambda v: v.with_columns(temp_c=pl.lit(500.0), power_w=pl.lit(-9999.0), ecc_errors=pl.lit(-3, pl.Int64)),
    lambda v: v.head(29),
])
def test_pipeline_rechaza_ventana_imposible_o_incompleta(df, salida, romper):
    modelo = joblib.load(salida / "modelo.joblib")
    with pytest.raises(ValueError):
        modelo.predict(romper(df.head(30).drop("episodio_id", "segundo", "estado")))


def test_pipeline_rechaza_features_en_otro_orden(df, salida):
    modelo = joblib.load(salida / "modelo.joblib")
    X = modelo[:2].transform(df.head(30).drop("episodio_id", "segundo", "estado"))
    with pytest.raises(ValueError, match="feature names"):
        modelo[2:].predict(X.select(reversed(X.columns)))


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
