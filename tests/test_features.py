import polars as pl
from polars.testing import assert_frame_equal

from features import construir_dataset, features


def test_ventanas_sin_ids_ni_nulos(df):
    X, y, grupos = construir_dataset(df)
    assert X.height == 480 and grupos.value_counts()["count"].unique().to_list() == [10]
    assert not {"episodio_id", "segundo", "estado"} & set(X.columns)
    assert X.null_count().sum_horizontal().item() == 0 and not X.select(pl.all().is_nan().any()).row(0)[0]


def test_train_y_api_calculan_lo_mismo(df):
    X, _, _ = construir_dataset(df)
    ventana = df.sort("episodio_id", "segundo").head(30).drop("episodio_id", "segundo", "estado")
    assert_frame_equal(features(ventana), X.head(1))


def test_fila_corrupta_descarta_solo_su_ventana(df, romper):
    assert construir_dataset(romper(df, 5, 42, temp_c=150.0))[0].height == construir_dataset(df)[0].height - 1


def test_segundo_duplicado_descarta_su_ventana(df):
    dup = df.filter((pl.col("episodio_id") == 5) & (pl.col("segundo") == 42))
    assert construir_dataset(pl.concat([df, dup]))[0].height == construir_dataset(df)[0].height - 1


def test_negativo_tolerable_se_recorta_a_cero(df):
    X, _, grupos = construir_dataset(df)
    fila = grupos.to_list().index(36) + 107 // 30  # episodio 36, segundo 107: -13.5 W
    assert X["power_w_min"][fila] == 0.0


def test_cada_fila_es_una_ventana_contigua_de_un_solo_episodio(df):
    X, y, grupos = construir_dataset(df)
    crudo = df.sort("episodio_id", "segundo").with_columns(ventana=pl.col("segundo") // 30)
    for i, ((ep, w), v) in enumerate(crudo.group_by("episodio_id", "ventana", maintain_order=True)):
        assert (grupos[i], y[i]) == (ep, v["estado"][0]) and v["estado"].n_unique() == 1
        assert v["segundo"].to_list() == list(range(w * 30, w * 30 + 30))
        assert_frame_equal(features(v.drop("episodio_id", "segundo", "estado", "ventana")), X[i])


def test_el_orden_de_las_filas_de_entrada_no_cambia_x(df):
    assert construir_dataset(df.sample(fraction=1.0, shuffle=True, seed=1))[0].equals(construir_dataset(df)[0])
