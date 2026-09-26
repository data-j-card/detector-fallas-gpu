"""Entrena el clasificador de fallas, lo evalua en episodios que no vio y lo serializa.

48 episodios -> 36 de entrenamiento y validacion (9 por clase)
             -> 12 de test (3 por clase), que solo se usan una vez, al final.

El artefacto va de la ventana cruda (30 lecturas) a la clase:
contrato -> features -> StandardScaler -> LogisticRegression, ajustado sobre los 36.

Uso: python src/train.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import construir_dataset, features
from schema import TAMANO_VENTANA, leer_csv, validar_ventana

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "telemetria_publica.csv"
MODELOS = RAIZ / "models"
SEMILLA = 42
N_FOLDS, REPETICIONES = 3, 10

log = logging.getLogger(__name__)


def clasificador():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def separar_test(y: np.ndarray, grupos: np.ndarray):
    """Indices de train y test por episodio: 1 de cada 4 episodios de cada clase va a test."""
    return next(StratifiedGroupKFold(4, shuffle=True, random_state=SEMILLA).split(np.zeros(len(y)), y, grupos))


def _metricas(m, X: pl.DataFrame, y: np.ndarray, grupos: np.ndarray) -> dict:
    clases = list(m.classes_)
    p = m.predict_proba(X)
    pred = m.classes_[p.argmax(1)]
    # diagnostico del episodio: promedio de las probabilidades de sus ventanas
    ep = pl.DataFrame(p, schema=clases).with_columns(ep=grupos, y=y).group_by("ep", "y").mean()
    p_ep = ep.select(clases).to_numpy()
    return {
        "accuracy_ventana": accuracy_score(y, pred),
        "f1_macro_ventana": f1_score(y, pred, average="macro"),
        "log_loss_ventana": log_loss(y, p, labels=clases),
        "accuracy_episodio": accuracy_score(ep["y"], m.classes_[p_ep.argmax(1)]),
        "prob_min_episodio": p_ep.max(1).min(),  # el diagnostico menos seguro del modelo
    }


def validacion_cruzada(X: pl.DataFrame, y: np.ndarray, grupos: np.ndarray) -> dict:
    """StratifiedGroupKFold repetido: ningun episodio cae en train y validacion a la vez."""
    folds = [_metricas(clasificador().fit(X[tr], y[tr]), X[va], y[va], grupos[va])
             for semilla in range(REPETICIONES)
             for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=semilla).split(X, y, grupos)]
    resumen = {}
    for k in folds[0]:
        v = [f[k] for f in folds]
        resumen[k] = {"media": np.mean(v), "p2_5": np.percentile(v, 2.5), "p97_5": np.percentile(v, 97.5)}
    return resumen


def evaluar_test(m, X: pl.DataFrame, y: np.ndarray, grupos: np.ndarray) -> dict:
    """Metricas en episodios no vistos, con IC 95% exacto de la accuracy por episodio.

    El IC se calcula sobre episodios y no sobre ventanas: las ventanas de un mismo
    episodio no son independientes y darian un intervalo falsamente estrecho.
    """
    r = _metricas(m, X, y, grupos)
    n = len(np.unique(grupos))
    aciertos = round(r["accuracy_episodio"] * n)
    ic = binomtest(aciertos, n).proportion_ci(0.95, method="exact")
    return {**r, "episodios_acertados": f"{aciertos}/{n}", "ic95_accuracy_episodio": [ic.low, ic.high]}


def _redondear(x):
    if isinstance(x, dict):
        return {k: _redondear(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_redondear(v) for v in x]
    return round(float(x), 4) if isinstance(x, (float, np.floating)) else x


def entrenar(datos: Path = DATOS, salida: Path = MODELOS) -> dict:
    X, y, grupos = construir_dataset(leer_csv(datos))  # X nunca lleva la columna estado
    y, grupos = y.to_numpy(), grupos.to_numpy()
    tr, te = separar_test(y, grupos)

    validacion = validacion_cruzada(X[tr], y[tr], grupos[tr])
    ajustado = clasificador().fit(X[tr], y[tr])
    test = evaluar_test(ajustado, X[te], y[te], grupos[te])  # y de test solo se usa para puntuar

    modelo = Pipeline([("contrato", FunctionTransformer(validar_ventana)),
                       ("features", FunctionTransformer(features)),
                       *ajustado.steps])
    salida.mkdir(exist_ok=True)
    joblib.dump(modelo, salida / "modelo.joblib")
    reporte = _redondear({
        "modelo": "contrato -> features -> StandardScaler -> LogisticRegression",
        "tamano_ventana_s": TAMANO_VENTANA,
        "clases": list(ajustado.classes_),
        "features": X.columns,
        "episodios_entrenamiento": sorted(np.unique(grupos[tr]).tolist()),
        "episodios_test": sorted(np.unique(grupos[te]).tolist()),
        "validacion": {"esquema": f"StratifiedGroupKFold {N_FOLDS} folds x {REPETICIONES} repeticiones "
                                  "sobre los episodios de entrenamiento",
                       "metricas": validacion},
        "test": test,
        "versiones": {"scikit-learn": sklearn.__version__, "polars": pl.__version__},
    })
    (salida / "metricas.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    for k, d in reporte["validacion"]["metricas"].items():
        log.info("validacion %-18s %.4f  [%.4f, %.4f]", k, d["media"], d["p2_5"], d["p97_5"])
    for k, v in reporte["test"].items():
        log.info("test       %-22s %s", k, v)
    return reporte


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entrenar()
