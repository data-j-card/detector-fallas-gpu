"""Entrena el clasificador de fallas y serializa el pipeline completo.

El artefacto va de la ventana cruda (30 lecturas ya validadas) a la clase:
features -> StandardScaler -> LogisticRegression.

Uso: python src/train.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import construir_dataset, features
from schema import TAMANO_VENTANA, leer_csv

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "telemetria_publica.csv"
MODELOS = RAIZ / "models"
N_FOLDS, REPETICIONES = 4, 10

log = logging.getLogger(__name__)


def clasificador():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def _resumen(valores: list[float]) -> dict:
    p2_5, p97_5 = np.percentile(valores, [2.5, 97.5])
    return {"media": round(float(np.mean(valores)), 4), "p2_5": round(float(p2_5), 4), "p97_5": round(float(p97_5), 4)}


def validacion_cruzada(X: pl.DataFrame, y: pl.Series, grupos: pl.Series) -> dict:
    """StratifiedGroupKFold repetido: ningun episodio cae en train y test a la vez
    y cada fold de test tiene 3 episodios por clase."""
    y, grupos = y.to_numpy(), grupos.to_numpy()
    folds = []
    for semilla in range(REPETICIONES):
        for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=semilla).split(X, y, grupos):
            m = clasificador().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            clases = list(m.classes_)
            # diagnostico del episodio: promedio de las probabilidades de sus ventanas
            ep = pl.DataFrame(p, schema=clases).with_columns(ep=grupos[te], y=y[te]).group_by("ep", "y").mean()
            folds.append({
                "accuracy_ventana": accuracy_score(y[te], m.classes_[p.argmax(1)]),
                "f1_macro_ventana": f1_score(y[te], m.classes_[p.argmax(1)], average="macro"),
                "log_loss_ventana": log_loss(y[te], p, labels=clases),
                "accuracy_episodio": accuracy_score(ep["y"], m.classes_[ep.select(clases).to_numpy().argmax(1)]),
            })
    return {k: _resumen([f[k] for f in folds]) for k in folds[0]}


def entrenar(datos: Path = DATOS, salida: Path = MODELOS) -> dict:
    X, y, grupos = construir_dataset(leer_csv(datos))
    metricas = validacion_cruzada(X, y, grupos)

    ajustado = clasificador().fit(X, y)
    modelo = Pipeline([("features", FunctionTransformer(features)), *ajustado.steps])

    salida.mkdir(exist_ok=True)
    joblib.dump(modelo, salida / "modelo.joblib")
    reporte = {
        "modelo": "features -> StandardScaler -> LogisticRegression",
        "tamano_ventana_s": TAMANO_VENTANA,
        "ventanas": X.height,
        "episodios": grupos.n_unique(),
        "clases": list(ajustado.classes_),
        "features": X.columns,
        "validacion": f"StratifiedGroupKFold {N_FOLDS} folds x {REPETICIONES} repeticiones, grupos = episodio_id",
        "metricas": metricas,
        "versiones": {"scikit-learn": sklearn.__version__, "polars": pl.__version__},
    }
    (salida / "metricas.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, d in metricas.items():
        log.info("%-18s %.4f  [%.4f, %.4f]", k, d["media"], d["p2_5"], d["p97_5"])
    return reporte


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entrenar()
