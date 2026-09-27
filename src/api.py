"""API del detector: lecturas de una GPU -> estado.

El modelo se carga una vez al arrancar. `/predecir` no calcula features por su
cuenta: el pipeline serializado aplica el mismo contrato y el mismo `features`
del entrenamiento.

Uso: uvicorn api:app --app-dir src
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from schema import Peticion

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "modelo.joblib"


class Prediccion(BaseModel):
    estado_predicho: str
    confianza: float  # probabilidad que el modelo le da a la clase elegida
    aviso: str | None = None  # que rompio el contrato o que lecturas no se usaron; None si nada


class Salud(BaseModel):
    estado: str
    clases: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = joblib.load(MODELO)
    yield


app = FastAPI(title="Detector de fallas en GPU", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def ventana_invalida(request: Request, exc: RequestValidationError):
    # 422 con donde y por que fallo, sin devolver las lecturas que envio el cliente
    errores = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errores})


@app.get("/health")
def health() -> Salud:
    """Confirma que la API responde y que el modelo esta cargado."""
    return Salud(estado="ok", clases=app.state.modelo.classes_.tolist())


@app.post("/predecir")
def predecir(peticion: Peticion) -> Prediccion:
    """Diagnostica la GPU a partir de 30 o mas lecturas consecutivas, de un segundo cada una."""
    # una peticion sin ninguna ventana valida no llega aqui: Peticion la rechaza con 422.
    # Con varias ventanas se promedian sus probabilidades, como en la evaluacion por episodio.
    modelo = app.state.modelo
    p = np.mean([modelo.predict_proba(v)[0] for v in peticion.ventanas], axis=0)
    i = p.argmax()
    return Prediccion(estado_predicho=modelo.classes_[i], confianza=round(float(p[i]), 4), aviso=peticion.aviso)
