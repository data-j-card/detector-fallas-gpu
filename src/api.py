"""API del detector: ventana de 30 lecturas -> estado de la GPU.

El modelo se carga una vez al arrancar. `/predecir` no calcula features por su
cuenta: el pipeline serializado aplica el mismo contrato y el mismo `features`
del entrenamiento.

Uso: uvicorn api:app
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from schema import Ventana

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "modelo.joblib"
EJEMPLOS = {r.stem: {"value": json.loads(r.read_text())} for r in sorted((RAIZ / "ejemplos").glob("*.json"))}


class Prediccion(BaseModel):
    estado_predicho: str
    confianza: float  # probabilidad que el modelo le da a la clase elegida


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
    # 422 con donde y por que fallo; sin devolver las 30 lecturas que envio el cliente
    errores = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errores})


@app.get("/health")
def health() -> Salud:
    """Confirma que la API responde y que el modelo esta cargado."""
    return Salud(estado="ok", clases=app.state.modelo.classes_.tolist())


@app.post("/predecir")
def predecir(ventana: Annotated[Ventana, Body(openapi_examples=EJEMPLOS)]) -> Prediccion:
    """Diagnostica una ventana de 30 lecturas consecutivas, de un segundo cada una."""
    # una ventana que viola el contrato no llega aqui: Ventana la rechaza con 422
    p = app.state.modelo.predict_proba(ventana.df)[0]
    i = p.argmax()
    return Prediccion(estado_predicho=app.state.modelo.classes_[i], confianza=round(float(p[i]), 4))
