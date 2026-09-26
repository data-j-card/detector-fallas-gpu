# Detector de fallas en GPU

API que recibe 30 segundos de telemetría de una GPU y predice su estado: `normal`,
`sobrecalentamiento`, `degradacion_memoria` o `falla_alimentacion`.

```
30 lecturas -> contrato de datos -> features -> StandardScaler -> LogisticRegression -> estado
```

## Estructura

```
detector-fallas-gpu/
├── src/
│   ├── features.py      # cálculo de features (usado por train y api)
│   ├── schema.py        # contrato de datos (pandera) y validación de la entrada (pydantic)
│   ├── train.py         # entrena y serializa el modelo
│   └── api.py           # la API de FastAPI
├── models/
│   └── modelo.joblib    # el modelo serializado
├── data/                # telemetría de entrenamiento
├── tests/
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Construir la imagen y correr el contenedor

```bash
docker build -t detector-gpu .
docker run -p 8000:8000 detector-gpu
```

La API queda en <http://localhost:8000/docs>.

La imagen parte de `python:3.12-slim`, instala `requirements.txt`, copia `src/` y el modelo,
expone el puerto 8000 y arranca `uvicorn`. `.dockerignore` deja fuera el entorno
virtual, git, los datos y los tests.

## Endpoints

`GET /health` confirma que la API responde y que el modelo está cargado.

`POST /predecir` recibe una ventana de 30 lecturas, una por segundo, en orden (aquí se muestran 2):

```json
{
  "lecturas": [
    {"temp_c": 88.3, "power_w": 296.5, "util_pct": 95.8, "clock_mhz": 2038, "ecc_errors": 0},
    {"temp_c": 89.8, "power_w": 298.9, "util_pct": 95.4, "clock_mhz": 2091, "ecc_errors": 1}
  ]
}
```

y devuelve el estado y la probabilidad que el modelo le da:

```json
{"estado_predicho": "sobrecalentamiento", "confianza": 0.998}
```

Para probarlo desde <http://localhost:8000/docs>: abrir `POST /predecir`, *Try it out*, pegar la
ventana en el cuerpo y *Execute*.

Si la ventana no tiene 30 lecturas, si a una lectura le falta un campo o trae un tipo
incorrecto, o si un valor es físicamente imposible (por ejemplo, 150 °C), la API responde 422
con el motivo y no llama al modelo.

## Features

La API y el entrenamiento calculan las features con la misma función, `features()` de
`src/features.py`, y el modelo serializado la incluye como paso del pipeline. Por cada ventana
se calculan 39:

| Grupo | Features |
|---|---|
| cada señal (temperatura, potencia, utilización, reloj) | media, desviación estándar, mínimo, máximo, pendiente y cambio medio entre segundos |
| inestabilidad | coeficiente de variación de la potencia y del reloj |
| memoria | total de errores ECC, fracción de segundos con error, máximo y pendiente |
| umbrales | fracción del tiempo con temperatura ≥ 83 °C y con utilización ≥ 90 % |
| relaciones | temperatura/potencia, potencia/utilización, reloj/utilización y 4 correlaciones entre señales |

## Modelo

Se entrena con 36 de los 48 episodios del dataset y se evalúa en los otros 12, que no ve
durante el entrenamiento. Acertó los 12; con tan pocos episodios, el intervalo de confianza al
95 % de la accuracy es [73.5 %, 100 %]. Las métricas quedan en `models/metricas.json`.

## Desarrollo local

Requiere Python 3.11 o superior.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                  # tests
python src/train.py               # reentrena y reescribe models/
uvicorn api:app --app-dir src     # la API sin Docker
```
