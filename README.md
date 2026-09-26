# Detector de fallas en GPU

Clasifica el estado de una GPU a partir de 30 segundos de telemetría (temperatura, potencia,
utilización, reloj y errores ECC) en cuatro clases: `normal`, `sobrecalentamiento`,
`degradacion_memoria` y `falla_alimentacion`.

```
ventana de 30 lecturas -> contrato (pandera) -> 39 features (polars) -> StandardScaler -> LogisticRegression -> estado
```

## Estructura

```
detector-fallas-gpu/
├── data/
│   └── telemetria_publica.csv   # 48 episodios x 300 s, 12 por clase
├── models/
│   ├── modelo.joblib            # pipeline completo serializado
│   └── metricas.json            # episodios de train/test, validación, test, versiones
├── src/
│   ├── schema.py                # contrato de datos (pandera) y entrada de la API (pydantic)
│   ├── features.py              # ventaneo por episodio y cálculo de features
│   ├── train.py                 # separa train/test, valida, evalúa y serializa
│   └── api.py                   # FastAPI: /health y /predecir
├── ejemplos/                    # ventanas de episodios de test para probar la API
├── tests/                       # pytest: contrato, features, pipeline serializado y API
└── requirements.txt
```

## Instalación

Requiere Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # incluye `-e .`: instala schema, features, train y api
```

La instalación hace importables los módulos de `src/` desde cualquier carpeta. El `.joblib`
guarda los pasos ajustados, pero no el código de `validar_ventana` y `features`: los
referencia por nombre y los importa al cargarse.

## Uso

```bash
python -m pytest                                  # 42 tests
python src/train.py                               # reentrena y reescribe models/
uvicorn api:app --host 127.0.0.1 --port 8000      # levanta la API, desde la raíz del repo
```

## API

| Endpoint | Entrada | Salida |
|---|---|---|
| `GET /health` | — | `{"estado": "ok", "clases": [...]}` si el modelo cargó |
| `POST /predecir` | `{"lecturas": [...]}` con 30 lecturas en orden | `{"estado_predicho": "...", "confianza": 0.998}` |

`confianza` es la probabilidad que el modelo asigna a la clase elegida. La documentación
interactiva queda en <http://127.0.0.1:8000/docs>, con ventanas de episodios de test (`ejemplos/`)
listas para ejecutar.

```bash
curl -X POST http://127.0.0.1:8000/predecir -H "Content-Type: application/json" -d @ejemplos/2_sobrecalentamiento.json
```

La API no calcula features propias: `modelo.predict_proba` recorre el pipeline serializado,
que usa las mismas funciones `validar_ventana` y `features` del entrenamiento.

Antes de llegar al modelo, `Lectura` y `Ventana` (`src/schema.py`) validan la petición. Cada
lectura debe traer los cinco campos, sin campos extra y con su tipo exacto (`strict`: `"88.3"`,
`true` o `1.0` no se convierten en números). La ventana debe tener exactamente 30 lecturas, el
tamaño con que se entrenó: con menos, las features cambian de escala (`ecc_total` es una suma) y
la confianza cae. Si algo falla, la respuesta es 422 con la posición y el motivo:

```json
{"detail":[{"loc":["body"],"msg":"Value error, lecturas fuera del rango fisico en las posiciones [10]"}]}
```

## Datos

| Columna | Tipo | Rango observado | Significado |
|---|---|---|---|
| `episodio_id` | int | 0–47 | tramo de 300 s de una GPU; no entra al modelo |
| `segundo` | int | 0–299 | posición dentro del episodio; no entra al modelo |
| `temp_c` | float | 22.1–99.2 | temperatura en °C |
| `power_w` | float | −18.1–335.2 | consumo en W |
| `util_pct` | float | 0–100 | utilización en % |
| `clock_mhz` | float | 1022–2546 | frecuencia del reloj; MHz enteros |
| `ecc_errors` | int | 0–14 | errores ECC en ese segundo (conteo, no acumulado) |
| `estado` | str | 4 clases | etiqueta; constante dentro de cada episodio |

La señal es estacionaria dentro de cada episodio: la falla cubre los 300 s, así que cada
ventana hereda la etiqueta de su episodio.

## Contrato de datos

`src/schema.py` valida la telemetría antes de calcular cualquier feature:

| Regla | Si no se cumple |
|---|---|
| columnas presentes y con su tipo | se rechaza el lote completo |
| temperatura 0–120 °C, utilización 0–100 %, reloj entero > 0, ECC ≥ 0, estado válido, sin nulos | se descarta la fila y, con ella, su ventana |
| par (`episodio_id`, `segundo`) único | se descartan ambas copias y su ventana |
| potencia ≥ −50 W | se descarta la fila; entre −50 y 0 W se recorta a 0 |

Las tres lecturas negativas del dataset (−13.5, −3.9 y −18.1 W) están en `falla_alimentacion`,
entre caídas reales de potencia: son el sensor pasándose de cero en un colapso, no basura.
Recortarlas a 0 conserva el síntoma. Descartar la ventana o imputar la mediana lo borraba
(la mediana bajaba la desviación estándar de la potencia hasta 17 %).

Para predecir no se descarta nada: `validar_ventana` exige exactamente 30 lecturas y rechaza
la ventana entera si una sola viola el contrato. Es el primer paso del pipeline serializado,
así que `modelo.predict` rechaza datos imposibles aunque no pasen por `Ventana`, el modelo
de entrada de la API, que usa la misma función.

## Features

Se calculan por ventana de 30 s, cortada dentro de cada episodio, y solo con las filas de esa
ventana: no hay estadísticos globales que filtren información entre train y test. Antes de
calcularlas, la potencia negativa se recorta a 0. Son 39.

Las cifras de las tablas son la mediana por clase sobre las 480 ventanas del dataset:
**N** normal, **S** sobrecalentamiento, **M** degradación de memoria, **A** falla de alimentación.

### Estadísticos de cada señal (24)

Para `temp_c`, `power_w`, `util_pct` y `clock_mhz` se calculan seis estadísticos:

| Sufijo | Cálculo | Qué captura |
|---|---|---|
| `_mean` | media de la ventana | nivel de operación |
| `_std` | desviación estándar | cuánto oscila la señal |
| `_min`, `_max` | extremos | caídas y picos que la media suaviza |
| `_pendiente` | pendiente de la recta ajustada contra el segundo (cov(t, x) / var(t)) | si la señal sube o baja durante la ventana |
| `_dif_abs` | media de \|x<sub>t</sub> − x<sub>t−1</sub>\| | saltos de un segundo al siguiente |

| Feature | N | S | M | A |
|---|---|---|---|---|
| `temp_c_mean` | 54.9 | **88.0** | 60.0 | 50.2 |
| `power_w_mean` | 199.8 | **290.3** | 208.5 | 140.5 |
| `power_w_std` | 19.9 | 10.0 | 24.7 | **44.8** |
| `power_w_min` | 160.2 | 270.6 | 157.1 | **48.6** |
| `util_pct_mean` | 69.9 | **95.0** | 67.8 | 54.3 |
| `util_pct_std` | 11.9 | 3.6 | 13.7 | **24.2** |
| `clock_mhz_mean` | 2400 | **2100** | 2380 | **1798** |
| `clock_mhz_std` | 29 | 79 | 38 | **203** |
| `clock_mhz_dif_abs` | 33 | 90 | 45 | **228** |

El sobrecalentamiento se ve en el nivel: la GPU trabaja al máximo, caliente y con el reloj
recortado. La falla de alimentación se ve en la dispersión: potencia, utilización y reloj
oscilan entre 2 y 7 veces más que en `normal`, y en una ventana típica la potencia llega a
bajar a 49 W.

Las cuatro `_pendiente` valen prácticamente 0 en todas las clases (entre −0.09 y 0.21 por
segundo): la falla cubre el episodio entero y la señal no deriva dentro de la ventana. Se
calculan igual, pero en este dataset no separan clases.

### Inestabilidad relativa (2)

| Feature | Cálculo | N | S | M | A |
|---|---|---|---|---|---|
| `power_w_cv` | std / media de la potencia | 0.10 | 0.04 | 0.12 | **0.31** |
| `clock_mhz_cv` | std / media del reloj | 0.01 | 0.04 | 0.02 | **0.11** |

El coeficiente de variación mide la oscilación en proporción al nivel, así que no depende de
que la GPU consuma 150 o 300 W. Es la señal más directa de una alimentación inestable.

### Memoria (4)

| Feature | Cálculo | N | S | M | A |
|---|---|---|---|---|---|
| `ecc_total` | suma de errores ECC en la ventana | 0.5 | 3 | **149.5** | 6 |
| `ecc_frac_con_error` | fracción de segundos con al menos un error | 0.02 | 0.10 | **1.00** | 0.17 |
| `ecc_max` | máximo de errores en un segundo | 0.5 | 1 | **10** | 1 |
| `ecc_pendiente` | pendiente de los errores contra el segundo | 0 | 0 | 0.01 | 0 |

Las otras clases tienen errores ECC aislados; la degradación de memoria los tiene todos los
segundos, unos 5 por segundo. `ecc_frac_con_error` separa las dos situaciones aunque el total
fuera parecido: pocos errores concentrados no son lo mismo que errores constantes.

### Umbrales físicos (2)

| Feature | Cálculo | N | S | M | A |
|---|---|---|---|---|---|
| `temp_frac_sobre_umbral` | fracción de segundos con temperatura ≥ 83 °C | 0 | **0.97** | 0 | 0 |
| `util_frac_alta` | fracción de segundos con utilización ≥ 90 % | 0.03 | **0.90** | 0.03 | 0.07 |

83 °C es la zona en que las GPU NVIDIA empiezan a bajar el reloj para protegerse (throttling).
Contar cuánto tiempo pasa la GPU sobre ese umbral es más directo que la media, que un solo pico
no mueve.

### Relaciones entre señales (7)

| Feature | Cálculo | N | S | M | A |
|---|---|---|---|---|---|
| `clock_por_util` | reloj medio / utilización media | 34.4 | **22.1** | 35.1 | 33.0 |
| `temp_por_watt` | temperatura media / potencia media | 0.27 | 0.30 | 0.29 | 0.36 |
| `watt_por_util` | potencia media / utilización media | 2.88 | 3.06 | 3.07 | 2.61 |
| `corr_temp_clock` | correlación temperatura–reloj | 0.00 | 0.00 | 0.04 | 0.01 |
| `corr_power_clock` | correlación potencia–reloj | −0.01 | −0.02 | −0.04 | 0.00 |
| `corr_util_power` | correlación utilización–potencia | 0.02 | −0.01 | 0.01 | 0.03 |
| `corr_power_temp` | correlación potencia–temperatura | 0.00 | −0.02 | −0.01 | −0.03 |

Buscan señales que dejan de moverse juntas. `clock_por_util` es la que funciona: con
sobrecalentamiento la GPU entrega un tercio menos de reloj por cada punto de utilización, que
es justo el throttling. Las cuatro correlaciones valen prácticamente 0 en todas las clases: en
este dataset las señales varían de forma independiente segundo a segundo, así que no aportan.
Un cociente o una correlación sin variación (divisor 0 o serie constante) se fija en 0.

## Separación, validación y test

```
48 episodios ──┬── 36 de entrenamiento (9 por clase) ── StratifiedGroupKFold 3 folds x 10 repeticiones
               └── 12 de test (3 por clase) ─────────── se evalúan una sola vez, al final
```

La separación es por episodio: las ventanas de un episodio de test nunca aparecen en el
entrenamiento. Los ids de ambos grupos quedan en `models/metricas.json`. El modelo
serializado es el ajustado sobre los 36 episodios, el mismo que se evaluó en test.

Dentro de los 36, la validación cruzada agrupa por `episodio_id`. Con `StratifiedKFold`,
las ventanas de un mismo episodio caen en ambos lados; con etiquetas aleatorias por
episodio ese esquema da 36 % de accuracy cuando el azar es 25 %: el modelo reconoce el
episodio, no la falla.

| Métrica | Validación: media [p2.5–p97.5 de 30 folds] | Test (12 episodios no vistos) |
|---|---|---|
| accuracy por ventana | 1.000 [1.000–1.000] | 1.000 |
| F1 macro por ventana | 1.000 [1.000–1.000] | 1.000 |
| log loss por ventana | 0.0045 [0.0042–0.0048] | 0.0031 |
| accuracy por episodio | 1.000 [1.000–1.000] | 1.000 (12/12) |
| probabilidad del diagnóstico menos seguro | 0.992 [0.991–0.993] | 0.994 |

### ¿Con qué confianza es 100 %?

Acertar 12 de 12 episodios no prueba que el modelo nunca falle. El intervalo de confianza
exacto al 95 % (Clopper-Pearson) para la accuracy por episodio en test es
**[73.5 %, 100 %]**: con 12 casos, un modelo que falla 1 de cada 4 episodios todavía podría
acertarlos todos por azar.

El intervalo se calcula sobre episodios y no sobre las 120 ventanas de test. Las ventanas de
un mismo episodio no son independientes, y contarlas daría [97.0 %, 100 %], una precisión
que los datos no respaldan. Para que el límite inferior llegue a 95 % sin errores harían
falta 72 episodios de test.

La otra lectura de la confianza es la del propio modelo: en el diagnóstico de test menos
seguro, la clase elegida recibió probabilidad 0.994.

Las clases de este dataset se separan por nivel: 88 °C contra 55 °C, reloj en 1800 contra
2400 MHz, unos 5 errores ECC por segundo contra 0. En validación, cualquier ventana desde 5 s
clasifica sin error, y el tamaño de 30 s se eligió porque es el mínimo con puntos suficientes para
correlaciones y pendientes. La muestra efectiva son 48 episodios, no 480 ventanas, y datos
con más ruido podrían bajar estas cifras.

El tamaño de ventana y el recorte de potencia se decidieron en el análisis exploratorio,
que miró los 48 episodios. El test protege el ajuste del modelo, no esas decisiones de diseño.
