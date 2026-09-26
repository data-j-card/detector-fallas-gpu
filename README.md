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
│   └── train.py                 # separa train/test, valida, evalúa y serializa
├── tests/                       # pytest: contrato, features y pipeline serializado
└── requirements.txt
```

## Instalación

Requiere Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # incluye `-e .`: instala schema, features y train
```

La instalación hace importables los módulos de `src/` desde cualquier carpeta. El `.joblib`
guarda los pasos ajustados, pero no el código de `validar_ventana` y `features`: los
referencia por nombre y los importa al cargarse.

## Uso

```bash
python -m pytest              # 27 tests
python src/train.py           # reentrena y reescribe models/
```

Diagnosticar una ventana con el modelo serializado:

```python
import joblib

from schema import Ventana, leer_csv

modelo = joblib.load("models/modelo.joblib")
lecturas = leer_csv("data/telemetria_publica.csv").head(30).drop("episodio_id", "segundo", "estado").to_dicts()
ventana = Ventana(lecturas=lecturas)  # aplica el contrato: una lectura fuera de rango lanza ValidationError
print(modelo.predict(ventana.df))                                   # ['normal']
print(dict(zip(modelo.classes_.tolist(), modelo.predict_proba(ventana.df)[0].round(3).tolist())))
```

Cada lectura es un diccionario con `temp_c`, `power_w`, `util_pct`, `clock_mhz` y `ecc_errors`,
en orden cronológico.

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

Se calculan por ventana de 30 s, cortada dentro de cada episodio, y solo con las filas de
esa ventana. No hay estadísticos globales que filtren información entre train y test.

| Grupo | Features | Falla que delata |
|---|---|---|
| nivel y dispersión de cada señal | media, std, min, max | sobrecalentamiento (temperatura y potencia altas, reloj bajo) |
| dinámica | pendiente, cambio absoluto medio entre segundos | inestabilidad |
| inestabilidad relativa | CV de potencia y de reloj | falla de alimentación |
| memoria | total, fracción de segundos con error, máximo y pendiente de ECC | degradación de memoria |
| umbrales físicos | fracción de tiempo ≥ 83 °C y ≥ 90 % de utilización | throttling térmico |
| desacoples | temp/W, W/util, reloj/util, correlaciones temp–reloj, potencia–reloj, util–potencia, potencia–temp | señales que dejan de moverse juntas |

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
