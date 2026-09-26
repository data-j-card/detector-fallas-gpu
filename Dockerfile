FROM python:3.12-slim

WORKDIR /app

# dependencias; requirements.txt trae `-e .`, que instala los modulos de src/
COPY requirements.txt pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -r requirements.txt

# modelo entrenado y ventanas de ejemplo que muestra /docs
COPY models/modelo.joblib models/
COPY ejemplos/ ejemplos/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
