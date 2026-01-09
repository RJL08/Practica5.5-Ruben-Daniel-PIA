# Predicción de Precios de Viviendas (California) — Práctica 5.2

Proyecto de regresión para predecir el precio de viviendas en California a partir del dataset `housing.csv` y desplegar el modelo mediante un API REST con FastAPI.

---

## 1) Requisitos

- Python (recomendado 3.10–3.12).
- Entorno virtual (venv/conda) activado.
- Dependencias instaladas (FastAPI usa modelos Pydantic para validar el body JSON y genera `/docs` automáticamente). [web:30][web:94]

---

## 2) Estructura esperada

california-house-prices/
├─ data/
│ └─ housing.csv
├─ models/
├─ src/
├─ api/
└─ scripts/


---

## 3) Preparar el dataset

1. Descargar el dataset “California Housing Prices” desde Kaggle. [web:25]
2. Copiar el archivo `housing.csv` en la ruta:
   - `data/housing.csv`

> Importante: la carpeta `data/` debe contener el CSV antes de entrenar.

---

## 4) Instalar dependencias

Desde la raíz del proyecto (con el entorno activado):

```bash
pip install -r requirements.txt (opcional)

```
Si uvicorn no está disponible como comando, instalarlo así:
```bash
pip install "uvicorn[standard]"
```

## 5) Entrenar y evaluar modelos (apartados a y b)

Ejecuta el script de entrenamiento y evaluación desde la **raíz del proyecto**:

```bash
python scripts/entrenar_y_evaluar.py
```

## 6) Probar predicción en local (sin API)
Desde la raíz del proyecto, ejecutar:

```bash
python scripts/probar_prediccion.py
```
o

```bash
python -m scripts.prueba_rapida_predictor
```
Esto cargará el modelo entrenado y realizará una predicción de prueba.

## 7) Desplegar API REST con FastAPI
Desde la raíz del proyecto, ejecutar:

```bash
python -m uvicorn api.aplicacion:app --reload --port 8000
```
Esto iniciará el servidor en `http://127.0.0.1:8000`.
o 
http://127.0.0.1:8000/docs para una visualización más amigable.
o
http://127.0.0.1:8000/redoc

## 8) Probar los endpoints de la API
Entramos en http://127.0.0.1:8000/docs y prueba:
GET / (inicio)
GET /salud
GET /info (información de modelos y características)
POST /predecir (predicción con JSON en el body)

Ejemplo de JSON para POST /predecir:
```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}

```



