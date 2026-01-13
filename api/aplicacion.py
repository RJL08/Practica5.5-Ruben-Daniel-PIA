from fastapi import FastAPI, HTTPException

from api.esquemas import DatosVivienda, RespuestaPrediccion
from src.config import Configuracion
from src.predictor_viviendas import PredictorViviendas

config = Configuracion()

app = FastAPI(
    title="Predicción de Precios de Viviendas (California)",
    version="1.0.0",
    description="API REST para la Práctica 5.2",
)

predictor: PredictorViviendas | None = None


@app.on_event("startup")
def cargar_modelos():
    global predictor
    predictor = PredictorViviendas(config.carpeta_modelos)


@app.get("/")
def inicio():
    return {"mensaje": "API para predecir precios", "docs": "/docs"}


@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.get("/info")
def informacion():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados todavía")

    return {
        "modelos_disponibles": ["Regresión Lineal", "Regresión Polinomial", "Red Neuronal"],
        "total_caracteristicas": len(predictor.preprocesador.nombres_caracteristicas),
        "nombres_caracteristicas": predictor.preprocesador.nombres_caracteristicas,
    }


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(datos: DatosVivienda):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados todavía")

    try:
        predicciones = predictor.predecir(datos.model_dump())
        return {"estado": "exito", **predicciones}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
