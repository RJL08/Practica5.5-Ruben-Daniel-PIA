from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from tensorflow import keras

from src.preprocesamiento import PreprocesadorViviendas


class PredictorViviendas:
    def __init__(self, carpeta_modelos: Path):
        self.carpeta_modelos = carpeta_modelos

        self.preprocesador = PreprocesadorViviendas.cargar(carpeta_modelos)

        self.modelo_regresion_lineal = joblib.load(carpeta_modelos / "modelo_regresion_lineal.pkl")
        self.modelo_polinomial = joblib.load(carpeta_modelos / "modelo_polinomial.pkl")
        self.transformador_polinomial = joblib.load(carpeta_modelos / "transformador_polinomial.pkl")
        self.modelo_red_neuronal = keras.models.load_model(carpeta_modelos / "modelo_red_neuronal.keras", compile=False)

    def predecir(self, datos: Dict[str, Any]) -> Dict[str, float]:
        """
        datos incluye ocean_proximity como texto (igual que el CSV).
        """
        df = pd.DataFrame([datos])
        X_escalado, _ = self.preprocesador.transformar(df, incluir_objetivo=False)

        pred_rl = float(self.modelo_regresion_lineal.predict(X_escalado)[0])

        X_poly = self.transformador_polinomial.transform(X_escalado)
        pred_poly = float(self.modelo_polinomial.predict(X_poly)[0])

        pred_rn = float(self.modelo_red_neuronal.predict(X_escalado, verbose=0)[0][0])

        return {
            "precio_predecido_regresion_lineal": pred_rl,
            "precio_predecido_polinomial": pred_poly,
            "precio_predecido_red_neuronal": pred_rn,
            "precio_promedio": float((pred_rl + pred_poly + pred_rn) / 3.0),
        }
