from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ModelosEntrenados:
    modelo_regresion_lineal: LinearRegression
    modelo_polinomial: LinearRegression
    transformador_polinomial: PolynomialFeatures
    modelo_red_neuronal: keras.Model


class EntrenadorModelos:
    def entrenar_regresion_lineal(self, X_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray) -> LinearRegression:
        modelo = LinearRegression()
        modelo.fit(X_entrenamiento, y_entrenamiento)
        return modelo

    def entrenar_regresion_polinomial(
        self, X_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray, grado: int = 2
    ) -> tuple[LinearRegression, PolynomialFeatures]:
        transformador = PolynomialFeatures(degree=grado, include_bias=False)
        X_polinomial = transformador.fit_transform(X_entrenamiento)

        modelo = LinearRegression()
        modelo.fit(X_polinomial, y_entrenamiento)
        return modelo, transformador

    def entrenar_red_neuronal(
        self,X_entrenamiento: np.ndarray,y_entrenamiento: np.ndarray,X_validacion: np.ndarray,
        y_validacion: np.ndarray,epocas: int = 50,tamano_lote: int = 32,semilla: int = 42,
    ) -> keras.Model:
        keras.utils.set_random_seed(semilla)

        modelo = keras.Sequential([
                layers.Input(shape=(X_entrenamiento.shape[1],)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ])

        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        modelo.fit(
            X_entrenamiento,
            y_entrenamiento,
            validation_data=(X_validacion, y_validacion),
            epochs=epocas,
            batch_size=tamano_lote,
            verbose=0,
        )
        return modelo

    @staticmethod
    def guardar(modelos: ModelosEntrenados, carpeta_modelos: Path) -> None:
        carpeta_modelos.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelos.modelo_regresion_lineal, carpeta_modelos / "modelo_regresion_lineal.pkl")
        joblib.dump(modelos.modelo_polinomial, carpeta_modelos / "modelo_polinomial.pkl")
        joblib.dump(modelos.transformador_polinomial, carpeta_modelos / "transformador_polinomial.pkl")
        modelos.modelo_red_neuronal.save(carpeta_modelos / "modelo_red_neuronal.keras")
