from __future__ import annotations # usado para permitir anotaciones de tipo con clases que aún no están definidas

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow import keras
from tensorflow.keras import layers

# clase modelos entrenados que contiene los modelos de regresión lineal, polinomial y red neuronal
# se utiliza dataclass para definir la clase de forma sencilla y legible
# se utiliza frozen=True para que la clase sea inmutable
# se utiliza Path para manejar las rutas de los archivos y carpetas de forma portable
from src.config import Configuracion
@dataclass
class ModelosEntrenados:
    modelo_regresion_lineal: LinearRegression
    modelo_polinomial: LinearRegression
    transformador_polinomial: PolynomialFeatures
    modelo_red_neuronal: keras.Model

# clase entrenador de modelos que contiene los métodos para entrenar los modelos de regresión lineal, polinomial y red neuronal
# se utiliza dataclass para definir la clase de forma sencilla y legible
class EntrenadorModelos:
    def entrenar_regresion_lineal(self, X_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray) -> LinearRegression:
        modelo = LinearRegression()
        modelo.fit(X_entrenamiento, y_entrenamiento)
        return modelo

    def entrenar_regresion_polinomial(self, X_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray, grado: int = 2) -> tuple[LinearRegression, PolynomialFeatures]:
        transformador = PolynomialFeatures(degree=grado, include_bias=False)
        X_polinomial = transformador.fit_transform(X_entrenamiento)

        modelo = LinearRegression()
        modelo.fit(X_polinomial, y_entrenamiento)
        return modelo, transformador
    # método para entrenar una red neuronal con Keras
    # se utiliza keras para definir la arquitectura de la red neuronal y entrenarla
    # se utiliza layers para definir las capas de la red neuronal
    # se utiliza keras.utils.set_random_seed para fijar la semilla aleatoria para reproducibilidad
    # se utiliza keras.Sequential para definir la arquitectura de la red neuronal de forma secuencial
    # se utiliza layers.Input para definir la capa de entrada de la red neuronal
    # se utilizan layers.Dense para definir las capas densas de la red neuronal
    # se utiliza layers.Dropout para añadir capas de dropout para evitar el sobreajuste
    # se utiliza modelo.compile para compilar el modelo con el optimizador, la función de pérdida y las métricas
    # se utiliza modelo.fit para entrenar el modelo con los datos
    # se utilizan los parámetros epocas, tamano_lote y semilla para controlar el entrenamiento
    # se devuelve el modelo entrenado
    # se utiliza keras.Model para definir el tipo de retorno del método
    # se utiliza np.ndarray para definir el tipo de los parámetros de entrada
    # se utiliza int para definir el tipo de los parámetros epocas y tamano_lote
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
        #
        modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss="mse",metrics=["mae"],)

        modelo.fit(X_entrenamiento,y_entrenamiento,validation_data=(X_validacion, y_validacion),epochs=epocas,batch_size=tamano_lote,verbose=0,)
        return modelo

    @staticmethod
    def guardar(modelos: ModelosEntrenados, carpeta_modelos: Path) -> None:
        carpeta_modelos.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelos.modelo_regresion_lineal, carpeta_modelos / "modelo_regresion_lineal.pkl")
        joblib.dump(modelos.modelo_polinomial, carpeta_modelos / "modelo_polinomial.pkl")
        joblib.dump(modelos.transformador_polinomial, carpeta_modelos / "transformador_polinomial.pkl")
        modelos.modelo_red_neuronal.save(carpeta_modelos / "modelo_red_neuronal.keras")
