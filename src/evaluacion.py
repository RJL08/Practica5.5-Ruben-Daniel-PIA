from typing import Dict
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# EvaluadorModelos es una clase que contiene métodos para evaluar modelos de regresión
# utilizando métricas como R2, RMSE y MAE.
class EvaluadorModelos:
    @staticmethod
    def metricas_regresion(y_real: np.ndarray, y_predicho: np.ndarray) -> Dict[str, float]:
        r2 = float(r2_score(y_real, y_predicho))
        rmse = float(np.sqrt(mean_squared_error(y_real, y_predicho)))
        mae = float(mean_absolute_error(y_real, y_predicho))
        return {"r2": r2, "rmse": rmse, "mae": mae}

    def evaluar_regresion_lineal(self, modelo, X_prueba: np.ndarray, y_prueba: np.ndarray) -> Dict[str, float]:
        y_pred = modelo.predict(X_prueba)
        return self.metricas_regresion(y_prueba, y_pred)

    def evaluar_regresion_polinomial(
        self, modelo, transformador_polinomial, X_prueba: np.ndarray, y_prueba: np.ndarray) -> Dict[str, float]:
        X_poly = transformador_polinomial.transform(X_prueba)
        y_pred = modelo.predict(X_poly)
        return self.metricas_regresion(y_prueba, y_pred)

    def evaluar_red_neuronal(self, modelo, X_prueba: np.ndarray, y_prueba: np.ndarray) -> Dict[str, float]:
        y_pred = modelo.predict(X_prueba, verbose=0).reshape(-1)
        return self.metricas_regresion(y_prueba, y_pred)
