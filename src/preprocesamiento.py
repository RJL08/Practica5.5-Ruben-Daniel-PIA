from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreprocesadorViviendas:
    COLUMNA_OBJETIVO = "median_house_value"
    COLUMNA_CATEGORICA = "ocean_proximity"

    COLUMNAS_OCEANO = [
        "ocean_proximity_<1H OCEAN",
        "ocean_proximity_INLAND",
        "ocean_proximity_ISLAND",
        "ocean_proximity_NEAR BAY",
        "ocean_proximity_NEAR OCEAN",
    ]

    def __init__(self):
        self.escalador = StandardScaler()
        self.nombres_caracteristicas: Optional[List[str]] = None
        self._entrenado = False

    @staticmethod
    def _limpieza_basica(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna().copy()

    @staticmethod
    def _crear_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["habitaciones_por_hogar"] = df["total_rooms"] / df["households"]
        df["dormitorios_por_habitacion"] = df["total_bedrooms"] / df["total_rooms"]
        df["poblacion_por_hogar"] = df["population"] / df["households"]
        return df

    def _one_hot_oceano(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.COLUMNA_CATEGORICA not in df.columns:
            raise ValueError(f"Falta columna '{self.COLUMNA_CATEGORICA}' en el CSV.")

        df = pd.get_dummies(df, columns=[self.COLUMNA_CATEGORICA], prefix=self.COLUMNA_CATEGORICA)

        for col in self.COLUMNAS_OCEANO:
            if col not in df.columns:
                df[col] = 0

        return df

    def entrenar(self, df: pd.DataFrame) -> "PreprocesadorViviendas":
        df = self._limpieza_basica(df)
        df = self._crear_caracteristicas(df)
        df = self._one_hot_oceano(df)

        if self.COLUMNA_OBJETIVO not in df.columns:
            raise ValueError(f"Falta la columna objetivo '{self.COLUMNA_OBJETIVO}'.")

        X = df.drop(columns=[self.COLUMNA_OBJETIVO])
        self.nombres_caracteristicas = X.columns.tolist()

        self.escalador.fit(X.values)
        self._entrenado = True
        return self

    def transformar(self, df: pd.DataFrame, incluir_objetivo: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._entrenado:
            raise RuntimeError("Preprocesador no entrenado. Llama a entrenar() primero.")

        df = self._limpieza_basica(df)
        df = self._crear_caracteristicas(df)
        df = self._one_hot_oceano(df)

        y = None
        if incluir_objetivo:
            if self.COLUMNA_OBJETIVO not in df.columns:
                raise ValueError(f"Falta la columna objetivo '{self.COLUMNA_OBJETIVO}'.")
            y = df[self.COLUMNA_OBJETIVO].values

        X = df.drop(columns=[self.COLUMNA_OBJETIVO]) if self.COLUMNA_OBJETIVO in df.columns else df

        for col in self.nombres_caracteristicas:
            if col not in X.columns:
                X[col] = 0
        X = X[self.nombres_caracteristicas]

        X_escalado = self.escalador.transform(X.values)
        return X_escalado, y

    def entrenar_y_transformar(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.entrenar(df)
        X_escalado, y = self.transformar(df, incluir_objetivo=True)
        return X_escalado, y

    def guardar(self, carpeta_modelos: Path) -> None:
        carpeta_modelos.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.escalador, carpeta_modelos / "escalador.pkl")
        joblib.dump(self.nombres_caracteristicas, carpeta_modelos / "nombres_caracteristicas.pkl")

    @classmethod
    def cargar(cls, carpeta_modelos: Path) -> "PreprocesadorViviendas":
        obj = cls()
        obj.escalador = joblib.load(carpeta_modelos / "escalador.pkl")
        obj.nombres_caracteristicas = joblib.load(carpeta_modelos / "nombres_caracteristicas.pkl")
        obj._entrenado = True
        return obj
