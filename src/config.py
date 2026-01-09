from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Configuracion:
    raiz_proyecto: Path = Path(__file__).resolve().parents[1]

    ruta_datos: Path = raiz_proyecto / "data" / "housing.csv"
    carpeta_modelos: Path = raiz_proyecto / "models"

    ruta_escalador: Path = carpeta_modelos / "escalador.pkl"
    ruta_nombres_caracteristicas: Path = carpeta_modelos / "nombres_caracteristicas.pkl"

    ruta_modelo_rl: Path = carpeta_modelos / "modelo_regresion_lineal.pkl"
    ruta_modelo_polinomial: Path = carpeta_modelos / "modelo_polinomial.pkl"
    ruta_transformador_polinomial: Path = carpeta_modelos / "transformador_polinomial.pkl"
    ruta_modelo_red_neuronal: Path = carpeta_modelos / "modelo_red_neuronal.keras"
