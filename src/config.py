from dataclasses import dataclass
from pathlib import Path

# clase configuracion  con la que se definen las rutas de los archivos y carpetas
# que se van a utilizar en el proyecto
# se utiliza frozen=True para que la clase sea inmutable
# se utiliza Path para manejar las rutas de los archivos y carpetas de forma portable
# se utiliza __file__ para obtener la ruta del archivo actual y luego se navega hacia
# arriba en la jerarquía de carpetas para obtener la ruta del proyecto
# se utiliza resolve() para obtener la ruta absoluta y normalizada
# se utiliza parents[1] para obtener la carpeta del proyecto, ya que __file__
# apunta a este archivo dentro de la carpeta src
# se definen las rutas de los archivos y carpetas que se van a utilizar en el
# proyecto, como la ruta de los datos, la carpeta de modelos, y las rutas de
# los modelos y transformadores específicos que se van a utilizar en el proyecto
# se utiliza Path para manejar las rutas de los archivos y carpetas de forma portable
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
