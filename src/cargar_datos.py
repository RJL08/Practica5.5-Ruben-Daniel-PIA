from pathlib import Path
import pandas as pd

# clase CargadorCSVViviendas con la que se cargan los datos de del csv de viviendas
# Se le pasa la ruta del csv y devuelve un dataframe con los datos
class CargadorCSVViviendas:
    def __init__(self, ruta_csv: Path):
        self.ruta_csv = ruta_csv

    def cargar(self) -> pd.DataFrame:
        if not self.ruta_csv.exists():
            raise FileNotFoundError(
                f"No existe el CSV: {self.ruta_csv}\n"

            )
        return pd.read_csv(self.ruta_csv)
