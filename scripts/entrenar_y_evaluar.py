from sklearn.model_selection import train_test_split

from src.config import Configuracion
from src.cargar_datos import CargadorCSVViviendas
from src.preprocesamiento import PreprocesadorViviendas
from src.entrenamiento import EntrenadorModelos, ModelosEntrenados
from src.evaluacion import EvaluadorModelos


def main():
    config = Configuracion()
    config.carpeta_modelos.mkdir(parents=True, exist_ok=True)

    df = CargadorCSVViviendas(config.ruta_datos).cargar()

    pre = PreprocesadorViviendas()
    X, y = pre.entrenar_y_transformar(df)

    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=42)

    pre.guardar(config.carpeta_modelos)

    entrenador = EntrenadorModelos()
    modelo_rl = entrenador.entrenar_regresion_lineal(X_ent, y_ent)
    modelo_poly, transf_poly = entrenador.entrenar_regresion_polinomial(X_ent, y_ent, grado=2)
    modelo_rn = entrenador.entrenar_red_neuronal(X_ent, y_ent, X_pru, y_pru, epocas=50)

    entrenador.guardar(
        ModelosEntrenados(
            modelo_regresion_lineal=modelo_rl,
            modelo_polinomial=modelo_poly,
            transformador_polinomial=transf_poly,
            modelo_red_neuronal=modelo_rn,
        ),
        config.carpeta_modelos,
    )

    evaluador = EvaluadorModelos()
    print("\nMÉTRICAS (PRUEBA)")
    print("Regresión Lineal:", evaluador.evaluar_regresion_lineal(modelo_rl, X_pru, y_pru))
    print("Polinomial:", evaluador.evaluar_regresion_polinomial(modelo_poly, transf_poly, X_pru, y_pru))
    print("Red Neuronal:", evaluador.evaluar_red_neuronal(modelo_rn, X_pru, y_pru))


if __name__ == "__main__":
    main()
