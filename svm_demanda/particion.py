"""División temporal y creación de pliegues de validación cruzada."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def dividir_y_pliegues(
    tabla: pd.DataFrame,
    columnas: list[str],
    fraccion_prueba: float = 0.2,
    n_pliegues: int = 3,
) -> tuple:
    """Divide la tabla en train/test por fecha y crea los pliegues temporales.

    Devuelve:
        X_train, y_train, X_test, y_test, pliegues, fecha_prueba
    """
    # 1. Split temporal (80% train, 20% test) basado en FECHAS, no en filas
    fechas_objetivo = np.sort(tabla["fecha"].unique())
    corte = int(len(fechas_objetivo) * (1 - fraccion_prueba))
    fecha_prueba = pd.Timestamp(fechas_objetivo[corte])

    entrenamiento = tabla.loc[tabla["fecha"] < fecha_prueba].reset_index(drop=True)
    prueba = tabla.loc[tabla["fecha"] >= fecha_prueba].reset_index(drop=True)

    X_train, y_train = entrenamiento[columnas], entrenamiento["demanda"]
    X_test, y_test = prueba[columnas], prueba["demanda"]

    # 2. Pliegues de validación cruzada temporal DENTRO del entrenamiento
    fechas_train = np.sort(entrenamiento["fecha"].unique())
    pliegues = []

    for it, iv in TimeSeriesSplit(n_splits=n_pliegues).split(fechas_train):
        fechas_t, fechas_v = fechas_train[it], fechas_train[iv]

        # Mapear las fechas de vuelta a los índices de las filas (hay 6 filas por día)
        filas_t = np.flatnonzero(entrenamiento["fecha"].isin(fechas_t).to_numpy())
        filas_v = np.flatnonzero(entrenamiento["fecha"].isin(fechas_v).to_numpy())

        # 3. Red de seguridad: evitar fuga de datos (data leakage)
        assert entrenamiento.loc[filas_t, "fecha"].max() < entrenamiento.loc[filas_v, "fecha"].min(), \
            "Fuga de datos: el entrenamiento se solapa con la validación."

        pliegues.append((filas_t, filas_v))

    assert entrenamiento["fecha"].max() < prueba["fecha"].min(), \
        "Fuga de datos: el entrenamiento se solapa con la prueba."

    return X_train, y_train, X_test, y_test, pliegues, fecha_prueba