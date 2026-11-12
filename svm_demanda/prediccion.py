"""Predicciones operativas y verificación de modelos cargados desde MLflow."""
import mlflow.sklearn
import numpy as np
import pandas as pd

from svm_demanda import config


def predecir_demanda(producto: str, fecha, modelo, diario: pd.DataFrame) -> float:
    """Predice la demanda de `producto` en `fecha` usando los `VENTANA` días previos.

    Exige ventas reales de todos los días de la ventana; si falta alguno,
    lanza un error en lugar de predecir con datos incompletos.
    """
    fecha = pd.Timestamp(fecha).normalize()
    categorias = sorted(diario["producto"].unique())
    if producto not in categorias:
        raise ValueError(f"Categoría desconocida. Usa una de {categorias}")

    # Los VENTANA días exactos anteriores a la fecha objetivo
    dias_previos = pd.date_range(
        fecha - pd.Timedelta(days=config.VENTANA),
        fecha - pd.Timedelta(days=1),
        freq="D",
    )
    historia = (
        diario.loc[diario["producto"] == producto]
        .set_index("fecha")
        .reindex(dias_previos)
    )
    if historia[config.VARIABLES_HISTORIA].isna().any().any():
        raise ValueError(
            f"Faltan ventas reales en los {config.VENTANA} días previos."
        )

    # Los mismos valores que el modelo vio en entrenamiento, con los mismos nombres
    valores = {}
    for retraso in range(config.VENTANA, 0, -1):
        fila = historia.loc[fecha - pd.Timedelta(days=retraso)]
        for variable in config.VARIABLES_HISTORIA:
            valores[f"{variable}_lag{retraso}"] = float(fila[variable])

    # El mismo orden de columnas que generó features.construir_lags
    columnas = [
        f"{variable}_lag{retraso}"
        for retraso in range(config.VENTANA, 0, -1)
        for variable in config.VARIABLES_HISTORIA
    ]
    entrada = pd.DataFrame([valores], columns=columnas)
    return float(modelo.predict(entrada)[0])


def cargar_modelo_mlflow(model_uri: str):
    """Recupera un pipeline guardado en MLflow a partir de su URI."""
    return mlflow.sklearn.load_model(model_uri)


def verificar_carga(model_uri: str, modelo_original, X_test: pd.DataFrame) -> None:
    """Comprueba que el modelo recargado de MLflow predice igual que el original."""
    modelo_recargado = cargar_modelo_mlflow(model_uri)
    np.testing.assert_allclose(
        modelo_recargado.predict(X_test.head()),
        modelo_original.predict(X_test.head()),
    )
    print("Verificación correcta: el modelo recuperado reproduce las predicciones.")