"""Feature engineering: variables de calendario y rezagos (ventana deslizante)."""
import pandas as pd

from svm_demanda import config


def agregar_calendario(diario: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables de calendario calculadas con la fecha de cada fila."""
    diario = diario.copy()
    diario["dia_semana"] = diario["fecha"].dt.dayofweek
    diario["mes"] = diario["fecha"].dt.month
    diario["es_fin_de_semana"] = (diario["dia_semana"] >= 5).astype(int)
    diario["fecha_especial"] = diario["fecha"].isin(
        pd.to_datetime(config.FECHAS_ESPECIALES)
    ).astype(int)
    return diario


def construir_lags(diario: pd.DataFrame,
                   ventana: int = config.VENTANA,
                   ) -> tuple[pd.DataFrame, list[str]]:
    """Convierte el historial en columnas: los `ventana` días previos a cada fecha.

    Devuelve la tabla de ventanas y los nombres de las features creadas.
    """
    tabla = diario.sort_values(["producto", "fecha"]).copy()
    columnas: list[str] = []
    for retraso in range(ventana, 0, -1):
        for variable in config.VARIABLES_HISTORIA:
            nombre = f"{variable}_lag{retraso}"
            tabla[nombre] = tabla.groupby("producto")[variable].shift(retraso)
            columnas.append(nombre)
    tabla = (
        tabla.dropna(subset=columnas)
        .sort_values(["fecha", "producto"])
        .reset_index(drop=True)
    )
    return tabla, columnas