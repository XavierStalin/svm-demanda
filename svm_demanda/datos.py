"""Carga del dataset Store Sales (Kaggle) y preparación de la demanda diaria."""
import pandas as pd

from svm_demanda import config



def cargar_diario() -> pd.DataFrame:

    """Lee train.csv, filtra una familia y suma sus ventas por día.

    La demanda queda en miles de unidades para que el SVR trabaje en una
    escala cómoda (los MAE/RMSE se interpretan en miles).
    """
    if not config.ARCHIVO_TRAIN.exists():
        raise FileNotFoundError(
            "Falta data/raw/train.csv. Descargalo de la competencia "
            "'store-sales-time-series-forecasting' de Kaggle (acepta las "
            "reglas primero) y ponlo en data/raw/. Ver README.md."
        )      
    
    train = pd.read_csv(
        config.ARCHIVO_TRAIN,
        usecols=["date", "family", "sales"],
        parse_dates=["date"],
    )
    ventas = train.loc[train["family"] == config.FAMILIA, ["date", "sales"]]
    if ventas.empty:
        raise ValueError(f"La familia {config.FAMILIA!r} no existe en train.csv.")

    serie = ventas.groupby("date")["sales"].sum().div(1000).round(3)

    # Una fila por día aunque falten ventas (los lags dependen de esto)
    fechas = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    return pd.DataFrame({
        "fecha": fechas,
        "producto": config.FAMILIA,
        "demanda": serie.reindex(fechas, fill_value=0.0).to_numpy(),
    })