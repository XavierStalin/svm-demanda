"""Evaluación de modelos: métricas, tablas comparativas, gráficos y reportes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from svm_demanda import config


def evaluar_modelos(modelo_base, modelo_optimizado, X_test, y_test,
                    tiempo_base: float, tiempo_optimizado: float) -> pd.DataFrame:
    """Calcula MAE y RMSE para Base, Optimizada y Persistencia, y guarda el CSV."""
    pred_base = modelo_base.predict(X_test)
    pred_opt = modelo_optimizado.predict(X_test)
    # La persistencia es simplemente predecir el valor del día anterior (lag1)
    pred_persistencia = X_test["demanda_lag1"].to_numpy()

    filas = []
    for nombre, pred, tiempo in [
        ("SVR inicial", pred_base, tiempo_base),
        ("SVR optimizada", pred_opt, tiempo_optimizado),
        ("Persistencia", pred_persistencia, 0.0),
    ]:
        filas.append({
            "Modelo": nombre,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "segundos_ajuste": tiempo,
        })
    
    comparacion = pd.DataFrame(filas).set_index("Modelo")
    comparacion.round(4).to_csv(config.RESULTADOS / "comparacion.csv")
    return comparacion


def imprimir_mejoras(comparacion: pd.DataFrame) -> None:
    """Imprime la reducción porcentual de MAE y RMSE."""
    for metrica in ["MAE", "RMSE"]:
        inicial = comparacion.loc["SVR inicial", metrica]
        optimizada = comparacion.loc["SVR optimizada", metrica]
        mejora = 100 * (inicial - optimizada) / inicial if inicial else np.nan
        print(f"Reducción de {metrica} en prueba: {mejora:.2f}%")


def generar_predicciones(prueba: pd.DataFrame, modelo_base, modelo_optimizado,
                         X_test) -> pd.DataFrame:
    """Crea el DataFrame con las predicciones de ambos modelos y lo guarda."""
    predicciones = prueba[["fecha", "producto", "demanda"]].copy()
    predicciones["SVR inicial"] = modelo_base.predict(X_test)
    predicciones["SVR optimizada"] = modelo_optimizado.predict(X_test)
    predicciones.to_csv(config.RESULTADOS / "predicciones.csv", index=False)
    return predicciones


def preparar_gridsearch(gs: GridSearchCV) -> pd.DataFrame:
    """Extrae los resultados del GridSearchCV, calcula el RMSE real y guarda el CSV."""
    resultados_cv = pd.DataFrame(gs.cv_results_)
    # El score de sklearn es negativo (para maximizar), lo volvemos positivo
    resultados_cv["RMSE_CV"] = -resultados_cv["mean_test_score"]
    resultados_cv.to_csv(config.RESULTADOS / "gridsearch.csv", index=False)
    return resultados_cv


def graficar_comparacion(comparacion: pd.DataFrame, predicciones: pd.DataFrame,
                         categoria_vista: str) -> None:
    """Genera y guarda el gráfico de barras de métricas y series de tiempo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico 1: Barras de MAE y RMSE
    comparacion[["MAE", "RMSE"]].plot.bar(ax=axes[0], rot=15)
    axes[0].set_ylabel("Unidades")
    axes[0].set_title("Métricas en prueba")

    # Gráfico 2: Serie de tiempo para una categoría
    vista = predicciones[predicciones["producto"] == categoria_vista]
    for nombre in ["demanda", "SVR inicial", "SVR optimizada"]:
        axes[1].plot(vista["fecha"], vista[nombre], marker=".", label=nombre)
    axes[1].set_title(categoria_vista)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.RESULTADOS / "comparacion.png", dpi=150)
    plt.close(fig)  # Cierra la figura para que no se quede abierta en memoria


def graficar_optimizacion(resultados_cv: pd.DataFrame) -> None:
    """Genera y guarda el mapa de calor de C vs Kernel."""
    mapa = resultados_cv.pivot_table(
        index="param_modelSVM__C", columns="param_modelSVM__kernel",
        values="RMSE_CV", aggfunc="min",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mapa.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(mapa.columns)), labels=mapa.columns)
    ax.set_yticks(range(len(mapa.index)), labels=mapa.index)
    ax.set_xlabel("Kernel")
    ax.set_ylabel("C")
    ax.set_title("RMSE de validación por C y kernel")

    for i in range(len(mapa.index)):
        for j in range(len(mapa.columns)):
            ax.text(j, i, f"{mapa.iloc[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()
    fig.savefig(config.RESULTADOS / "optimizacion.png", dpi=150)
    plt.close(fig)