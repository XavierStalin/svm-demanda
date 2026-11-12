"""MLflow: arranque del servidor local y registro de las ejecuciones."""
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from svm_demanda import config

PAQUETES_MODELO = ["scikit-learn", "pandas", "numpy"]


def servidor_disponible() -> bool:
    try:
        with urlopen(config.MLFLOW_URI + "/health", timeout=2) as respuesta:
            return respuesta.status == 200
    except OSError:
        return False


def iniciar_servidor() -> None:
    if servidor_disponible():
        return
    url = urlparse(config.MLFLOW_URI)
    comando = [
        sys.executable, "-m", "mlflow", "server",
        "--host", url.hostname, "--port", str(url.port),
        "--backend-store-uri", f"sqlite:///{config.MLFLOW_DB.name}",
        "--default-artifact-root", config.MLFLOW_ARTIFACTS.as_uri(),
    ]
    opciones = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    with open(config.BASE / "mlflow_demanda.log", "a", encoding="utf-8") as registro:
        subprocess.Popen(comando, cwd=config.BASE, stdout=registro,
                         stderr=subprocess.STDOUT, **opciones)
    for _ in range(60):
        if servidor_disponible():
            break
        time.sleep(1)
    else:
        raise RuntimeError("El servidor MLflow no responde. Revisa mlflow_demanda.log.")


def _requisitos() -> list[str]:
    import importlib.metadata as metadata
    return [f"{p}=={metadata.version(p)}" for p in PAQUETES_MODELO]


def registrar_comparacion(modelos: dict, comparacion: pd.DataFrame, gs,
                          X_train: pd.DataFrame, X_test: pd.DataFrame,
                          fecha_prueba: pd.Timestamp,
                          tiempo_busqueda: float,
                          grupo: str) -> dict:
    mlflow.sklearn.autolog(disable=True)
    iniciar_servidor()
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.EXPERIMENTO)
    if mlflow.active_run() is not None:
        raise RuntimeError("Hay una ejecución MLflow activa. Finalízala antes de registrar.")

    uris = {}
    for nombre, modelo in modelos.items():
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tags({
                "comparacion": grupo, "objetivo": "demanda_diaria",
                "dataset": "store_sales", "protocolo": "un_paso_diario",
            })
            mlflow.log_params({
                "ventana": config.VENTANA,
                "inicio_prueba": str(fecha_prueba.date()),
                "filas_train": len(X_train),
                "filas_test": len(X_test),
                **modelo.named_steps["modelSVM"].get_params(),
            })
            mlflow.log_metrics({
                "test_mae": comparacion.loc[nombre, "MAE"],
                "test_rmse": comparacion.loc[nombre, "RMSE"],
                "segundos_ajuste": comparacion.loc[nombre, "segundos_ajuste"],
            })
            if nombre == "SVR optimizada":
                mlflow.log_metric("cv_rmse", -gs.best_score_)
                mlflow.log_metric("segundos_busqueda", tiempo_busqueda)
            mlflow.log_artifacts(str(config.RESULTADOS), artifact_path="resultados")

            ejemplo = X_train.head(3).copy()
            ejemplo.columns = pd.Index(ejemplo.columns, dtype=object)
            info = mlflow.sklearn.log_model(
                sk_model=modelo, name="pipeline_svr", input_example=ejemplo,
                pip_requirements=_requisitos(),
                signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            )
            uris[nombre] = info.model_uri
    return uris


def registrar_candidatos_gridsearch(gs, grupo: str) -> None:
    """Registra cada candidato del GridSearchCV como una run independiente."""
    resultados = pd.DataFrame(gs.cv_results_)
    resultados["RMSE_CV"] = -resultados["mean_test_score"]

    for idx, fila in resultados.iterrows():
        with mlflow.start_run(run_name=f"candidato_{idx:02d}"):  # ← SIN nested=True
            mlflow.set_tags({
                "comparacion": grupo,
                "tipo": "candidato_gridsearch",
            })
            params = fila["params"]
            mlflow.log_params({
                "C": params["modelSVM__C"],
                "kernel": params["modelSVM__kernel"],
                "epsilon": params["modelSVM__epsilon"],
                "gamma": params["modelSVM__gamma"],
            })
            mlflow.log_metrics({
                "cv_rmse": fila["RMSE_CV"],
                "cv_rmse_std": fila["std_test_score"],
                "rank": int(fila["rank_test_score"]),
            })