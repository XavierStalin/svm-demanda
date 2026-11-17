"""Publica en MLflow nombres, gráficos y selección sin volver a entrenar."""
from pathlib import Path
import json
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient

BASE = Path(__file__).resolve().parent

class DemandaRNN(mlflow.pyfunc.PythonModel):
    """Incluye los dos escaladores y la red; acepta las mismas 35 columnas que SVR."""
    def load_context(self, context):
        import joblib
        import rnn_demanda as rnn
        rnn.configurar_cpu()
        self.paquete = joblib.load(context.artifacts["preparacion"])
        self.paquete["modelo"] = rnn.keras.models.load_model(context.artifacts["red"], compile=False)

    def predict(self, context, model_input, params=None):
        import rnn_demanda as rnn
        x = np.asarray(model_input, dtype="float32").reshape(-1, 7, 5)
        return rnn.predecir(self.paquete, x)


def publicar(carpeta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    carpeta = Path(carpeta).resolve()
    indice = json.loads((carpeta / "mlflow_tres_modelos.json").read_text(encoding="utf-8"))
    grupo, experimento = indice["grupo"], str(indice["experiment_id"])
    uri = "sqlite:///" + (BASE / "mlflow_cuaderno.db").as_posix()
    mlflow.set_tracking_uri(uri)
    cliente = MlflowClient(tracking_uri=uri)
    ids = indice["run_ids"]
    meta = json.loads((carpeta / "metadatos_experimento.json").read_text(encoding="utf-8"))
    diario = pd.read_csv(carpeta / "demanda_diaria.csv", parse_dates=["fecha"]).set_index("fecha")
    diario["dia_semana"] = diario.index.dayofweek
    diario["mes"] = diario.index.month
    diario["es_fin_de_semana"] = (diario.index.dayofweek >= 5).astype(int)
    diario["fecha_especial"] = diario.index.isin(pd.to_datetime(meta["fechas_especiales_heredadas"])).astype(int)
    variables = meta["variables_historia"]
    x = pd.concat([diario[v].shift(lag).rename(f"{v}_lag{lag}")
        for lag in range(meta["ventana"], 0, -1) for v in variables], axis=1).dropna()
    x = x[meta["columnas_modelo"]]
    test = x.loc[meta["inicio_test"]:meta["fin_test"]]
    esperado = pd.read_csv(carpeta / "predicciones_tres_modelos.csv")
    modelos = {}
    for nombre, rid in ids.items():
        nombre_modelo = {"SVR inicial": "SVM_inicial", "SVR optimizada": "SVM_optimizada", "RNN": "RNN"}[nombre]
        run = cliente.get_run(rid)
        tags = {"comparacion": grupo, "vista": "comparacion_tres", "modelo": nombre_modelo,
                "fecha_ejecucion": grupo[:8], "unidad": "miles_de_unidades"}
        encontrados = [m for m in cliente.search_logged_models([experimento])
            if m.source_run_id == rid and m.name == nombre_modelo]
        if encontrados:
            mid = encontrados[0].model_id
        else:
            with mlflow.start_run(run_id=rid):
                if nombre != "RNN":
                    salidas = run.outputs.model_outputs
                    modelo = mlflow.sklearn.load_model("models:/" + salidas[0].model_id)
                    info = mlflow.sklearn.log_model(modelo, name=nombre_modelo, input_example=x.head(2),
                        signature=mlflow.models.infer_signature(x.head(2), modelo.predict(x.head(2))),
                        pip_requirements=["scikit-learn", "numpy", "pandas", "cloudpickle"])
                else:
                    red = cliente.download_artifacts(rid, "modelo_rnn/modelo.keras")
                    prep = cliente.download_artifacts(rid, "modelo_rnn/preparacion.joblib")
                    info = mlflow.pyfunc.log_model(name=nombre_modelo, python_model=DemandaRNN(),
                        artifacts={"red": red, "preparacion": prep},
                        code_paths=[str(BASE / "mlflow_presentacion.py"), str(BASE / "rnn_demanda.py")],
                        signature=mlflow.models.infer_signature(x.head(2), np.zeros(2)),
                        input_example=x.head(2),
                        pip_requirements=["tensorflow==2.16.2", "numpy==1.26.4", "protobuf==4.25.9",
                            "pandas", "scikit-learn", "joblib", "cloudpickle"])
                mid = info.model_id
        cliente.set_logged_model_tags(mid, tags)
        mlflow.log_metrics({k: v for k, v in run.data.metrics.items()
            if k in ["test_mae", "test_rmse", "test_r2", "cv_rmse"]}, run_id=rid, model_id=mid)
        pred = mlflow.pyfunc.load_model("models:/" + mid).predict(test)
        np.testing.assert_allclose(np.asarray(pred).ravel(), esperado[nombre], atol=2e-4, rtol=1e-6)
        modelos[nombre] = {"model_id": mid, "model_uri": "models:/" + mid}
        cliente.set_tag(rid, "nombre_modelo", nombre_modelo)
        print("Verificado:", nombre_modelo, mid, flush=True)

    cv = pd.read_csv(carpeta / "gridsearch.csv")
    fechas = pd.read_csv(carpeta / "pliegues_temporales.csv")
    ganador = int(cv["RMSE_CV"].idxmin())
    mejor = cv.loc[ganador]
    destino = carpeta / "explicacion_mlflow"
    destino.mkdir(exist_ok=True)
    etiquetas = [f"{r.param_modelSVM__kernel} | C={r.param_modelSVM__C:g}" for r in cv.itertuples()]
    orden = cv["RMSE_CV"].sort_values().index
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(orden)), cv.loc[orden, "RMSE_CV"],
        color=["#16846a" if i == ganador else "#8195a5" for i in orden])
    ax.set_yticks(range(len(orden)), [etiquetas[i] + ("  ELEGIDO" if i == ganador else "") for i in orden])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Escala logarítmica — RMSE medio de validación temporal (miles de unidades); menor es mejor")
    ax.set_title(f"16 candidatos SVM | ejecución {grupo[:8]}\nElegido: {etiquetas[ganador]} | RMSE CV={mejor.RMSE_CV:.3f}")
    fig.tight_layout();fig.savefig(destino / "01_candidatos_y_elegido.png", dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6))
    detalle = []
    for i, fila in cv.iterrows():
        errores = [-float(fila[f"split{j}_test_score"]) for j in range(len(fechas))]
        ax.plot(range(1, len(fechas)+1), errores, marker="o", color="#16846a" if i == ganador else "#9da8b1",
            alpha=1 if i == ganador else .35, linewidth=3 if i == ganador else 1,
            label=f"Elegido: {etiquetas[i]}" if i == ganador else None, zorder=3 if i == ganador else 1)
        for j, f in fechas.iterrows():
            detalle.append({"candidato": i, "kernel": fila["param_modelSVM__kernel"], "C": fila["param_modelSVM__C"],
                "elegido": i == ganador, "validacion_desde": f["Validación desde"],
                "validacion_hasta": f["Validación hasta"], "RMSE": errores[j]})
    ax.set_xticks(range(1,len(fechas)+1), [f'{f["Validación desde"]}\na {f["Validación hasta"]}' for _, f in fechas.iterrows()])
    ax.set_yscale("log")
    ax.set_ylabel("RMSE de validación (miles de unidades; escala logarítmica)")
    ax.set_title("Candidatos evaluados en las mismas fechas\nVerde: menor RMSE PROMEDIO de los tres pliegues")
    ax.legend();fig.tight_layout();fig.savefig(destino / "02_rmse_por_fechas_validacion.png", dpi=160);plt.close(fig)
    pd.DataFrame(detalle).to_csv(destino / "candidatos_fechas_rmse.csv", index=False)
    ranking = cv[["param_modelSVM__kernel", "param_modelSVM__C", "RMSE_CV", "rank_test_score"]].copy()
    ranking["elegido"] = ranking.index == ganador
    ranking.sort_values("RMSE_CV").to_csv(destino / "ranking_candidatos.csv", index=False)
    fechas.to_csv(destino / "fechas_validacion.csv", index=False)
    rid = ids["SVR optimizada"]
    resumen = (f"## Selección de SVM optimizada\n"
        f"Ejecución: {grupo}. Se evaluaron {len(cv)} candidatos con los mismos tres pliegues temporales.\n\n"
        f"**Elegido: kernel={mejor['param_modelSVM__kernel']}, C={mejor['param_modelSVM__C']:g}, epsilon=0.1. "
        f"RMSE medio CV={mejor.RMSE_CV:.4f}.**\n\n"
        "Se elige por el menor RMSE PROMEDIO de validación; el test se reserva para la comparación final. "
        "Las fechas de cada pliegue están en Artifacts > explicacion_mlflow > 02_rmse_por_fechas_validacion.png. "
        "El ranking completo está en 01_candidatos_y_elegido.png. El mapa C/kernel se conserva en resultados/optimizacion.png.")
    cliente.set_tag(rid, "mlflow.note.content", resumen)
    cliente.set_tag(rid, "criterio_seleccion", "Menor RMSE medio en 3 pliegues temporales (no test)")
    cliente.set_tag(rid, "candidato_elegido", etiquetas[ganador])
    cliente.log_artifacts(rid, str(destino), artifact_path="explicacion_mlflow")
    candidatos = cliente.search_runs([experimento], f"tags.comparacion = '{grupo}' and tags.tipo = 'candidato_gridsearch'", max_results=100)
    for c in candidatos:
        indice_c = next((i for i,r in cv.iterrows() if
            str(r['param_modelSVM__kernel']) == c.data.params.get('kernel') and
            float(r['param_modelSVM__C']) == float(c.data.params.get('C', 'nan'))), None)
        if indice_c is None: continue
        elegido = indice_c == ganador
        cliente.set_tag(c.info.run_id, "mlflow.runName", f"{'ELEGIDO' if elegido else 'Candidato'} | {etiquetas[indice_c]}")
        cliente.set_tag(c.info.run_id, "seleccionado", str(elegido))
        cliente.set_tag(c.info.run_id, "mlflow.note.content", resumen if elegido else
            "Candidato GridSearchCV: mismas fechas de validación que los demás. Ver parámetros y métricas por pliegue.")
        for j,f in fechas.iterrows():
            cliente.log_param(c.info.run_id, f"validacion_{j+1}_desde", f["Validación desde"])
            cliente.log_param(c.info.run_id, f"validacion_{j+1}_hasta", f["Validación hasta"])
    for nombre, rid in ids.items():
        cliente.set_tag(rid, "fecha_comparacion", grupo[:8])
        if nombre != "SVR optimizada":
            cliente.set_tag(rid, "mlflow.note.content", f"{nombre}. Comparación {grupo}. "
                "MAE y RMSE: menor es mejor. R2: mayor es mejor. Las predicciones están en miles de unidades. "
                "La selección de candidatos se explica en la ejecución SVR optimizada > Artifacts > explicacion_mlflow.")
    indice["logged_models"] = modelos
    indice["filtro_models"] = f"tags.comparacion = '{grupo}' AND tags.vista = 'comparacion_tres'"
    (carpeta / "mlflow_tres_modelos.json").write_text(json.dumps(indice, indent=2), encoding="utf-8")
    visibles = cliente.search_logged_models([experimento], filter_string=indice["filtro_models"])
    assert len(visibles) == 3, f"Se esperaban 3 modelos, hay {len(visibles)}"
    assert {m.name for m in visibles} == {"SVM_inicial", "SVM_optimizada", "RNN"}
    print("MODELS: exactamente tres modelos con filtro:", indice["filtro_models"], flush=True)
    return indice


def publicar_compatible(carpeta):
    """Permite ejecutar el cuaderno desde un kernel sin TensorFlow compatible."""
    import subprocess, sys
    python = BASE / ".venv-rnn" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() == python.resolve():
        return publicar(carpeta)
    p = subprocess.run([str(python), "-X", "utf8", str(Path(__file__).resolve()), str(carpeta)],
        cwd=BASE, capture_output=True, text=True, encoding="utf-8", timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    print(p.stdout)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return json.loads((Path(carpeta) / "mlflow_tres_modelos.json").read_text(encoding="utf-8"))

if __name__ == "__main__":
    import sys
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
        (BASE / "resultados_cuaderno").glob("*/mlflow_tres_modelos.json"), key=lambda p:p.stat().st_mtime_ns).parent
    publicar(carpeta)
