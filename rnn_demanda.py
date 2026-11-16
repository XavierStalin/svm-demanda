"""SimpleRNN para demanda: validación temporal, escalado y recuperación."""
from time import perf_counter
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def configurar_cpu():
    """Limita hilos para no saturar la computadora durante la exposición."""
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        pass  # El kernel pudo inicializar TensorFlow en una celda anterior.
    tf.config.experimental.enable_op_determinism()


def crear_red(forma, unidades=16, learning_rate=0.001, semilla=42):
    keras.backend.clear_session()
    keras.utils.set_random_seed(semilla)
    modelo = keras.Sequential([
        keras.Input(shape=forma),
        keras.layers.SimpleRNN(unidades, activation="tanh", name="memoria_recurrente"),
        keras.layers.Dense(1, name="demanda_escalada"),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
                   loss="mse")
    return modelo


def ajustar_escaladores(X, y):
    # Una escala por variable histórica, compartida entre los siete pasos.
    return (StandardScaler().fit(X.reshape(-1, X.shape[-1])),
            StandardScaler().fit(np.asarray(y).reshape(-1, 1)))


def escalar_X(X, escalador):
    return escalador.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")


def dataset(X, y=None, batch_size=32):
    elementos = X if y is None else (X, y)
    datos = tf.data.Dataset.from_tensor_slices(elementos).batch(batch_size)
    opciones = tf.data.Options()
    opciones.threading.private_threadpool_size = 1
    opciones.threading.max_intra_op_parallelism = 1
    opciones.experimental_deterministic = True
    return datos.with_options(opciones).prefetch(1)


def predecir(paquete, X):
    """Devuelve demanda en miles de unidades, deshaciendo el escalado de y."""
    entrada = escalar_X(np.asarray(X), paquete["scaler_X"])
    salida = paquete["modelo"](tf.convert_to_tensor(entrada), training=False).numpy()
    return paquete["scaler_y"].inverse_transform(salida).ravel()


def elegir_epocas(X, y, parametros, semilla=42, max_epocas=80, paciencia=10):
    """EarlyStopping en el último 20% del TRAIN; nunca ve el pliegue externo."""
    corte = int(len(X) * 0.8)
    if corte < 10 or len(X) - corte < 2:
        raise ValueError("Historia insuficiente para validación interna temporal.")
    sx, sy = ajustar_escaladores(X[:corte], y[:corte])
    red = crear_red(X.shape[1:], **parametros, semilla=semilla)
    xs = escalar_X(X, sx)
    ys = sy.transform(np.asarray(y).reshape(-1, 1)).astype("float32")
    historia = red.fit(dataset(xs[:corte], ys[:corte]),
        validation_data=dataset(xs[corte:], ys[corte:]), epochs=max_epocas, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                    restore_best_weights=True), keras.callbacks.TerminateOnNaN()])
    curva = pd.DataFrame(historia.history)
    if not np.isfinite(curva[["loss", "val_loss"]].to_numpy()).all():
        raise RuntimeError("Entrenamiento RNN produjo pérdidas no finitas.")
    curva.index = pd.RangeIndex(1, len(curva) + 1, name="epoca")
    # Estas pérdidas pertenecen a la validación interna, no al test ni al CV externo.
    curva["RMSE_train"] = np.sqrt(curva["loss"]) * sy.scale_[0]
    curva["RMSE_validacion_interna"] = np.sqrt(curva["val_loss"]) * sy.scale_[0]
    return int(curva["val_loss"].idxmin()), curva


def entrenar_final(X, y, parametros, epocas, semilla=42):
    """Reajusta escaladores y red en TODO el train durante un número fijo de épocas."""
    tic = perf_counter()
    sx, sy = ajustar_escaladores(X, y)
    red = crear_red(X.shape[1:], **parametros, semilla=semilla)
    historia = red.fit(dataset(escalar_X(X, sx),
        sy.transform(np.asarray(y).reshape(-1, 1)).astype("float32")),
        epochs=int(epocas), verbose=0, callbacks=[keras.callbacks.TerminateOnNaN()])
    if not np.isfinite(historia.history["loss"]).all():
        raise RuntimeError("El reajuste RNN produjo pérdidas no finitas.")
    return {"modelo": red, "scaler_X": sx, "scaler_y": sy,
            "parametros": dict(parametros), "epocas": int(epocas),
            "segundos_ajuste": perf_counter() - tic}


def buscar_rnn(X, y, pliegues, candidatos, semilla=42, max_epocas=80, paciencia=10):
    """Mismos pliegues externos que SVR; épocas elegidas mediante holdout interno."""
    filas, curvas = [], {}
    tic = perf_counter()
    for candidato, parametros in enumerate(candidatos):
        for numero, (it, iv) in enumerate(pliegues, 1):
            assert it.max() < iv.min()
            inicio = perf_counter()
            epocas, curva = elegir_epocas(X[it], y[it], parametros,
                semilla=semilla, max_epocas=max_epocas, paciencia=paciencia)
            paquete = entrenar_final(X[it], y[it], parametros, epocas, semilla)
            pred = predecir(paquete, X[iv])
            filas.append({"candidato": candidato, "pliegue": numero, **parametros,
                "epocas": epocas, "RMSE_CV": float(np.sqrt(mean_squared_error(y[iv], pred))),
                "segundos": perf_counter() - inicio})
            curvas[(candidato, numero)] = curva
            print(f"RNN {candidato + 1}/{len(candidatos)} | pliegue {numero}/{len(pliegues)} "
                  f"| épocas={epocas} | RMSE={filas[-1]['RMSE_CV']:.3f}", flush=True)
            del paquete
    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby("candidato", sort=True).agg(
        RMSE_CV=("RMSE_CV", "mean"), std_CV=("RMSE_CV", "std"),
        epocas=("epocas", "median"), segundos_busqueda=("segundos", "sum"))
    resumen["epocas"] = resumen["epocas"].round().astype(int)
    for i, parametros in enumerate(candidatos):
        for nombre, valor in parametros.items():
            resumen.loc[i, nombre] = valor
    ganador = int(resumen["RMSE_CV"].idxmin())
    return resumen, detalle, curvas, ganador, perf_counter() - tic


def guardar(paquete, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    paquete["modelo"].save(carpeta / "modelo.keras")
    joblib.dump({k: v for k, v in paquete.items() if k != "modelo"}, carpeta / "preparacion.joblib")


def cargar(carpeta):
    # Cargar únicamente artefactos propios y de confianza.
    paquete = joblib.load(carpeta / "preparacion.joblib")
    paquete["modelo"] = keras.models.load_model(carpeta / "modelo.keras", compile=False)
    return paquete
