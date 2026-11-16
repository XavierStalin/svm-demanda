"""Predicciones SVR y RNN por Run ID, sin reentrenar ni depender del servidor web."""
from pathlib import Path
import argparse
import json
import os
import tempfile
import subprocess
import sys
from io import StringIO
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

BASE = Path(__file__).resolve().parent
TRACKING_URI = 'sqlite:///' + (BASE / 'mlflow_cuaderno.db').as_posix()


def _rnn_en_entorno_compatible(run_id, fechas, tracking_uri):
    """Aísla TensorFlow de los NumPy/protobuf del kernel habitual del usuario."""
    interprete = BASE / '.venv-rnn' / 'Scripts' / 'python.exe'
    if not interprete.is_file():
        raise RuntimeError('Falta .venv-rnn. Prepara ese entorno con requirements-rnn.txt.')
    codigo = (
        'import json,sys; from prueba_modelos_mlflow import probar_modelo_desde_mlflow; '
        'p=json.load(sys.stdin); '
        'r=probar_modelo_desde_mlflow(p["run_id"],p["fechas"],tracking_uri=p["uri"]); '
        'print("RESULTADO_JSON="+r.to_json(orient="split",date_format="iso",double_precision=15))'
    )
    proceso = subprocess.run([str(interprete), '-X', 'utf8', '-c', codigo], cwd=BASE,
        input=json.dumps({'run_id':run_id, 'fechas':[str(f) for f in fechas], 'uri':tracking_uri}),
        capture_output=True, text=True, encoding='utf-8',
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0,
        timeout=180)
    if proceso.returncode:
        raise RuntimeError('La prueba RNN en el entorno compatible falló:\n' + proceso.stderr[-2500:])
    lineas = [l for l in proceso.stdout.splitlines() if l.startswith('RESULTADO_JSON=')]
    if not lineas:
        raise RuntimeError('La RNN no devolvió su tabla de predicciones.')
    resultado = pd.read_json(StringIO(lineas[-1].split('=',1)[1]), orient='split')
    resultado['Fecha'] = pd.to_datetime(resultado['Fecha'])
    return resultado


def ids_ultima_comparacion():
    indices = list((BASE / 'resultados_cuaderno').glob('*/mlflow_tres_modelos.json'))
    if not indices:
        raise FileNotFoundError('Ejecuta la sección 16 del cuaderno original primero.')
    indice = max(indices, key=lambda p: p.stat().st_mtime_ns)
    return json.loads(indice.read_text(encoding='utf-8'))['run_ids']


def probar_modelo_desde_mlflow(run_id: str, fechas: list[str], *, tracking_uri=TRACKING_URI):
    """Recupera un modelo propio de confianza y devuelve predicciones en miles.

    La venta del día objetivo se usa solo para informar el error, nunca como entrada.
    Una fecha futura requiere los siete días anteriores completos de historia real.
    """
    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient(tracking_uri=tracking_uri)
    run = cliente.get_run(run_id)
    if run.info.status != 'FINISHED':
        raise ValueError('La ejecución debe estar finalizada.')
    raices = {p.path for p in cliente.list_artifacts(run_id)}
    es_rnn = 'modelo_rnn' in raices or 'unidades' in run.data.params
    compatible = BASE / '.venv-rnn' / 'Scripts' / 'python.exe'
    if es_rnn and Path(sys.executable).resolve() != compatible.resolve():
        return _rnn_en_entorno_compatible(run_id, fechas, tracking_uri)
    with tempfile.TemporaryDirectory(prefix='prueba_mlflow_') as temporal:
        def descargar(rid, ruta):
            return Path(cliente.download_artifacts(rid, ruta, temporal))
        if es_rnn:
            if 'modelo_rnn' in raices:
                carpeta = descargar(run_id, 'modelo_rnn')
                ruta_meta = descargar(run_id, 'procedencia/metadatos.json')
            elif 'modelo' in raices:
                carpeta = descargar(run_id, 'modelo')
                ruta_meta = descargar(run_id, 'resultados/metadatos.json')
            else:
                raise ValueError('Este candidato de validación no tiene modelo final guardado.')
            import rnn_demanda as rnn
            rnn.configurar_cpu()
            paquete = rnn.cargar(carpeta)
            origen = run.data.tags.get('origen_run_id', run_id)
            ruta_diario = descargar(origen, 'resultados/demanda_diaria.csv')
        else:
            salidas = getattr(run.outputs, 'model_outputs', []) if run.outputs else []
            if salidas:
                uri = 'models:/' + salidas[0].model_id
            elif 'pipeline_svr' in raices:
                uri = f'runs:/{run_id}/pipeline_svr'
            else:
                raise ValueError('Este ID no contiene un modelo SVR final guardado.')
            modelo = mlflow.sklearn.load_model(uri)
            ruta_meta = descargar(run_id, 'resultados/metadatos_experimento.json')
            ruta_diario = descargar(run_id, 'resultados/demanda_diaria.csv')
        meta = json.loads(ruta_meta.read_text(encoding='utf-8'))
        diario = pd.read_csv(ruta_diario, parse_dates=['fecha']).set_index('fecha')
        ventana = int(meta['ventana'])
        variables = meta.get('variables_historia', meta.get('variables'))
        especiales = meta.get('fechas_especiales_heredadas', ['2019-01-01', '2019-02-14'])
        diario['dia_semana'] = diario.index.dayofweek
        diario['mes'] = diario.index.month
        diario['es_fin_de_semana'] = (diario.index.dayofweek >= 5).astype(int)
        diario['fecha_especial'] = diario.index.isin(pd.to_datetime(especiales)).astype(int)
        if diario.index.has_duplicates:
            raise ValueError('El historial contiene fechas duplicadas.')
        filas = []
        for fecha in fechas:
            objetivo = pd.Timestamp(fecha).normalize()
            dias = pd.date_range(objetivo - pd.Timedelta(days=ventana), periods=ventana)
            historia = diario.reindex(dias)[variables]
            if not np.isfinite(historia.to_numpy(dtype=float)).all():
                raise ValueError(f'Falta historia completa para {objetivo.date()}.')
            if es_rnn:
                estimada = float(rnn.predecir(paquete, historia.to_numpy(dtype='float32')[None, :, :])[0])
            else:
                columnas = [f'{v}_lag{lag}' for lag in range(ventana, 0, -1) for v in variables]
                entrada = pd.DataFrame([historia.to_numpy().ravel()], columns=columnas)
                estimada = float(modelo.predict(entrada)[0])
            real = float(diario.loc[objetivo, 'demanda']) if objetivo in diario.index else np.nan
            filas.append({'Modelo': run.data.tags.get('mlflow.runName', 'RNN' if es_rnn else 'SVR'),
                'Run ID': run_id, 'Fecha': objetivo, 'Predicción (miles)': estimada,
                'Real (miles)': real, 'Error absoluto (miles)': abs(estimada - real)})
        return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='Si se omite, prueba los tres modelos de la última comparación completa.')
    parser.add_argument('--fechas', nargs='+', default=['2016-09-13', '2017-08-15'])
    args = parser.parse_args()
    ids = {'Modelo': args.run_id} if args.run_id else ids_ultima_comparacion()
    resultado = pd.concat([probar_modelo_desde_mlflow(rid, args.fechas) for rid in ids.values()], ignore_index=True)
    print(resultado.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
