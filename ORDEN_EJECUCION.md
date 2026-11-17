# Orden de ejecución

Abrir los cuadernos desde la carpeta `svm-demanda`. Para los tres se recomienda
el intérprete `.venv-rnn/Scripts/python.exe` (Python 3.12).
No instalar `requirements.txt` sobre este entorno: usa NumPy 2 y el entorno RNN usa NumPy 1.26.

| Orden | Cuaderno | Resultado |
|---|---|---|
| 1 | `Comparacion_RNN_vs_SVM_GROCERY_I_MLflow.ipynb` | Entrena y valida la RNN, guarda modelos, escaladores, predicciones y registros MLflow. |
| 2 | `Comparacion_SVM_GROCERY_I_MLflow.ipynb` | Entrena las dos SVM, incorpora la RNN compatible más reciente y registra los tres modelos juntos. |
| 3 | `Prueba_3_Modelos_por_ID.ipynb` | Recupera los modelos por Run ID y compara sus predicciones sin reentrenar. |

Ejecutar todas las celdas en orden y esperar a que termine cada cuaderno antes de abrir la siguiente ejecución.
En el primero mantener `REGISTRAR_MLFLOW = True`. En el segundo, dejar `INICIAR_UI = False`
durante la ejecución completa. La base de datos se actualiza aunque la interfaz esté apagada.

## Abrir MLflow al terminar

Desde PowerShell, en `svm-demanda`:

```powershell
& .\.venv-rnn\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow_cuaderno.db --host 127.0.0.1 --port 5002 --no-serve-artifacts --workers 1
```

Si http://127.0.0.1:5002 ya funciona, usar esa instancia y no iniciar otra.
En el experimento `Comparacion_SVR_GROCERY_I_Cuaderno`, usar el filtro que imprime
la sección 16 del segundo cuaderno. Seleccionar solo los tres modelos principales.
Cada ejecución completa crea nuevos Run ID: el cuaderno de prueba obtiene automáticamente
los de la última comparación finalizada. Para reproducir la exposición ya preparada también
puede usarse el grupo fijo indicado en el guion.

## Para una demostración con los resultados ya preparados

Si no cambian los datos ni la configuración, no es necesario reentrenar.
Se puede abrir MLflow y ejecutar únicamente `Prueba_3_Modelos_por_ID.ipynb`.
Conservar `resultados_rnn`, `resultados_cuaderno`, `mlflow_cuaderno.db` y `mlartifacts_cuaderno`.


## Exponer directamente en MLflow

En el experimento `Comparacion_SVR_GROCERY_I_Cuaderno`:

1. **Models:** usar el filtro de Models impreso al final del cuaderno. Aparecen `SVM_inicial`, `SVM_optimizada` y `RNN`, con sus Model ID y métricas. Se retiraron del catálogo los ocho pipeline_svr antiguos; sus ejecuciones siguen disponibles en Runs.
2. **Runs:** filtrar `tags.comparacion = '<grupo>' and tags.tipo = 'modelo_principal'` para comparar MAE, RMSE y R2 de los tres.
3. Abrir **SVR optimizada**: su descripción explica qué candidato ganó y por qué. En **Artifacts > explicacion_mlflow** abrir `01_candidatos_y_elegido.png` y `02_rmse_por_fechas_validacion.png`. Las tablas CSV incluyen las fechas y el RMSE de cada candidato por pliegue. El mapa C/kernel sigue en **Artifacts > resultados > optimizacion.png**.
4. Quitar el filtro `tipo = 'modelo_principal'` y expandir la SVM optimizada para ver los 16 candidatos hijos. El ganador se llama **ELEGIDO | linear | C=1**. Sus parámetros muestran las fechas de validación.

Para publicar esta vista sobre la comparación más reciente ya terminada, sin reentrenar:

```powershell
& .\.venv-rnn\Scripts\python.exe -X utf8 mlflow_presentacion.py
```

Este comando es idempotente para la misma comparación: reutiliza los tres modelos con nombre. La RNN incluye la red y los escaladores; se comprueba que los tres modelos recuperados reproduzcan las 337 predicciones de prueba.


## Model Registry

Se registran `SVM_inicial`, `SVM_optimizada` y `RNN`, vinculados a su Model ID y Run ID. Cada registro tiene el alias `comparacion_actual`, que apunta a la versión de la comparación publicada. Las descripciones incluyen MAE, RMSE y R2 de prueba. Repetir la publicación del mismo Model ID no crea otra versión; una nueva comparación crea una versión nueva. Este alias identifica la comparación, no un despliegue en producción.
