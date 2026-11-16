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
