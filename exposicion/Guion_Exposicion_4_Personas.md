# Guion de exposición: comparación SVM y RNN

**Integrantes:** Daniel, Xavier, Angélica y José.  
**Duración aproximada:** 4-5 minutos, incluida una demostración breve de MLflow.  
**Presentación:** `Comparacion_SVM_RNN_GROCERY_I.pptx`. Mostrar las diapositivas y los resultados preparados, sin código. Cada integrante mantiene una intervención continua.

| Integrante | Diapositivas | Tema |
|---|---|---|
| Daniel | 1-2 | Objetivo, datos y evaluación |
| Xavier | 3-4 | Selección de modelos y aprendizaje de la RNN |
| Angélica | 5-6 | Errores y predicciones |
| José | 7-8 y MLflow | Evidencia, demostración y conclusión |

## Daniel: objetivo y datos

*Mostrar la portada y avanzar a la diapositiva 2. Señalar la parte dorada del historial, que corresponde a la prueba.*

> Buenos días. Comparamos una SVM de regresión con una red neuronal recurrente para predecir las ventas diarias de GROCERY I, agregadas entre 54 tiendas. Usamos los siete días anteriores para estimar el siguiente. El gráfico muestra las variaciones que buscamos aprender. Reservamos 1344 días para entrenar y los últimos 337 para probar, respetando el orden temporal. Cada predicción supone que conocemos las ventas reales anteriores. Las cantidades se expresan en miles de unidades. Xavier explicará la selección de los modelos.

## Xavier: selección y aprendizaje

*En la diapositiva 3, señalar las barras de menor error. Pasar a la 4 para explicar las curvas y la configuración RNN.*

> Ambos modelos reciben la misma información: 35 columnas en SVM y una secuencia de siete días por cinco variables en RNN. Para SVM usamos GridSearchCV: 16 combinaciones de kernel y C, evaluadas en tres pliegues temporales, equivalentes a 48 ajustes. Elegimos el menor RMSE de validación, que correspondió al kernel lineal con C igual a uno. MLflow es compatible: registramos explícitamente los parámetros y errores de cada candidato y guardamos el modelo elegido. La RNN utiliza una búsqueda temporal propia de cuatro configuraciones. Conservó 16 neuronas, tasa 0.001 y 53 épocas finales, por lo que su versión inicial y seleccionada coinciden.

## Angélica: errores y predicciones

*En la diapositiva 5, comparar las barras de SVM optimizada y RNN. En la 6, señalar las curvas y algún pico subestimado.*

> El MAE mide el error absoluto promedio y el RMSE penaliza más los errores grandes. La SVM optimizada obtuvo un MAE de 24,74 y un RMSE de 46,93. La RNN alcanzó 23,47 y 43,45, respectivamente, en miles de unidades. Esto representa un 7,41 por ciento menos RMSE frente a la SVM optimizada. En los últimos 60 días, ambos modelos siguen el patrón semanal, aunque subestiman algunos picos. La RNN tiene menor error global en esta prueba, pero no acierta mejor en todas las fechas.

## José: evidencia, MLflow y cierre

*Mostrar la diapositiva 7. Cambiar a MLflow durante unos 40 segundos: comparación de tres modelos y artefactos de la RNN. Después, mostrar el ejemplo de la diapositiva 8.*

> Al excluir el día imputado de prueba, la RNN también conserva menor RMSE: 39,40 frente a 40,96 de la SVM optimizada. En MLflow podemos revisar los tres modelos principales: SVM inicial, SVM optimizada y RNN, además de 16 candidatos SVM. Sus artefactos incluyen las predicciones y los modelos guardados. Con el Run ID recuperamos cada modelo y probamos la misma función sin reentrenarlo. En el ejemplo del 15 de agosto, la RNN estimó 213,68 frente a 224,21 reales y tuvo el menor error. Concluimos que la RNN presentó menor error en este experimento. El resultado corresponde a predicciones de un día con historia real disponible. Gracias.

## Preparación breve: no se lee al público

- Abrir el PowerPoint en modo presentación. No hace falta mostrar el cuaderno ni la vista HTML anterior.
- Dejar MLflow funcionando en **http://127.0.0.1:5002**. Abrir el experimento **Comparacion_SVR_GROCERY_I_Cuaderno**, identificador **1**.
- Filtrar por `tags.comparacion = '20261116_102624_b851b1' and tags.tipo = 'modelo_principal'`. Preparar **Compare** con las tres ejecuciones principales y las métricas `test_mae` y `test_rmse`. Quitar C y kernel de la comparación, porque no aplican a RNN. Los 16 candidatos SVM tienen métricas de validación.
- Dejar otra pestaña con los artefactos de **RNN**. La carpeta `modelo_rnn` contiene `modelo.keras` y `preparacion.joblib`, con los escaladores. Xavier ayuda a José a preparar las pestañas.
- En MLflow aparece una sola RNN porque la búsqueda conservó su configuración inicial. El ejemplo es una fecha histórica de prueba, no una predicción para una fecha futura.

**Referencia rápida:** RMSE SVM inicial **66,45**; SVM optimizada **46,93**; RNN inicial y seleccionada **43,45**; persistencia **67,28**. La reducción **7,41%** compara RNN con SVM optimizada. Las métricas usan miles de unidades. El historial conserva cuatro fechas ausentes imputadas con cero, una de ellas en prueba. La comparación adicional excluye esa respuesta, conservando los mismos modelos.

## Comandos para la demostración

### 1. Abrir MLflow desde PowerShell

Si MLflow ya abre en el navegador, no iniciar otro servidor. De lo contrario, ejecutar:

```powershell
Set-Location -LiteralPath 'C:\Users\ASUS\Desktop\7mo Ciclo\Aprendizaje automático\Repositorio TrABAJO U2\svm-demanda'

& .\.venv-rnn\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow_cuaderno.db --host 127.0.0.1 --port 5002 --no-serve-artifacts --workers 1
```

Dejar esa terminal abierta. Abrir **http://127.0.0.1:5002** y entrar al experimento **Comparacion_SVR_GROCERY_I_Cuaderno**.

### 2. Comparar los tres modelos en MLflow

Pegar este filtro en **Training runs** (este filtro va en MLflow, no en PowerShell):

```text
tags.comparacion = '20261116_102624_b851b1' and tags.tipo = 'modelo_principal'
```

Seleccionar **SVR inicial**, **SVR optimizada** y **RNN**, y pulsar **Compare**. Mostrar `test_mae` y `test_rmse`. Quitar C y kernel de Parameters para la comparación conjunta.

### 3. Probar los tres modelos por ID desde otra terminal

Desde la carpeta `svm-demanda`, ejecutar:

```powershell
& .\.venv-rnn\Scripts\python.exe .\probar_candidato.py
```

Para probar únicamente la RNN del grupo indicado con las mismas dos fechas:

```powershell
& .\.venv-rnn\Scripts\python.exe .\probar_candidato.py --run-id 5c7f54a1107c4c859924ffd84450ebda --fechas 2016-09-13 2017-08-15
```

La prueba recupera el modelo guardado sin reentrenar. Sin `--run-id`, utiliza los tres IDs de la última comparación completa.
