# Guion de exposición: SVM inicial y optimizada

**Integrantes:** Daniel, Xavier, Angélica y José. **Duración aproximada:** 6–8 minutos, incluida la demostración de MLflow. Cada integrante tiene una intervención continua. Mostrar los resultados ya preparados en `Resultados_SVM_Sin_Codigo.html` y cambiar a MLflow durante la intervención de José.

| Integrante | Tema | Apoyo visual |
|---|---|---|
| Daniel | Objetivo, datos y evaluación | Gráfico de demanda y separación temporal |
| Xavier | Parte técnica y optimización | Mapa de GridSearchCV |
| Angélica | Resultados e interpretación | Comparación de errores, predicciones e importancia |
| José | Demostración y conclusión | Dos casos preparados y MLflow |

## Daniel — introducción y contexto

*Mostrar el gráfico de demanda y la separación temporal mientras explica, sin interrumpir el discurso.*

> Buenos días. Nuestro proyecto compara una SVM inicial con una optimizada para predecir la demanda diaria de la familia GROCERY I. Trabajamos con las ventas agregadas de 54 tiendas, expresadas en miles de unidades. El gráfico muestra variaciones y patrones semanales que buscamos aprovechar mediante los siete días anteriores. Para evaluar el resultado respetamos el orden de las fechas: utilizamos 1.344 días para entrenar y reservamos los 337 más recientes para la prueba. El escenario consiste en predecir un día con la historia real anterior disponible. Xavier explicará cómo seleccionamos la configuración optimizada.

## Xavier — optimización y selección del modelo

*Mostrar el mapa de optimización y señalar la celda del kernel lineal con C = 1.*

> Como estimamos una cantidad numérica, utilizamos SVR, la variante de regresión de las máquinas de vectores de soporte. La configuración inicial tiene kernel RBF y C igual a uno. Para optimizarla usamos GridSearchCV con cuatro kernels y cuatro valores de C: 16 combinaciones evaluadas en tres pliegues temporales, es decir, 48 ajustes de validación. C controla la penalización del error y el kernel define la relación que aprende el modelo. El escalado se ajusta dentro de cada entrenamiento para evitar usar información de validación. El mapa muestra que ganó el kernel lineal con C igual a uno, con un RMSE medio de validación de 34,58 miles de unidades. Elegimos esa configuración antes de revisar la prueba final. Angélica presentará sus resultados.

## Angélica — resultados y gráficos

*Mostrar la comparación de errores y predicciones; después, el gráfico de importancia. La ampliación de 60 días queda como apoyo opcional.*

> En los mismos 337 días de prueba, el MAE pasó de 47,99 a 24,74 y el RMSE de 66,45 a 46,93 miles de unidades. Esto representa reducciones del 48,45 y del 29,38 por ciento, respectivamente. El MAE resume el error absoluto promedio y el RMSE penaliza más los errores grandes. La optimizada también superó a la persistencia, que estima para hoy lo vendido ayer y obtuvo un RMSE de 67,28. En las curvas observamos que el modelo optimizado acompaña mejor varios cambios semanales, aunque todavía presenta diferencias frente a las ventas reales. El gráfico de importancia indica que la demanda de ayer es la entrada más relevante, seguida de variables de calendario. José mostrará ahora los resultados guardados en MLflow.

## José — demostración de MLflow y cierre

*Mostrar brevemente los dos casos preparados. Cambiar a MLflow: comparación de las dos ejecuciones, candidatos y detalle de la optimizada con un gráfico y el modelo guardado.*

> Estos dos casos muestran estimaciones para fechas de prueba con ventas reales conocidas. La conclusión se apoya en todo el período evaluado. En MLflow encontramos las dos ejecuciones principales: SVM inicial y SVM optimizada. La comparación conserva los mismos errores que acabamos de presentar y permite revisar sus parámetros. También se registraron las 16 configuraciones de la búsqueda con sus errores de validación. En los artefactos podemos abrir los gráficos, las tablas y las predicciones; además, quedaron guardados los dos modelos finales, cuya recuperación se verificó. Concluimos que, para GROCERY I y el escenario evaluado, la optimización redujo el MAE un 48,45 por ciento y el RMSE un 29,38 por ciento. MLflow permite conservar y revisar la evidencia de esta mejora. Gracias.

## Preparación breve — no se lee al público

- Abrir la vista sin código y dejar MLflow funcionando en **http://127.0.0.1:5002**, en **Model training → Training runs**.
- Usar el experimento **Comparacion_SVR_GROCERY_I_Cuaderno** y el grupo **20261112_121650_70da61**. Preparar **Compare** con las dos ejecuciones principales y las métricas `test_mae` y `test_rmse` visibles.
- Dejar otra pestaña con las 16 candidatas y otra con el detalle de la optimizada, sus artefactos y el modelo `pipeline_svr`. José solo cambia entre las pestañas ya preparadas.
- Xavier ayuda a José a preparar MLflow y responde las preguntas más técnicas sobre kernels, C y validación. Los porcentajes expresan reducción del error; las cantidades están en miles de unidades.

**Referencia rápida:** MAE inicial/optimizada: **47,99 / 24,74**. RMSE inicial/optimizada: **66,45 / 46,93**. Ganador: **lineal, C = 1**. Se conservan las limitaciones del cuaderno: cuatro fechas ausentes imputadas con cero y evaluación de un día con historia real disponible.
