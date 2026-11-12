# SVM Demanda Retail (Store Sales - Kaggle)

Predicción de demanda diaria por familia de producto (GROCERY I) con SVR
(base vs. optimizada con GridSearchCV) y seguimiento en MLflow.

## Requisitos

1. Python 3.12 con conda o venv.
2. Instalar dependencias:
   pip install -r requirements.txt
3. **Descargar los datos de Kaggle** (ver sección Datos).

## Datos

El dataset no se incluye en el repositorio por el límite de 100 MB de GitHub.

1. Crea cuenta en https://www.kaggle.com
2. Entra a la competencia y **acepta las reglas**:
   https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules
3. Descarga el ZIP y extrae estos CSV en la carpeta `data/raw/`:
   - train.csv  (obligatorio)
   - stores.csv, oil.csv, holidays_events.csv, transactions.csv, sample_submission.csv (opcionales para este flujo)

## Ejecutar

python main.py

Esto:
- Carga train.csv y filtra la familia GROCERY I
- Construye features (7 lags x 5 variables = 35 columnas)
- Entrena SVR base y optimizada (GridSearchCV, 16 candidatos x 3 pliegues)
- Genera gráficos en resultados/
- Levanta MLflow y registra 2 runs principales + 16 candidatas
- Verifica la carga del modelo desde MLflow

## Estructura

- svm_demanda/ : codigo modular (config, datos, features, particion,
  modelos, evaluacion, interpretacion, tracking, prediccion)
- main.py      : orquestador (un solo comando)
- data/raw/    : CSV de Kaggle (no versionado)
- resultados/  : metricas, predicciones y graficos .png
- notebooks/   : exploracion
- probar_candidato.py : carga un modelo de MLflow por Run ID y predice

## Resultados esperados

| Modelo | MAE | RMSE |
|---|---|---|
| SVR inicial (C=1, rbf) | 47.99 | 66.45 |
| SVR optimizada (C=1, linear) | 24.74 | 46.93 |
| Persistencia | 45.97 | 67.28 |

Reduccion de MAE: 48.45% | Reduccion de RMSE: 29.38%

## Graficos

![Comparacion](resultados/comparacion.png)
![Importancia](resultados/importancia_caracteristicas.png)
![Optimizacion](resultados/optimizacion.png)