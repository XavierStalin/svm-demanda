# Rutas y constantes del proyecto.
from pathlib import Path

# --- Rutas (se calculan desde la ubicación de este archivo) ---
BASE = Path(__file__).resolve().parent.parent   # raíz: svm-demanda-retail/
RAW = BASE / "data" / "raw"                     # CSV original, no se toca
PROCESSED = BASE / "data" / "processed"         # datos listos para el modelo
RESULTADOS = BASE / "resultados"                # métricas, predicciones, gráficos

ARCHIVO_TRAIN = RAW / "train.csv"
FAMILIA = "GROCERY I"                          # la única familia que usaremos
EXPERIMENTO = "PracticaUnidad2_SVM_StoreSales"  # reemplaza la línea de EXPERIMENTO anterior

for carpeta in (RAW, PROCESSED, RESULTADOS):
    carpeta.mkdir(parents=True, exist_ok=True)

ARCHIVO_CSV = RAW / "supermarket_sales.csv"
URL_CSV = "https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv"

# --- Datos y features ---
VENTANA = 7                       # días de historia por ventana
VARIABLES_HISTORIA = [            # variables a las que se les hacen lags
    "demanda", "dia_semana", "mes", "es_fin_de_semana", "fecha_especial",
]
FECHAS_ESPECIALES = ["2019-01-01", "2019-02-14"]

# --- MLflow ---
MLFLOW_URI = "http://127.0.0.1:5001"
EXPERIMENTO = "PracticaUnidad2_SVM_Demanda"
MLFLOW_DB = BASE / "mlflow_demanda.db"
MLFLOW_ARTIFACTS = BASE / "mlartifacts_demanda"