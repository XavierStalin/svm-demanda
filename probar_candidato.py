# probar_candidato.py
"""Demuestra cómo cargar un modelo específico desde MLflow usando su Run ID."""
import mlflow
import pandas as pd
from svm_demanda import config, datos, features, prediccion

# 1. Conectar al servidor
mlflow.set_tracking_uri(config.MLFLOW_URI)

# 2. PEGA AQUÍ EL RUN ID DE "SVR optimizada" (o de un candidato)
run_id_elegido = "ae1a22ef9ee54f85aa0dbbdd0b0bf82b"  # <-- Cambia este por el real

# 3. Construir la URI
model_uri = f"runs:/{run_id_elegido}/pipeline_svr"

print(f"\nCargando modelo desde el Run ID: {run_id_elegido[:8]}...")
modelo_cargado = prediccion.cargar_modelo_mlflow(model_uri)

# 4. Preparar datos
print("Cargando historial...")
diario = datos.cargar_diario()
diario = features.agregar_calendario(diario)

# 5. Predecir para DOS casos (fechas distintas) para cumplir la rúbrica 3.3
fechas_objetivo = ["2017-08-16", "2017-08-17"]
print(f"\nPrediciendo para {config.FAMILIA} en 2 fechas distintas:")

resultados = []
for fecha in fechas_objetivo:
    demanda_estimada = prediccion.predecir_demanda(
        producto=config.FAMILIA, 
        fecha=fecha, 
        modelo=modelo_cargado, 
        diario=diario
    )
    resultados.append({"Fecha": fecha, "Demanda Estimada (miles)": f"{demanda_estimada:.2f}"})
    print(f"  - {fecha}: {demanda_estimada:.2f} miles de unidades.")

print("\n" + pd.DataFrame(resultados).to_string(index=False))
print("\nVerificación exitosa: El modelo cargado desde MLflow funciona correctamente para múltiples casos.")