# probar_candidato.py
"""
Demuestra cómo cargar cualquier modelo desde MLflow usando su Run ID 
y predecir con él. Cumple la rúbrica 3.3.
"""
import mlflow
import pandas as pd
from svm_demanda import config, datos, features, prediccion

# Conectar al servidor de MLflow
mlflow.set_tracking_uri(config.MLFLOW_URI)

def probar_modelo_desde_mlflow(run_id: str, fechas: list[str]):
    """
    Función que carga un modelo específico desde MLflow mediante su Run ID
    y predice la demanda para una lista de fechas.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Probando modelo con Run ID: {run_id[:8]}...")
    print(f"{'='*60}")
    
    # 1. Construir la URI y cargar el modelo
    model_uri = f"runs:/{run_id}/pipeline_svr"
    try:
        modelo = prediccion.cargar_modelo_mlflow(model_uri)
        print(f"✅ Modelo cargado exitosamente desde MLflow.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # 2. Preparar datos históricos
    diario = datos.cargar_diario()
    diario = features.agregar_calendario(diario)

    # 3. Predecir para cada fecha (los "dos casos" de la rúbrica)
    print(f"\n🎯 Predicciones para {config.FAMILIA}:\n")
    resultados = []
    for fecha in fechas:
        demanda = prediccion.predecir_demanda(
            producto=config.FAMILIA, 
            fecha=fecha, 
            modelo=modelo, 
            diario=diario
        )
        resultados.append({"Fecha": fecha, "Demanda Estimada (miles)": f"{demanda:.2f}"})
        print(f"  📅 {fecha} -> {demanda:.2f} miles de unidades.")

    print("\n📊 Resumen:")
    print(pd.DataFrame(resultados).to_string(index=False))


def main():
    # Fechas de prueba (dentro del periodo de prueba, con historial completo)
    fechas_demo = ["2016-09-13", "2016-09-14"]

    # ⚠️ IDs REALES de tu última ejecución (copiados de tu terminal)
    id_svr_inicial = "8070daa3a23d4c22be383dfa286cf664"
    id_svr_optimizada = "d6a01e7529f94dc18141f5bd91900272"

    print("🚀 INICIANDO DEMOSTRACIÓN DE MODELOS")
    
    # Caso 1: Probar el modelo BASE
    probar_modelo_desde_mlflow(id_svr_inicial, fechas_demo)
    
    # Caso 2: Probar el modelo OPTIMIZADO (solo cambiamos el ID)
    probar_modelo_desde_mlflow(id_svr_optimizada, fechas_demo)

    print("\n✨ Demostración completada. Se probó la misma función con dos modelos distintos.")

if __name__ == "__main__":
    main()