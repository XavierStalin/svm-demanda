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
    print(f" Probando modelo con Run ID: {run_id[:8]}...")
    print(f"{'='*60}")
    
    # 1. Construir la URI y cargar el modelo
    model_uri = f"runs:/{run_id}/pipeline_svr"
    try:
        modelo = prediccion.cargar_modelo_mlflow(model_uri)
        print(f"Modelo cargado exitosamente desde MLflow.")
    except Exception as e:
        print(f" Error al cargar el modelo: {e}")
        return

    # 2. Preparar datos históricos
    diario = datos.cargar_diario()
    diario = features.agregar_calendario(diario)

    # 3. Predecir para cada fecha (los "dos casos" de la rúbrica)
    print(f"\n Predicciones para {config.FAMILIA}:\n")
    resultados = []
    for fecha in fechas:
        demanda = prediccion.predecir_demanda(
            producto=config.FAMILIA, 
            fecha=fecha, 
            modelo=modelo, 
            diario=diario
        )
        resultados.append({"Fecha": fecha, "Demanda Estimada (miles)": f"{demanda:.2f}"})
        print(f"   {fecha} -> {demanda:.2f} miles de unidades.")

    print("\n Resumen:")
    print(pd.DataFrame(resultados).to_string(index=False))


def main():
    fechas_demo = ["2016-09-13", "2016-09-14"]
    
    print("="*70)
    print(" MODELOS DISPONIBLES EN MLFLOW")
    print("="*70)
    print("1. SVR Inicial      : 8070daa3a23d4c22be383dfa286cf664")
    print("2. SVR Optimizada   : d6a01e7529f94dc18141f5bd91900272")
    print("3. Candidato 05     : 8e98d94ae43041cea55b6b9e6781b151")
    print("="*70)
    
    # Aquí el profesor puede decirte un número o un ID
    eleccion = input("\n Escribe el ID que quieras probar (o presiona Enter para usar el Optimizado por defecto): ").strip()
    
    if eleccion == "1":
        id_a_probar = "8070daa3a23d4c22be383dfa286cf664"
    elif eleccion == "2" or eleccion == "":
        id_a_probar = "d6a01e7529f94dc18141f5bd91900272"
    elif eleccion == "3":
        id_a_probar = "8e98d94ae43041cea55b6b9e6781b151"
    else:
        id_a_probar = eleccion # Por si pega el ID completo manualmente

    print(f"\n Modelo seleccionado. Iniciando predicción...\n")
    
    # Ejecutamos la función con el ID que el profesor eligió
    probar_modelo_desde_mlflow(id_a_probar, fechas_demo)

if __name__ == "__main__":
    main()