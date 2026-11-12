"""Ejecución completa del proyecto: de los datos al registro en MLflow."""
import pandas as pd

from svm_demanda import (config, datos, features, particion,
                         modelos, evaluacion, tracking, prediccion, interpretacion)


def main():
    print("1. Preparando datos...")
    diario = datos.cargar_diario()
    diario = features.agregar_calendario(diario)
    tabla, columnas = features.construir_lags(diario, config.VENTANA)
    print(f"   -> Días: {len(diario)} | Ventanas: {len(tabla)} | Features: {len(columnas)}")

    print("2. Dividiendo datos (Train/Test)...")
    (X_train, y_train, X_test, y_test, pliegues, fecha_prueba) = particion.dividir_y_pliegues(tabla, columnas)
    print(f"   -> Prueba desde: {fecha_prueba.date()} | Pliegues: {len(pliegues)}")

    print("3. Entrenando modelos...")
    modelo_base, tiempo_base = modelos.entrenar_base(X_train, y_train)
    print("   -> Optimizando con GridSearchCV (puede tardar unos segundos)...")
    modelo_optimizado, gs, tiempo_busqueda = modelos.optimizar(X_train, y_train, pliegues)
    print(f"   -> Mejores parámetros: {gs.best_params_}")

    print("4. Evaluando y generando gráficos...")
    comparacion = evaluacion.evaluar_modelos(modelo_base, modelo_optimizado, X_test, y_test, tiempo_base, gs.refit_time_)
    print(comparacion.round(4))
    evaluacion.imprimir_mejoras(comparacion)

    prueba = tabla.loc[tabla["fecha"] >= fecha_prueba].reset_index(drop=True)
    predicciones = evaluacion.generar_predicciones(prueba, modelo_base, modelo_optimizado, X_test)
    resultados_cv = evaluacion.preparar_gridsearch(gs)
    
    categorias = sorted(tabla["producto"].unique())
    evaluacion.graficar_comparacion(comparacion, predicciones, categorias[0])  # <-- AQUÍ EL CAMBIO
    evaluacion.graficar_optimizacion(resultados_cv)
    
    # NUEVO: Gráfico de interpretación para cumplir la rúbrica 3.2
    interpretacion.graficar_importancia(modelo_optimizado, X_test, y_test)

    print("5. Registrando comparación en MLflow...")
    grupo = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    uris = tracking.registrar_comparacion(
        modelos={"SVR inicial": modelo_base, "SVR optimizada": modelo_optimizado},
        comparacion=comparacion, gs=gs, X_train=X_train, X_test=X_test,
        fecha_prueba=fecha_prueba, tiempo_busqueda=tiempo_busqueda, grupo=grupo
    )
    print("   -> Registrando 16 candidatos del GridSearchCV...")
    tracking.registrar_candidatos_gridsearch(gs, grupo)
    print(f"   -> Abre {config.MLFLOW_URI} para ver las ejecuciones.")

    print("6. Demostración operativa (2 casos/fechas distintas para cumplir rúbrica 3.3)...")
    # Usamos 2 fechas distintas como los "dos casos" requeridos
    fechas_demo = [fecha_prueba.date(), (fecha_prueba + pd.Timedelta(days=1)).date()]
    demo = []
    for f in fechas_demo:
        demo.append({
            "producto": config.FAMILIA,
            "fecha": f,
            "SVR_inicial": prediccion.predecir_demanda(config.FAMILIA, f, modelo_base, diario),
            "SVR_optimizada": prediccion.predecir_demanda(config.FAMILIA, f, modelo_optimizado, diario)
        })
    print(pd.DataFrame(demo))
    
    print("7. Verificando carga del modelo desde MLflow...")
    prediccion.verificar_carga(uris["SVR optimizada"], modelo_optimizado, X_test)
    print("\n¡Proceso completado con éxito! 🎉")


if __name__ == "__main__":
    main()