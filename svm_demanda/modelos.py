"""Definición, entrenamiento y optimización de los modelos SVR."""
from time import perf_counter

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Hiperparámetros de la búsqueda (idénticos al notebook para reproducibilidad).
# Si quieres probar otros valores (ej. epsilon, C más pequeños), cámbialos aquí.
PARAM_GRID = {
    "modelSVM__C": [0.1, 1.0, 10.0, 100.0],
    "modelSVM__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "modelSVM__epsilon": [0.1],
    "modelSVM__gamma": ["scale"],
}


def crear_pipeline() -> Pipeline:
    """Crea el pipeline base: escalado + SVR.
    
    El StandardScaler va DENTRO del pipeline para evitar fugas de datos:
    se ajusta solo con los datos de entrenamiento de cada pliegue.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("modelSVM", SVR()),
    ])


def entrenar_base(X_train, y_train) -> tuple[Pipeline, float]:
    """Entrena el modelo inicial (C=1, kernel=rbf por defecto en sklearn)."""
    tic = perf_counter()
    # clone() crea una copia fresca del pipeline sin tocar el original
    modelo = clone(crear_pipeline()).fit(X_train, y_train)
    tiempo = perf_counter() - tic
    return modelo, tiempo


def optimizar(X_train, y_train, pliegues) -> tuple[Pipeline, GridSearchCV, float]:
    """Ejecuta GridSearchCV sobre los pliegues temporales.
    
    Devuelve:
        - El mejor modelo encontrado (best_estimator_).
        - El objeto GridSearchCV completo (para acceder a cv_results_ y best_params_).
        - El tiempo total de búsqueda.
    """
    gs = GridSearchCV(
        crear_pipeline(),
        PARAM_GRID,
        scoring="neg_root_mean_squared_error",
        cv=pliegues,
        refit=True,             # Reentrena el mejor modelo con todo el train al final
        return_train_score=True,
        error_score="raise",
        verbose=1,              # 1 = barra de progreso/resumen; 0 = silencio; 2 = detalle
        n_jobs=1,               # 1 = secuencial (seguro en Windows); -1 = todos los núcleos
    )
    tic = perf_counter()
    gs.fit(X_train, y_train)
    tiempo = perf_counter() - tic
    
    return gs.best_estimator_, gs, tiempo