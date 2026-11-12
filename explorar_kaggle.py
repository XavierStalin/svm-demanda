# explorar_kaggle.py
"""Exploración rápida de Store Sales para decidir el nivel de agregación."""
import matplotlib.pyplot as plt
import pandas as pd

from svm_demanda import config

train = pd.read_csv(
    config.RAW / "train.csv",
    usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
    parse_dates=["date"],
)

print("Filas:", len(train))
print("Fechas:", train["date"].min().date(), "->", train["date"].max().date())
print("Tiendas:", train["store_nbr"].nunique(), "| Familias:", train["family"].nunique())

# Total nacional por día (suma de todas las tiendas y familias)
nacional = train.groupby("date")["sales"].sum().sort_index()
print("Días con ventas cero (cierres):", int((nacional == 0).sum()))

# ¿Hay señal temporal? Autocorrelación de la serie nacional (sin días de cierre)
serie = nacional[nacional > 0]
print("corr con ayer:       ", round(serie.corr(serie.shift(1)), 3))
print("corr con hace 7 días:", round(serie.corr(serie.shift(7)), 3))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
nacional.plot(ax=axes[0], title="Ventas totales por día")
nacional.groupby(nacional.index.dayofweek).mean().plot.bar(
    ax=axes[1], title="Promedio por día de semana")
nacional.groupby(nacional.index.day).mean().plot.bar(
    ax=axes[2], title="Promedio por día del mes (efecto quincena)")
fig.tight_layout()
fig.savefig(config.RESULTADOS / "exploracion_kaggle.png", dpi=150)
plt.close(fig)
print("Gráfico guardado en resultados/exploracion_kaggle.png")