import pandas as pd
import numpy as np

df = pd.read_csv("Steel_industry_data.csv")
print(df.columns)   # primero mirá cómo se llaman las columnas

x = df["NSM"].values          # hora del día, en segundos
y = df["Usage_kWh"].values    # consumo en cada intervalo

# armamos la tabla: una columna de unos, una con x, otra con x²
X = np.column_stack([np.ones(len(x)), x, x**2])

# calculamos a, b, c (en el código: el resultado es beta[0], beta[1], beta[2])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
print("a =", beta[0], " b =", beta[1], " c =", beta[2])
print("pico a las", -beta[1] / (2*beta[2]) / 3600, "horas")

X_lin = np.column_stack([np.ones(len(x)), x])
beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
print(beta_lin)