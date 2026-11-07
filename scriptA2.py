import pandas as pd
import numpy as np

df = pd.read_csv("Steel_industry_data.csv")
x = df["Lagging_Current_Reactive.Power_kVarh"].values   # lleva un punto, no guion bajo
y = df["Usage_kWh"].values

# Vía 1: fórmulas del desarrollo a mano
n = len(x)
xbar, ybar = x.mean(), y.mean()
Sxx = np.sum(x**2) - np.sum(x)**2 / n
Sxy = np.sum(x * y) - np.sum(x) * np.sum(y) / n
b1_mano = Sxy / Sxx
b0_mano = ybar - b1_mano * xbar

# Vía 2: la misma herramienta de A.1.3 (lstsq), ahora con x = reactiva inductiva
X_lin = np.column_stack([np.ones(len(x)), x])
beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]

print("xbar =", xbar, " ybar =", ybar)
print("Sxx =", Sxx, " Sxy =", Sxy)
print("A mano: b0 =", b0_mano, " b1 =", b1_mano)
print("lstsq:  b0 =", beta_lin[0], " b1 =", beta_lin[1])
print("Diferencias:", abs(b0_mano - beta_lin[0]), abs(b1_mano - beta_lin[1]))
print("Coinciden:", np.allclose([b0_mano, b1_mano], beta_lin))