"""
B.5 Regresión múltiple
    • Estimar por mínimos cuadrados el modelo de regresión múltiple de Usage_kWh sobre las variables
    Lagging_Current_Reactive.Power_kVarh, Leading_Current_Reactive_Power_kVarh, Lagging_Current_Power_Factor,
    NSM, resolviendo el sistema de ecuaciones normales en forma matricial:
    β̂ = (XᵀX)⁻¹ XᵀY ŷ = β̂₀ + β̂₁x₁ + β̂₂x₂ + … + β̂x
"""

import pandas as pd
import numpy as np
import patsy

df = pd.read_csv("Steel_industry_data.csv")

df_relevant_data = df[["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh", "Lagging_Current_Power_Factor", "NSM"]]

df_relevant_data_capped = df_relevant_data

formula = (
    '1 + Q("Lagging_Current_Reactive.Power_kVarh")'
    ' + Q("Leading_Current_Reactive_Power_kVarh")'
    ' + Q("Lagging_Current_Power_Factor")'
    ' + Q("NSM")'
)

matrix_x = patsy.dmatrix(formula, df_relevant_data_capped, return_type='dataframe')

X = matrix_x.to_numpy()
Y = df_relevant_data_capped[["Usage_kWh"]].to_numpy()

# Transpuesta de X
XT = X.T

# Producto X transpuesta por X
XTX = XT @ X

# Inversa del producto X transpuesta por X
XTX_inv = np.linalg.inv(XTX)

# Producto X transpuesta por Y
XTY = XT @ Y

B = XTX_inv @ XTY