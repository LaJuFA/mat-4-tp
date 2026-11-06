import pandas as pd
import numpy as np
from ejercicio_b2 import obtener_resultados_b2

def ejecutar_b5(csv_path="Steel_industry_data.csv"):
    # 1. Cargar el dataset
    df = pd.read_csv(csv_path)

    # 2. Definir las variables
    variables_predictoras = [
        "Lagging_Current_Reactive.Power_kVarh", 
        "Leading_Current_Reactive_Power_kVarh", 
        "Lagging_Current_Power_Factor", 
        "NSM"
    ]
    
    # Extraemos Y como matriz columna (n x 1)
    Y = df["Usage_kWh"].to_numpy().reshape(-1, 1)
    
    # Extraemos los datos de las variables predictoras (n x k)
    X_datos = df[variables_predictoras].to_numpy()
    n = len(Y)
    k = X_datos.shape[1] # k = 4
    
    # Armamos la matriz de diseño X agregando una columna de unos al principio para beta_0
    X = np.c_[np.ones(n), X_datos]

    # 3. Sistema de ecuaciones normales en forma matricial: β = (X^T * X)^-1 * X^T * Y
    XT = X.T                         # Transpuesta de X
    XTX = XT @ X                     # Producto matricial (X^T * X)
    XTX_inv = np.linalg.inv(XTX)     # Inversa de (X^T * X)
    XTY = XT @ Y                     # Producto matricial (X^T * Y)
    
    B = XTX_inv @ XTY                # Matriz de coeficientes Beta (beta_0, beta_1... beta_4)

    print("=== Forma matricial de la regresion multiple ===")
    print(B)

    # 4. Predicciones (Y_hat) y residuales
    Y_hat = X @ B
    residuos = Y - Y_hat

    # 5. Sumas de cuadrados del modelo múltiple
    y_bar = np.mean(Y)
    SCE_mult = np.sum(residuos ** 2)          # Suma de Cuadrados del Error
    STC_mult = np.sum((Y - y_bar) ** 2)       # Suma Total de Cuadrados

    # 6. R^2 ordinario y R^2 ajustado (usando la fórmula estadística estandarizada)
    R2_mult = 1 - (SCE_mult / STC_mult)
    R2_adj_mult = 1 - ((n - 1) / (n - k - 1)) * (1 - R2_mult)

    # 7. Obtener los datos del modelo simple B.2 para la tabla comparativa
    _, modelos_b2 = obtener_resultados_b2(csv_path)
    var_b2 = "Lagging_Current_Reactive.Power_kVarh"
    R2_b2 = modelos_b2[var_b2]["R2"]
    
    k_b2 = 1 # El modelo simple solo tiene 1 variable predictora
    R2_adj_b2 = 1 - ((n - 1) / (n - k_b2 - 1)) * (1 - R2_b2)

    # --- IMPRESIÓN DE RESULTADOS EN TERMINAL ---
    
    # Aplanamos la matriz de coeficientes B para iterarla fácilmente
    coefs = B.flatten()
    nombres_variables = ["Intercepto (beta_0)"] + variables_predictoras

    print("=== COEFICIENTES ESTIMADOS (B.5) ===")
    for nombre, coef in zip(nombres_variables, coefs):
        print(f"{nombre}: {coef:.6f}")

    print("\n=== ECUACIÓN DEL MODELO FINAL ===")
    # Empezamos el string con beta_0
    ecuacion = f"Usage_kWh = {coefs[0]:.4f}"
    # Iteramos sobre el resto de las variables (desde beta_1)
    for i, col in enumerate(variables_predictoras, start=1):
        signo = "+" if coefs[i] >= 0 else "-"
        ecuacion += f" {signo} {abs(coefs[i]):.6f} * ({col})"
    print(ecuacion)

    print("\n=== TABLA COMPARATIVA DE MODELOS ===")
    df_comparacion = pd.DataFrame([
        {
            "Modelo": f"Simple B.2 ({var_b2})",
            "k": k_b2,
            "R^2 Ordinario": f"{R2_b2:.6f}",
            "R^2 Ajustado": f"{R2_adj_b2:.6f}"
        },
        {
            "Modelo": "Múltiple B.5 (4 variables)",
            "k": k,
            "R^2 Ordinario": f"{R2_mult:.6f}",
            "R^2 Ajustado": f"{R2_adj_mult:.6f}"
        }
    ])
    print(df_comparacion.to_string(index=False))

if __name__ == "__main__":
    ejecutar_b5()