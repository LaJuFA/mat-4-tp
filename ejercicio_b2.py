import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt # Importación necesaria para graficar

def calcular_regresion_simple(X, Y, x_star=None, alpha=0.05):
    """
    Calcula los parámetros y métricas de regresión lineal simple para un par (X, Y).
    """
    # 1. Definimos el tamaño de la muestra y el valor t crítico para los intervalos (n-2 grados de libertad)
    n = len(Y)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    
    # 2. Calculamos las medias muestrales de X e Y
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    
    # 3. Sumas de cuadrados (Sxx, Syy) y de productos cruzados (Sxy)
    Sxx = np.sum((X - x_bar) ** 2)
    Syy = np.sum((Y - y_bar) ** 2)
    Sxy = np.sum((X - x_bar) * (Y - y_bar))
    
    # 4. Estimadores de mínimos cuadrados (Pendiente beta_1 y ordenada al origen beta_0)
    beta_1 = Sxy / Sxx
    beta_0 = y_bar - beta_1 * x_bar
    
    # 5. Cálculo de variabilidad y ajuste
    STC = Syy                          # Suma Total de Cuadrados
    SCE = Syy - (Sxy ** 2) / Sxx       # Suma de Cuadrados del Error (Residuales)
    sigma2 = SCE / (n - 2)             # Varianza residual (Cuadrado Medio del Error)
    R2 = 1 - (SCE / STC)               # Coeficiente de determinación (R-cuadrado)
    r = np.sign(beta_1) * np.sqrt(R2)  # Coeficiente de correlación de Pearson (r)
    
    # 6. Error estándar e Intervalo de Confianza para la pendiente (beta_1)
    se_beta1 = np.sqrt(sigma2 / Sxx)
    ic_beta1 = (beta_1 - t_crit * se_beta1, beta_1 + t_crit * se_beta1)
    
    # 7. Error estándar e Intervalo de Confianza para la ordenada al origen (beta_0)
    se_beta0 = np.sqrt(sigma2 * (1 / n + (x_bar ** 2) / Sxx))
    ic_beta0 = (beta_0 - t_crit * se_beta0, beta_0 + t_crit * se_beta0)
    
    # 8. Definición del punto x* a evaluar (si no se provee, usamos la media de X)
    if x_star is None:
        x_star = x_bar
        
    y_star = beta_0 + beta_1 * x_star  # Predicción puntual en x*
    
    # 9. Intervalo de Confianza para el valor Medio de Y en x* (ICM)
    se_icm = np.sqrt(sigma2 * (1 / n + ((x_star - x_bar) ** 2) / Sxx))
    icm = (y_star - t_crit * se_icm, y_star + t_crit * se_icm)
    
    # 10. Intervalo de Predicción para un valor Individual de Y en x* (IP)
    # Nota: El error estándar del IP suma '1' extra dentro del paréntesis
    se_ip = np.sqrt(sigma2 * (1 + 1 / n + ((x_star - x_bar) ** 2) / Sxx))
    ip = (y_star - t_crit * se_ip, y_star + t_crit * se_ip)
    
    # Empaquetamos y retornamos los resultados necesarios para la tabla
    return {
        "beta_1": beta_1, "beta_0": beta_0, "sigma2": sigma2, "R2": R2, "r": r,
        "ic_beta1": ic_beta1, "ic_beta0": ic_beta0, "icm": icm, "ip": ip
    }

def obtener_resultados_b2(csv_path="Steel_industry_data.csv"):
    """
    Carga el dataset, itera sobre las variables X solicitadas y genera el DataFrame
    final para imprimir en la terminal.
    """
    # 1. Cargamos el DataFrame y extraemos la variable dependiente Y
    df = pd.read_csv(csv_path)
    Y = df["Usage_kWh"].to_numpy()
    
    # 2. Definimos las variables independientes a analizar
    variables_x = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor",
        "NSM"
    ]
    
    # 3. Definimos en qué valor (x*) evaluaremos los intervalos ICM e IP.
    x_star_values = {
        "Lagging_Current_Reactive.Power_kVarh": 20.0,
        "Leading_Current_Reactive_Power_kVarh": df["Leading_Current_Reactive_Power_kVarh"].mean(),
        "Lagging_Current_Power_Factor": df["Lagging_Current_Power_Factor"].mean(),
        "NSM": df["NSM"].mean()
    }
    modelos = {}
    filas_tabla = []
    
    # 4. Iteramos sobre cada variable X
    for col in variables_x:
        X = df[col].to_numpy()
        x_star = x_star_values[col]
        
        # Llamamos a la función de regresión matemática
        res = calcular_regresion_simple(X, Y, x_star=x_star)
        modelos[col] = res
        
        # Formateamos los resultados para que se ajusten al cuadro B.2 (4 decimales)
        filas_tabla.append({
            "Xi": col,
            "y = beta_1*x + beta_0": f"y = {res['beta_1']:.4f}x + {res['beta_0']:.4f}",
            "sigma^2": f"{res['sigma2']:.4f}",
            "R^2": f"{res['R2']:.4f}",
            "r": f"{res['r']:.4f}", 
            "IC(beta_1)": f"[{res['ic_beta1'][0]:.4f}, {res['ic_beta1'][1]:.4f}]",
            "IC(beta_0)": f"[{res['ic_beta0'][0]:.4f}, {res['ic_beta0'][1]:.4f}]",
            "ICM(Y|x*)": f"[{res['icm'][0]:.4f}, {res['icm'][1]:.4f}]",
            "IP(Y|x*)": f"[{res['ip'][0]:.4f}, {res['ip'][1]:.4f}]"
        })
        
    # Convertimos la lista de diccionarios en un DataFrame de Pandas
    df_b2 = pd.DataFrame(filas_tabla)
    return df_b2, modelos

if __name__ == "__main__":
    csv_path = "Steel_industry_data.csv"
    
    # Desempaquetamos correctamente la tabla y el diccionario de modelos
    df_b2, modelos = obtener_resultados_b2(csv_path)
    
    print("=== CUADRO B.2 ===")
    print(df_b2.to_string(index=False))
    
    # --- LÓGICA DE GRAFICACIÓN ---
    # Seleccionamos la mejor variable (la de mayor R^2)
    mejor_variable = "Lagging_Current_Reactive.Power_kVarh"
    
    # Cargamos los datos originales para graficar los puntos
    df = pd.read_csv(csv_path)
    X_grafico = df[mejor_variable].to_numpy()
    Y_grafico = df["Usage_kWh"].to_numpy()
    
    # Extraemos los coeficientes ajustados desde nuestro diccionario de modelos
    beta_0 = modelos[mejor_variable]["beta_0"]
    beta_1 = modelos[mejor_variable]["beta_1"]
    
    # Generamos los puntos para la recta (desde el mínimo al máximo valor de X)
    x_linea = np.linspace(X_grafico.min(), X_grafico.max(), 100)
    y_linea = beta_0 + beta_1 * x_linea
    
    # Configuramos el gráfico
    plt.figure(figsize=(10, 6))
    
    # Diagrama de dispersión (puntos de los datos reales)
    plt.scatter(X_grafico, Y_grafico, alpha=0.3, color='blue', label='Datos observados')
    
    # Recta ajustada superpuesta
    plt.plot(x_linea, y_linea, color='red', linewidth=2, label=f'Recta ajustada: y = {beta_1:.4f}x + {beta_0:.4f}')
    
    # Etiquetas y diseño
    plt.xlabel(mejor_variable)
    plt.ylabel("Usage_kWh")
    plt.title(f"Diagrama de Dispersión y Regresión Lineal\nUsage_kWh vs {mejor_variable}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mostramos la ventana con el gráfico
    plt.show()