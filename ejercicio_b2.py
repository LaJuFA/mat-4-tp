import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calcular_regresion_simple(X, Y, x_star=None, alpha=0.05):
    """
    Calcula los parámetros y métricas de regresión lineal simple para un par (X, Y).
    """
    n = len(Y)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    
    Sxx = np.sum((X - x_bar) ** 2)
    Syy = np.sum((Y - y_bar) ** 2)
    Sxy = np.sum((X - x_bar) * (Y - y_bar))
    
    beta_1 = Sxy / Sxx
    beta_0 = y_bar - beta_1 * x_bar
    
    STC = Syy
    SCE = Syy - (Sxy ** 2) / Sxx
    sigma2 = SCE / (n - 2)
    R2 = 1 - (SCE / STC)
    
    # Intervalos de confianza para los coeficientes
    se_beta1 = np.sqrt(sigma2 / Sxx)
    ic_beta1 = (beta_1 - t_crit * se_beta1, beta_1 + t_crit * se_beta1)
    
    se_beta0 = np.sqrt(sigma2 * (1 / n + (x_bar ** 2) / Sxx))
    ic_beta0 = (beta_0 - t_crit * se_beta0, beta_0 + t_crit * se_beta0)
    
    # Intervalos para respuesta media e individual en x*
    if x_star is None:
        x_star = x_bar
        
    y_star = beta_0 + beta_1 * x_star
    se_icm = np.sqrt(sigma2 * (1 / n + ((x_star - x_bar) ** 2) / Sxx))
    icm = (y_star - t_crit * se_icm, y_star + t_crit * se_icm)
    
    se_ip = np.sqrt(sigma2 * (1 + 1 / n + ((x_star - x_bar) ** 2) / Sxx))
    ip = (y_star - t_crit * se_ip, y_star + t_crit * se_ip)
    
    return {
        "n": n,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "Sxx": Sxx,
        "Syy": Syy,
        "Sxy": Sxy,
        "beta_1": beta_1,
        "beta_0": beta_0,
        "SCE": SCE,
        "STC": STC,
        "sigma2": sigma2,
        "R2": R2,
        "se_beta1": se_beta1,
        "ic_beta1": ic_beta1,
        "se_beta0": se_beta0,
        "ic_beta0": ic_beta0,
        "x_star": x_star,
        "y_star": y_star,
        "icm": icm,
        "ip": ip
    }

def obtener_resultados_b2(csv_path="Steel_industry_data.csv", mostrar_grafico=True):
    """
    Carga el dataset, calcula las regresiones B.2, muestra el gráfico de dispersión
    y retorna el DataFrame B.2 junto con el diccionario de modelos.
    """
    df = pd.read_csv(csv_path)
    Y = df["Usage_kWh"].to_numpy()
    
    variables_x = [
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor",
        "NSM"
    ]
    
    x_star_values = {
        "Lagging_Current_Reactive.Power_kVarh": 20.0,
        "Leading_Current_Reactive_Power_kVarh": df["Leading_Current_Reactive_Power_kVarh"].mean(),
        "Lagging_Current_Power_Factor": df["Lagging_Current_Power_Factor"].mean(),
        "NSM": df["NSM"].mean()
    }
    
    modelos = {}
    filas_tabla = []
    
    for col in variables_x:
        X = df[col].to_numpy()
        x_star = x_star_values[col]
        
        res = calcular_regresion_simple(X, Y, x_star=x_star)
        modelos[col] = res
        
        filas_tabla.append({
            "Xi": col,
            "y = beta_1*x + beta_0": f"y = {res['beta_1']:.4f}x + {res['beta_0']:.4f}",
            "sigma^2": round(res["sigma2"], 4),
            "R^2": round(res["R2"], 4),
            "IC(beta_1)": f"[{res['ic_beta1'][0]:.4f}, {res['ic_beta1'][1]:.4f}]",
            "IC(beta_0)": f"[{res['ic_beta0'][0]:.4f}, {res['ic_beta0'][1]:.4f}]",
            "ICM(Y|x*)": f"[{res['icm'][0]:.4f}, {res['icm'][1]:.4f}]",
            "IP(Y|x*)": f"[{res['ip'][0]:.4f}, {res['ip'][1]:.4f}]"
        })
        
    df_b2 = pd.DataFrame(filas_tabla)
    
    # Graficación de la mejor variable
    if mostrar_grafico:
        mejor_variable = "Lagging_Current_Reactive.Power_kVarh"
        X_best = df[mejor_variable].to_numpy()
        
        b1_b = modelos[mejor_variable]["beta_1"]
        b0_b = modelos[mejor_variable]["beta_0"]
        
        x_linea = np.linspace(X_best.min(), X_best.max(), 500)
        y_linea = b0_b + b1_b * x_linea
        
        plt.figure(figsize=(9, 5))
        plt.scatter(X_best, Y, alpha=0.15, color='blue', label='Observaciones')
        plt.plot(x_linea, y_linea, color='red', linewidth=2, label=f'Recta: y = {b1_b:.4f}x + {b0_b:.4f}')
        plt.xlabel(mejor_variable)
        plt.ylabel("Usage_kWh")
        plt.title(f"Diagrama de Dispersión y Recta Ajustada: Usage_kWh vs {mejor_variable}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return df, modelos, df_b2

if __name__ == "__main__":
    _, _, df_b2 = obtener_resultados_b2(mostrar_grafico=True)
    print("=== CUADRO B.2 ===")
    print(df_b2.to_string(index=False))