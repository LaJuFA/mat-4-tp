from scipy import stats
from ejercicio_b2 import obtener_resultados_b2

# Se extraen los datos calculados en B2 (desactivando la ventana gráfica si solo se quiere ejecutar el test)
df, modelos, df_b2 = obtener_resultados_b2("Steel_industry_data.csv", mostrar_grafico=False)

var_elegida = "Lagging_Current_Reactive.Power_kVarh"
modelo_elegido = modelos[var_elegida]

beta_1 = modelo_elegido["beta_1"]
se_beta1 = modelo_elegido["se_beta1"]
n = modelo_elegido["n"]
alpha = 0.05
df_degrees = n - 2

T_stat = beta_1 / se_beta1
t_critico = stats.t.ppf(1 - alpha / 2, df=df_degrees)
p_valor = 2 * (1 - stats.t.cdf(abs(T_stat), df=df_degrees))

print("=== RESULTADOS B.3 (IMPORTANDO DESDE B2) ===")
print(f"Variable analizada:          {var_elegida}")
print(f"Pendiente estimada (beta_1): {beta_1:.6f}")
print(f"Error estándar de beta_1:    {se_beta1:.6f}")
print(f"Estadístico T calculado:     {T_stat:.4f}")
print(f"Valor crítico t(0.025, {df_degrees}): {t_critico:.4f}")
print(f"p-valor:                     {p_valor}")

if abs(T_stat) > t_critico:
    print("\nDecisión: Se RECHAZA la hipótesis nula H0.")
    print("Conclusión: La pendiente es significativamente distinta de cero; "
          "la potencia reactiva inductiva influye directamente en el consumo eléctrico.")
else:
    print("\nDecisión: NO se rechaza la hipótesis nula H0.")