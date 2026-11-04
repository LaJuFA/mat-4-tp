"""
Trabajar con el modelo de regresión simple sobre Lagging_Current_Reactive.Power_kVarh y con el valor fijo x* = 20 kVarh.
    • Construir el intervalo de confianza del 95 % para la respuesta media, es decir para β₀ + β₁x*:
    β̂₀ + β̂₁x* ± t(α/2, n-2) · √( σ̂² ( 1/n + (x* - x̄)² / Sxx ) )
    • Construir el intervalo de predicción del 95 % para un valor futuro de Y:
    ŷ* ± t(α/2, n-2) · √( σ̂² ( 1 + 1/n + (x* - x̄)² / Sxx ) )
    • Explicar a qué se refiere cada uno de los dos intervalos.

"""

import pandas as pd
import numpy as np
import ejercicio_b1 as b1
import scipy.stats as stats

df = pd.read_csv("Steel_industry_data.csv")

x_average = df["Lagging_Current_Reactive.Power_kVarh"].mean()

x_value = 20

alpha = 0.05

n = df["Lagging_Current_Reactive.Power_kVarh"].size

t_crit = stats.t.ppf(1 - alpha / 2, n - 2)

sup = b1.get_b0() + b1.get_b1() * x_value + t_crit * np.sqrt(b1.get_o2() * (1 / n + (x_value - x_average) ** 2 / b1.get_sxx()))

inf = b1.get_b0() + b1.get_b1() * x_value - t_crit * np.sqrt(b1.get_o2() * (1 / n + (x_value - x_average) ** 2 / b1.get_sxx()))

prediction_sup = b1.get_b0() + b1.get_b1() * x_value + t_crit * np.sqrt(b1.get_o2() * (1 + 1 / n + (x_value - x_average) ** 2 / b1.get_sxx()))

prediction_inf = b1.get_b0() + b1.get_b1() * x_value - t_crit * np.sqrt(b1.get_o2() * (1 + 1 / n + (x_value - x_average) ** 2 / b1.get_sxx()))

print("Intervalo de confianza del 95%:", inf, "|", sup)
print("Intervalo de prediccion del 95%:", prediction_inf, "|", prediction_sup)

"""
    CONCLUSION

    Nuestro intervalo de confianza nos dio [40.01919611797702 ; 40.35715907667871].
    Es decir que si la potencia reactiva es de 20 kVarh el consumo electrico promedio para cada observacion 
    va a estar entre estos dos datos

    En cambio nuestro intervalo de prediccion nos dio [11.098456628771302 | 69.27789856588443]
    Es decir a futuro si la potencia reactiva es 20 kVarh el consumo electrico va a estar entre estos datos 
"""