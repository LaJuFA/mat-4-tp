"""
ENUNCIADO
B.1 Varianza
    • Para el modelo de regresión simple de Usage_kWh sobre Lagging_Current_Reactive.Power_kVarh, verificar
    numéricamente la descomposición de la variabilidad y presentarla en una tabla:
        STC = SCR + SCE
        (total = explicada por la regresión + no explicada)
"""

"""
MI EXPLICACION
    • La varianza es un valor que representa cuanto varian los datos respecto a su promedio.
    Aparentemente hay como tres tipos de varianza diferentes con diferenetes ecuaciones
    El ejercicio pide que calcule el STC, SCR y SCE los meta en una tabla y lo muestre
    Voy a tener que usar pandas y alguna otra libreria que muestre cosas pero nidea
    Se
"""

import pandas as pd
import numpy as np

def get_average(list):
    return sum(list) / list.size

def calculate_s(first_list, second_list):
    first_average = get_average(first_list)
    second_average = get_average(second_list)
    aux_first = (first_list - first_average).to_list()
    aux_second = (second_list - second_average).to_list()
    return sum(map(lambda x, y: x * y, aux_first, aux_second))

def get_yi(x):
    return b0 + b1 * x

df = pd.read_csv("Steel_industry_data.csv")

df_relevant_data = df[["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh"]]

df_relevant_data_capped = df_relevant_data #No quiero usar un trillon de datos por ahora gracias (esto tenia un head en el commit anterior que hacia que agarre solo las primeras 5 filas)

only_kwh = pd.Series(df_relevant_data_capped["Usage_kWh"])

only_kvarh = pd.Series(df_relevant_data_capped["Lagging_Current_Reactive.Power_kVarh"])

#CALCULO DE STC

aux_stc = (np.power((only_kwh - get_average(only_kwh)), 2)).to_list()

stc = sum(aux_stc)

#CALCULO DE SCR

sxy = calculate_s(only_kvarh, only_kwh)

sxx = calculate_s(only_kvarh, only_kvarh)

b1 = sxy / sxx

b0 = get_average(only_kwh) - b1 * get_average(only_kvarh)

aux_yi = pd.Series(map(get_yi, only_kvarh))

aux_scr = (np.power((aux_yi - get_average(only_kwh)), 2)).to_list()

scr = sum(aux_scr)

#CALCULO SCE

aux_sce = (np.power((only_kwh - aux_yi), 2)).to_list()

sce = sum(aux_sce)

#Calculo de la estimacion de la varianza

o2 = sce / (only_kwh.size - 2)

#Piola

#Calculo del coeficiente de determinacion

r2 = 1 - sce / stc

#Tabla

d = {
    "STC": [stc],
    "SCR": [scr],
    "SCE": [sce],
    "Estimacion de la varianza": [o2],
    "Coeficiente de determinacion": [r2]
}

df_table = pd.DataFrame(d)

print(df_table)