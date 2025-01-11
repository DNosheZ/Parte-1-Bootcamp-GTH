import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


# Cargar el archivo
df = pd.read_csv('Data_empleados_historicos_clean.csv')

# mean_distance = df.groupby('abandono')['distancia_casa'].mean()
# std_distance = df.groupby('abandono')['distancia_casa'].std()
# print("Promedio de distancia a casa por abandono:")
# print(mean_distance)
# print("\nDesviación estándar de distancia a casa por abandono:")
# print(std_distance,'\n')

# '''
# Si el promedio de distancia a casa es significativamente mayor para 
# quienes abandonaron, podría ser un factor importante.
# '''

# sns.boxplot(data=df, x='abandono', y='distancia_casa')
# plt.title("Distribución de Distancia a Casa por Estado de Abandono")
# plt.xlabel("Abandono")
# plt.ylabel("Distancia a Casa")
# plt.show()

# #Edad y abandono
# mean_edad = df.groupby('abandono')['edad'].mean()
# std_edad = df.groupby('abandono')['edad'].std()
# print("Promedio de edad por abandono:")
# print(mean_edad)
# print("\nDesviación estándar de edad por abandono:")
# print(std_edad,'\n')

# variables = list(set(['edad', 'viajes', 'departamento', 'distancia_casa', 'educacion', 
#                       'carrera', 'satisfaccion_entorno', 'implicacion', 
#                       'nivel_laboral', 'puesto', 'satisfaccion_trabajo', 'estado_civil', 
#                       'salario_mes', 'num_empresas_anteriores', 'mayor_edad', 
#                       'horas_extra', 'incremento_salario_porc', 'evaluacion', 
#                       'satisfaccion_companeros', 'nivel_acciones', 
#                       'anos_experiencia', 'num_formaciones_ult_ano', 'anos_compania', 
#                       'anos_desde_ult_promocion', 'anos_con_manager_actual']))

# df['abandono_bin'] = df['abandono'].map({'Yes': 1, 'No': 0})
# # Filtrar solo variables numéricas
# numeric_vars = [col for col in variables if df[col].dtype in ['int64', 'float64']]

# # Calcular correlación solo para variables numéricas
# for i in numeric_vars:
#     print(f"Correlacion {i} y abandono: {df['abandono_bin'].corr(df[i])}\n")

# categorical_vars = ['satisfaccion_entorno', 'implicacion', 'satisfaccion_trabajo', 
#                     'estado_civil', 'mayor_edad', 'evaluacion', 'satisfaccion_companeros', 
#                     'carrera', 'educacion', 'departamento', 'viajes']

# df_dummies = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# def cramers_v(confusion_matrix):
#     chi2 = stats.chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum()
#     return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

# # Calcular Cramér's V para cada variable categórica con abandono
# for var in categorical_vars:
#     confusion_matrix = pd.crosstab(df[var], df['abandono'])
#     v = cramers_v(confusion_matrix)
#     print(f"Cramér's V entre {var} y {v}x \n")


# #Comparativa abandono y nivel laboral

# # Promedio y desviación estándar del nivel laboral por abandono
# mean_level = df.groupby('abandono')['nivel_laboral'].mean()
# std_level = df.groupby('abandono')['nivel_laboral'].std()

# print("Promedio de nivel laboral por abandono:")
# print(mean_level)
# print("\nDesviación estándar de nivel laboral por abandono:")
# print(std_level)

# # Frecuencia de niveles laborales por abandono
# frecuencia_nivel = pd.crosstab(df['nivel_laboral'], df['abandono'])
# print("Frecuencia de niveles laborales por abandono:")
# print(frecuencia_nivel)


# # Gráfico de barras apilado
# frecuencia_nivel.plot(kind='bar', stacked=True, figsize=(8, 6))
# plt.title("Distribución de Niveles Laborales por Abandono")
# plt.xlabel("Nivel Laboral")
# plt.ylabel("Frecuencia")
# plt.legend(title="Abandono")
# plt.show()

#Boxplot
# sns.boxplot(data=df, x='abandono', y='distancia_casa')
# plt.title("Distribución de Lejania por Estado de Abandono")
# plt.xlabel("Abandono")
# plt.ylabel("Distancia de casa (km)")
# plt.show()

# # Promedio y desviación estándar del nivel laboral por abandono
# mean_level = df.groupby('abandono')['distancia_casa'].mean()
# std_level = df.groupby('abandono')['distancia_casa'].std()

# print("Promedio de Lejania por abandono:")
# print(mean_level)
# print("\nDesviación estándar de nivel laboral por abandono:")
# print(std_level)


sns.boxplot(data=df, x='abandono', y='salario_mes')
plt.title("Distribución de salario por Estado de Abandono")
plt.xlabel("Abandono")
plt.ylabel("Salario mensual ($)")
plt.show()