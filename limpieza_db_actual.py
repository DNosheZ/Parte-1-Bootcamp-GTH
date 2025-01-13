import pandas as pd

# Cargar el archivo
df1 = pd.read_csv('Data_empleados_actuales_2024.csv')

# Renombrar columnas usando un diccionario
df1 = df1.drop(columns=['Unnamed: 0'])
df1 = df1.drop(columns=['conciliacion'])
df1 = df1.drop(columns=['anos_en_puesto'])
df1 = df1.drop(columns=['id'])
df1 = df1.drop(columns=['sexo'])
df1['abandono'] = 'No'



# Mostrar las dimensiones del dataset
#print("Dimensiones del dataset:", df.shape)

# # Tipos de datos
print("\nTipos de datos por columna:")
print(df1.dtypes)

# # Valores faltantes

print("\nValores faltantes por columna:")
print(df1.isnull().sum())
# # Vista previa de los primeros registros
# print("\nPrimeros registros del dataset:")
# print(df.head())
# Exportar a un nuevo archivo
# df.to_csv('Data_empleados_actuales_2024_clean.csv', index=False)

df2 = pd.read_csv('Data_empleados_historicos_clean.csv')
print("\nValores faltantes por columna:")
print(df2.isnull().sum())

# if df1.equals(df2):
#     print("Las bases de datos son idénticas.")
# else:
#     print("Las bases de datos son diferentes.")
#     columnas_exclusivas_df1 = set(df1.columns) - set(df2.columns)
#     print("Columnas exclusivas de df1:", columnas_exclusivas_df1)

#     # Columnas en df2 pero no en df1
#     columnas_exclusivas_df2 = set(df2.columns) - set(df1.columns)
#     print("Columnas exclusivas de df2:", columnas_exclusivas_df2)
df1.to_csv('Data_empleados_actuales_2024_clean.csv', index=False)