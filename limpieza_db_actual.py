import pandas as pd

# Cargar el archivo
df = pd.read_csv('Data_empleados_actuales_2024.csv')

# Renombrar columnas usando un diccionario
df = df.drop(columns=['Unnamed: 0'])


# Mostrar las dimensiones del dataset
print("Dimensiones del dataset:", df.shape)

# # Tipos de datos
print("\nTipos de datos por columna:")
print(df.dtypes)

# # Valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# # Vista previa de los primeros registros
print("\nPrimeros registros del dataset:")
print(df.head())
# Exportar a un nuevo archivo
# df.to_csv('Data_empleados_actuales_2024_clean.csv', index=False)
