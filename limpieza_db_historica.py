import pandas as pd

# Cargar el archivo
df = pd.read_csv('Data_empleados_historico.csv', index_col='id', na_values='#N/D')

# Mostrar las dimensiones del dataset
# print("Dimensiones del dataset:", df.shape)

# # Tipos de datos
# print("\nTipos de datos por columna:")
# print(df.dtypes)

# # Valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# # Vista previa de los primeros registros
# print("\nPrimeros registros del dataset:")
# print(df.head())
df = df.drop(columns=['Unnamed: 0'])#parece ser un índice duplicado generado al exportar el archivo, se elimina

df = df.drop(columns=['conciliacion'])#eliminaremos esta columna por recomendacion

df = df.drop(columns=['anos_en_puesto'])#en vista que faltan mas del 80% de registros de esta columna, la eliminaremos


print("\nPrimeros registros del dataset:")
print(df.head())

print(f"Filas duplicadas: {df.duplicated().sum()}")


# print('Analisis de correlacion para determinar educacion\n')
# Mapear valores educativos a números


# Correlación entre educación y nivel_laboral/salario_mes
# correlacion_nivel = df['educacion_num'].corr(df['nivel_laboral'])
# correlacion_salario = df['educacion_num'].corr(df['salario_mes'])

# print(f"Correlación entre educación y nivel laboral: {correlacion_nivel}")
# print(f"Correlación entre educación y salario mensual: {correlacion_salario}")
# # Relación entre educación y puesto
# relacion_puesto = df.groupby('puesto')['educacion_num'].mean()
# print("Promedio de educación por puesto:")
# print(relacion_puesto)
import seaborn as sns
import matplotlib.pyplot as plt

# sns.scatterplot(data=df, x='educacion_num', y='salario_mes')
# plt.title("Relación entre Educación y Salario Mensual")
# plt.xlabel("Nivel de Educación (Numérico)")
# plt.ylabel("Salario Mensual")
# plt.show()
# sns.boxplot(data=df, x='puesto', y='educacion_num')
# plt.title("Distribución de Educación por Puesto")
# plt.xlabel("Puesto")
# plt.ylabel("Nivel de Educación (Numérico)")
# plt.xticks(rotation=45)  # Rotar las etiquetas para mejor visibilidad
# plt.show()
# Filtrar filas sin valores faltantes
# df_completo = df.dropna(subset=['educacion_num'])

# Correlación en las filas completas
# correlacion_nivel = df_completo['educacion_num'].corr(df_completo['nivel_laboral'])
# correlacion_salario = df_completo['educacion_num'].corr(df_completo['salario_mes'])

# print(f"Correlación en filas completas (nivel laboral): {correlacion_nivel}")
# print(f"Correlación en filas completas (salario mensual): {correlacion_salario}")
educacion_mapping = {'Primaria': 1, 'Secundaria': 2, 'Universitaria': 3, 'Master': 4}
df['educacion_num'] = df['educacion'].map(educacion_mapping)
# Rellenar los valores nulos en educacion_num basados en el promedio por puesto
df['educacion_num'] = df.groupby('puesto')['educacion_num'].transform(
    lambda x: x.fillna(x.mean())
)

# Rellenar los valores restantes con la moda global
df['educacion_num'] = df['educacion_num'].fillna(df['educacion_num'].mode()[0])

# Convertir educacion_num a enteros después de la imputación
df['educacion_num'] = df['educacion_num'].round().astype(int)

# Mapeo inverso para convertir de números a categorías
reverse_mapping = {1: 'Primaria', 2: 'Secundaria', 3: 'Universitaria', 4: 'Master'}
df['educacion'] = df['educacion_num'].map(reverse_mapping)

# Validar que no hay valores nulos en educacion
print("Valores faltantes en educacion después de la conversión:", df['educacion'].isnull().sum())
df = df.drop(columns=['educacion_num'])



#Analisis y limpieza de implicacion

# sns.countplot(data=df, x='implicacion', order=df['implicacion'].value_counts().index)
# plt.title("Distribución de Implicación")
# plt.xlabel("Niveles de Implicación")
# plt.ylabel("Frecuencia")
# plt.show()

# implicacion_abandono = df.groupby(['implicacion', 'abandono']).size().unstack()
# implicacion_abandono.plot(kind='bar', stacked=True)
# plt.title("Implicación por estado de abandono")
# plt.xlabel("Nivel de Implicación")
# plt.ylabel("Frecuencia")
# plt.show()

# implicacion_departamento = df.groupby('departamento')['implicacion'].value_counts(normalize=True).unstack()
# implicacion_departamento.plot(kind='bar', stacked=True, figsize=(8, 6))
# plt.title("Distribución de Implicación por Departamento")
# plt.xlabel("Departamento")
# plt.ylabel("Proporción")
# plt.legend(title="Nivel de Implicación")
# plt.show()

# sns.boxplot(data=df, x='implicacion', y='satisfaccion_trabajo')
# plt.title("Relación entre Implicación y Satisfacción Laboral")
# plt.xlabel("Nivel de Implicación")
# plt.ylabel("Satisfacción en el Trabajo")
# plt.show()

implicacion_mapping = {'Baja': 1, 'Media': 2, 'Alta': 3, 'Muy_Alta': 4}
satifaccion_mapping = {'Baja': 1, 'Media': 2, 'Alta': 3, 'Muy_Alta': 4}
df['implicacion_num'] = df['implicacion'].map(implicacion_mapping)
df['satisfaccion_num'] = df['satisfaccion_trabajo'].map(satifaccion_mapping)
correlacion = df['implicacion_num'].corr(df['satisfaccion_num'])
print(f"Correlación entre implicación y satisfacción laboral: {correlacion}")


# Calcular la mediana de implicacion_num para cada nivel de satisfaccion_num
mediana_implicacion_por_satisfaccion = df.groupby('satisfaccion_num')['implicacion_num'].median()


# Calcular la mediana de satisfaccion_num para cada nivel de implicacion_num
mediana_satisfaccion_por_implicacion = df.groupby('implicacion_num')['satisfaccion_num'].median()

# Rellenar implicacion_num según satisfaccion_num
df.loc[df['implicacion_num'].isnull() & df['satisfaccion_num'].notnull(), 'implicacion_num'] = df['satisfaccion_num'].map(mediana_implicacion_por_satisfaccion)

# Rellenar satisfaccion_num según implicacion_num
df.loc[df['satisfaccion_num'].isnull() & df['implicacion_num'].notnull(), 'satisfaccion_num'] = df['implicacion_num'].map(mediana_satisfaccion_por_implicacion)

# Rellenar casos donde ambas son nulas con medianas globales
df.loc[df['implicacion_num'].isnull() & df['satisfaccion_num'].isnull(), 'implicacion_num'] = df['implicacion_num'].median()
df.loc[df['implicacion_num'].isnull() & df['satisfaccion_num'].isnull(), 'satisfaccion_num'] = df['satisfaccion_num'].median()



# Mapeo inverso a las columnas categóricas originales
reverse_implicacion_mapping = {1: 'Baja', 2: 'Media', 3: 'Alta', 4: 'Muy_Alta'}
reverse_satisfaccion_mapping = {1: 'Baja', 2: 'Media', 3: 'Alta', 4: 'Muy_Alta'}

df['implicacion'] = df['implicacion_num'].map(reverse_implicacion_mapping)
df['satisfaccion_trabajo'] = df['satisfaccion_num'].map(reverse_satisfaccion_mapping)

# Validar que no hay valores faltantes
print("Valores faltantes en implicacion:", df['implicacion'].isnull().sum())
print("Valores faltantes en satisfaccion_trabajo:", df['satisfaccion_trabajo'].isnull().sum())
# Eliminar columnas numéricas si ya no son necesarias
df = df.drop(columns=[ 'implicacion_num', 'satisfaccion_num'])

#Analisis correlacion satisfaccion_trabajo, satisfaccion_companeros, satisfaccion_entorno, nivel_laboral y salario_mes

sat_compas_mapping = {'Baja': 1, 'Media': 2, 'Alta': 3, 'Muy_Alta': 4}
satifaccion_mapping = {'Baja': 1, 'Media': 2, 'Alta': 3, 'Muy_Alta': 4}
df['sat_compas_num'] = df['satisfaccion_companeros'].map(sat_compas_mapping)
df['satisfaccion_num'] = df['satisfaccion_trabajo'].map(satifaccion_mapping)
correlacion = df['satisfaccion_num'].corr(df['sat_compas_num'])
print(f"Correlación entre satisfaccion compañeros y satisfacción laboral: {correlacion}")

#Correlacion satisfaccion entorno
sat_entorno_mapping = {'Baja': 1, 'Media': 2, 'Alta': 3, 'Muy_Alta': 4}
df['sat_entorno_num'] = df['satisfaccion_entorno'].map(sat_entorno_mapping)
correlacion = df['satisfaccion_num'].corr(df['sat_entorno_num'])
print(f"Correlación entre satisfaccion entorno y satisfacción laboral: {correlacion}")

#Correlacion nivel laboral
correlacion = df['satisfaccion_num'].corr(df['nivel_laboral'])
print(f"Correlación entre nivel laboral y satisfacción laboral: {correlacion}")

#Correlacion salario mes

correlacion = df['satisfaccion_num'].corr(df['salario_mes'])
print(f"Correlación entre salario mensual y satisfacción laboral: {correlacion}")


df['satisfaccion_trabajo'] = df['satisfaccion_num'].map(reverse_satisfaccion_mapping)
# Crear una nueva categoría para valores faltantes
df['satisfaccion_trabajo'] = df['satisfaccion_trabajo'].fillna('Desconocido')

df = df.drop(columns=['sat_entorno_num', 'satisfaccion_num', 'sat_compas_num'])

#Analisis y limpieza sexo

df = df.drop(columns=['sexo'])

print("\nValores faltantes por columna:")
print(df.isnull().sum())