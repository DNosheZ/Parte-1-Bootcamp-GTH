import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo de datos
df = pd.read_csv('Data_empleados_historicos_clean.csv')

# Convertir la variable 'abandono' de Yes/No a formato categórico para mejor visualización
df['abandono'] = df['abandono'].map({'Yes': 'Sí', 'No': 'No'})

# Verificar valores nulos en la columna nivel_acciones
print(f"Valores nulos en nivel_acciones: {df['nivel_acciones'].isnull().sum()}")

# Eliminar filas con valores nulos en nivel_acciones para evitar errores en el gráfico
df_clean = df.dropna(subset=['nivel_acciones'])

# Crear el boxplot con Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='abandono',
    y='nivel_acciones',
    data=df_clean,
    palette="pastel",  # Colores suaves
    showfliers=True,  # Mostrar valores atípicos
    boxprops={'edgecolor': 'black'},  # Personalización de bordes de la caja
)

# Personalización del gráfico
plt.title('Distribución de Nivel de Acciones por Estado de Abandono', fontsize=14)
plt.xlabel('Estado de Abandono', fontsize=12)
plt.ylabel('Nivel de Acciones', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.show()

