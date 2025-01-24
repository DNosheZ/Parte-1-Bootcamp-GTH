import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo de datos
df = pd.read_csv('Data_empleados_historicos_clean.csv')

# Verificar los valores únicos de la columna abandono
print(df['abandono'].unique())

# Convertir la variable 'abandono' de Yes/No a formato categórico para mejor visualización
df['abandono'] = df['abandono'].map({'Yes': 'Sí', 'No': 'No'})
for i in ['num_formaciones_ult_ano','anos_con_manager_actual','num_empresas_anteriores','anos_compania','distancia_casa','distancia_casa','edad']:
    # Verificar valores nulos en la columna anos_experiencia
    print(f"Valores nulos en {i}: {df[i].isnull().sum()}")

    # Eliminar filas con valores nulos en anos_experiencia para evitar errores en el gráfico
    df_clean = df.dropna(subset=[i])

    # Crear el boxplot con Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='abandono',
        y=i,
        data=df_clean,
        palette="Set2",  # Colores personalizados
        showfliers=True,  # Mostrar o no los valores atípicos (outliers)
        boxprops={'edgecolor': 'black'},  # Personalización de los bordes de las cajas
    )

    # Personalización del gráfico
    plt.title('Distribución de '+i+' por Estado de Abandono', fontsize=14)
    plt.xlabel('Estado de Abandono', fontsize=12)
    plt.ylabel(i, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrar el gráfico
    plt.show()
