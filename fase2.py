import pandas as pd

# Cargar la base de predicciones
df_predicciones = pd.read_csv('Predicciones_abandono_empleados_actuales.csv')

# Función para convertir variables dummy a categóricas
def revertir_dummies(df, prefix):
    columnas = [col for col in df.columns if col.startswith(prefix)]
    df[prefix] = df[columnas].idxmax(axis=1).str.replace(prefix + '_', '', regex=True)
    df.drop(columns=columnas, inplace=True)
    return df

# Convertir cada grupo de variables dummy a su formato categórico original
df_predicciones = revertir_dummies(df_predicciones, 'satisfaccion_entorno')
df_predicciones = revertir_dummies(df_predicciones, 'implicacion')
df_predicciones = revertir_dummies(df_predicciones, 'satisfaccion_trabajo')
df_predicciones = revertir_dummies(df_predicciones, 'estado_civil')
df_predicciones = revertir_dummies(df_predicciones, 'evaluacion')
df_predicciones = revertir_dummies(df_predicciones, 'satisfaccion_companeros')
df_predicciones = revertir_dummies(df_predicciones, 'carrera')
df_predicciones = revertir_dummies(df_predicciones, 'educacion')
df_predicciones = revertir_dummies(df_predicciones, 'departamento')
df_predicciones = revertir_dummies(df_predicciones, 'viajes')
df_predicciones = revertir_dummies(df_predicciones, 'puesto')
df_predicciones = revertir_dummies(df_predicciones, 'horas_extra')

# Guardar el archivo con las variables categóricas restauradas
df_predicciones.to_csv('Predicciones_abandono_empleados_actuales_categorico.csv', index=False)

print("Transformación completada. Archivo guardado como 'Predicciones_abandono_empleados_actuales_categorico.csv'.")

# Cargar el archivo CSV
df_actuales = pd.read_csv('Predicciones_abandono_empleados_actuales_categorico.csv')

# Lista de IDs de interés
ids_interes = [148, 129, 128, 125, 130]

# Informe 1: Selección de columnas específicas
columnas_seleccionadas = ['id', 'puesto', 'carrera', 'satisfaccion_entorno', 'estado_civil', 
                          'satisfaccion_companeros', 'horas_extra', 'educacion', 'departamento', 'viajes']

df_informe1 = df_actuales[df_actuales['id'].isin(ids_interes)][columnas_seleccionadas]

# Guardar en CSV
df_informe1.to_csv('Informe_1_seleccion.csv', index=False)

# Informe 2: Todas las columnas para los registros con los IDs indicados
df_informe2 = df_actuales[df_actuales['id'].isin(ids_interes)]

# Guardar en CSV
df_informe2.to_csv('Informe_2_completo.csv', index=False)

# Mostrar los informes en consola
print("\nInforme 1 - Columnas seleccionadas:")
print(df_informe1)

print("\nInforme 2 - Todos los datos:")
print(df_informe2)
