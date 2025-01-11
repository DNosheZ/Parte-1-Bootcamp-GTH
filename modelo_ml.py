import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Funciones 
def gini_index(y):
    probs = y.value_counts(normalize=True)
    return 1 - sum(probs**2)

# Cargar el archivo
df = pd.read_csv('Data_empleados_historicos_clean.csv')

# Convertir la variable objetivo en binaria
df['abandono'] = df['abandono'].map({'Yes': 1, 'No': 0})

# Crear dummies para variables categóricas
categorical_vars = ['satisfaccion_entorno', 'implicacion', 'satisfaccion_trabajo', 
                    'estado_civil', 'mayor_edad', 'evaluacion', 'satisfaccion_companeros', 
                    'carrera', 'educacion', 'departamento', 'viajes']

df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Separar las características (X) y la variable objetivo (y)
X = df.drop('abandono', axis=1)
y = df['abandono']

# Dividir los datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las dimensiones de cada conjunto
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Verificar distribución de la variable objetivo
print("Distribución en y_train:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en y_test:")
print(y_test.value_counts(normalize=True))

gini = gini_index(df['abandono'])
print(f"Gini Index: {gini:.2f}")

# Dividir los datos con estratificación para mantener la misma proporción de la variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('\nPost-estratificacion\n')
# Verificar distribución de la variable objetivo
print("Distribución en y_train:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en y_test:")
print(y_test.value_counts(normalize=True))

gini = gini_index(df['abandono'])
print(f"Gini Index: {gini:.2f}")

'''
Si una clase representa más del 70-75% de los datos, es un desequilibrio moderado.
Si una clase representa más del 80-85%, es un desequilibrio fuerte.
Si una clase representa más del 90%, es un desequilibrio extremo.
'''

#hay un fuerte desequilibrio en las distribuciones abtenidas


'''guardado de conjuntos para entrenamiento y prueba'''
# X_train.to_csv('X_train.csv', index=False)
# X_test.to_csv('X_test.csv', index=False)
# y_train.to_csv('y_train.csv', index=False)
# y_test.to_csv('y_test.csv', index=False)



#para SVM o Redes neuronales
# Crear el escalador
# scaler = StandardScaler()

# # Ajustar y transformar los datos de entrenamiento
# X_train_scaled = scaler.fit_transform(X_train)

# # Transformar los datos de prueba
# X_test_scaled = scaler.transform(X_test)