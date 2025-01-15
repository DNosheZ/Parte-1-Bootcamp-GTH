import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


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
                    'carrera', 'educacion', 'departamento', 'viajes','puesto','horas_extra']

df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Separar las características (X) y la variable objetivo (y)
X = df.drop('abandono', axis=1)
y = df['abandono']

# Dividir los datos en entrenamiento y prueba (80% - 20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las dimensiones de cada conjunto
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")


# # Verificar distribución de la variable objetivo
# print("Distribución en y_train:")
# print(y_train.value_counts(normalize=True))

# print("\nDistribución en y_test:")
# print(y_test.value_counts(normalize=True))

# gini = gini_index(df['abandono'])
# print(f"Gini Index: {gini:.2f}")

# Dividir los datos con estratificación para mantener la misma proporción de la variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print('\nPost-estratificacion\n')
# # Verificar distribución de la variable objetivo
# print("Distribución en y_train:")
# print(y_train.value_counts(normalize=True))

# print("\nDistribución en y_test:")
# print(y_test.value_counts(normalize=True))

# gini = gini_index(df['abandono'])
# print(f"Gini Index: {gini:.2f}")

# Aplicar SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verificar la proporción de clases después de SMOTE
print("\nDistribución después de SMOTE en y_train:")
print(y_train_resampled.value_counts(normalize=True))

# Aplicar RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Verificar proporciones
print(y_train_resampled.value_counts(normalize=True))



'''
Si una clase representa más del 70-75% de los datos, es un desequilibrio moderado.
Si una clase representa más del 80-85%, es un desequilibrio fuerte.
Si una clase representa más del 90%, es un desequilibrio extremo.
'''

#hay un fuerte desequilibrio en las distribuciones obtenidas


'''guardado de conjuntos para entrenamiento y prueba'''
X_train.to_csv('X_train_resampled.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train_resampled.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # .squeeze() convierte el DataFrame a Series
y_test = pd.read_csv('y_test.csv').squeeze()


#para SVM o Redes neuronales
# Crear el escalador
scaler = StandardScaler()

# Ajustar y transformar los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Transformar los datos de prueba
X_test_scaled = scaler.transform(X_test)

# Modelos a evaluar
modelos = {
    "Regresión Logística": LogisticRegression(random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Red Neuronal": MLPClassifier(random_state=42)
}

for nombre, modelo in modelos.items():
    print(f"\nEvaluando: {nombre}")
    
    # Para SVM y Red Neuronal usamos los datos escalados
    if nombre in ["SVM", "Red Neuronal"]:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
    
    # Imprimir el reporte de clasificación
    print(classification_report(y_test, y_pred))