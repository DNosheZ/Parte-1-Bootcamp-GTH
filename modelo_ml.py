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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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
X_train_resampled1, y_train_resampled1 = smote.fit_resample(X_train, y_train)

# Verificar la proporción de clases después de SMOTE
print("\nDistribución después de SMOTE en y_train:")
print(y_train_resampled1.value_counts(normalize=True))

# Aplicar RandomUnderSampler
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled2, y_train_resampled2 = undersampler.fit_resample(X_train, y_train)#Se emplearan para arboles de decisicion y regresion logistica

# Verificar proporciones
print(y_train_resampled2.value_counts(normalize=True))



'''
Si una clase representa más del 70-75% de los datos, es un desequilibrio moderado.
Si una clase representa más del 80-85%, es un desequilibrio fuerte.
Si una clase representa más del 90%, es un desequilibrio extremo.
'''

#hay un fuerte desequilibrio en las distribuciones obtenidas


'''guardado de conjuntos para entrenamiento y prueba'''
# X_train_resampled1.to_csv('X_train_resampled1.csv', index=False)#oversampled
# X_train_resampled2.to_csv('X_train_resampled2.csv', index=False)#undersampled
# X_test.to_csv('X_test.csv', index=False)
# y_train_resampled1.to_csv('y_train_resampled1.csv', index=False)#oversampled
# y_train_resampled2.to_csv('y_train_resampled2.csv', index=False)#undersampled
# y_test.to_csv('y_test.csv', index=False)

X_train1 = pd.read_csv('X_train_resampled1.csv')#aversampled
X_train2 = pd.read_csv('X_train_resampled2.csv')#undersampled
X_test = pd.read_csv('X_test.csv')
y_train1 = pd.read_csv('y_train_resampled1.csv').squeeze() #oversampled
y_train2 = pd.read_csv('y_train_resampled2.csv').squeeze()  # .squeeze() convierte el DataFrame a Series, undersampled
y_test = pd.read_csv('y_test.csv').squeeze()


#para SVM o Redes neuronales
# Crear el escalador
scaler = StandardScaler()

# Escalar los datos y conservar columnas originales
X_train_scaled1 = pd.DataFrame(scaler.fit_transform(X_train1), columns=X_train1.columns)
X_train_scaled2 = pd.DataFrame(scaler.fit_transform(X_train2), columns=X_train2.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Modelos a evaluar
modelos = {
    "Regresión Logística": LogisticRegression(random_state=42, max_iter=500, class_weight='balanced'),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100),
    "SVM": SVC(random_state=42, probability=True),
    "Red Neuronal": MLPClassifier(random_state=42, max_iter=500)
}

resultados = {}

for nombre, modelo in modelos.items():
    print(f"\nEvaluando: {nombre}")
    
    if nombre in ["SVM", "Red Neuronal"]:
        modelo.fit(X_train_scaled1, y_train1)
        y_pred = modelo.predict(X_test_scaled)
        y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    elif nombre in ["Random Forest", "Árbol de Decisión"]:
        modelo.fit(X_train2, y_train2)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
    else:
        modelo.fit(X_train_scaled2, y_train2)
        y_pred = modelo.predict(X_test_scaled)
        y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
    resultados[nombre] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=1),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    }

    print(classification_report(y_test, y_pred))

resultados_df = pd.DataFrame(resultados).T
print(resultados_df)


'''
Accuracy	Proporción de predicciones correctas sobre el total.
Precision	Proporción de positivos predichos que realmente son positivos.
Recall	Proporción de positivos reales que fueron identificados.
F1-score	Promedio armonizado de Precision y Recall.


en vista que los datos estan balanceados, se prioriza la eleccion segun el parametro Accuracy
dado que se deben interpretar los resultados, Precision sera clave para escoger el mejor modelo

en base a las dos metricas anteriores, se escoge Random Forest como el modelo a emplear para predecir la rotacion de puesto
'''