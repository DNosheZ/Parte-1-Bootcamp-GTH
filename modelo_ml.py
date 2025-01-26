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
import matplotlib.pyplot as plt


#Funciones 
def gini_index(y):
    probs = y.value_counts(normalize=True)
    return 1 - sum(probs**2)

# Cargar el archivo
df = pd.read_csv('Data_empleados_historicos_clean.csv')
df['id'] = range(1, len(df) + 1)  # Agregar columna incremental 'id'

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
X_train_resampled1.to_csv('X_train_resampled1.csv', index=False)#oversampled
X_train_resampled2.to_csv('X_train_resampled2.csv', index=False)#undersampled
X_test.to_csv('X_test.csv', index=False)
y_train_resampled1.to_csv('y_train_resampled1.csv', index=False)#oversampled
y_train_resampled2.to_csv('y_train_resampled2.csv', index=False)#undersampled
y_test.to_csv('y_test.csv', index=False)

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

# resultados = {}

# for nombre, modelo in modelos.items():
#     print(f"\nEvaluando: {nombre}")
    
#     if nombre in ["SVM", "Red Neuronal"]:
#         modelo.fit(X_train_scaled1, y_train1)
#         y_pred = modelo.predict(X_test_scaled)
#         y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
#     elif nombre in ["Random Forest", "Árbol de Decisión"]:
#         modelo.fit(X_train2, y_train2)
#         y_pred = modelo.predict(X_test)
#         y_prob = modelo.predict_proba(X_test)[:, 1]
#     else:
#         modelo.fit(X_train_scaled2, y_train2)
#         y_pred = modelo.predict(X_test_scaled)
#         y_prob = modelo.predict_proba(X_test_scaled)[:, 1]
    
#     resultados[nombre] = {
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred, zero_division=1),
#         "Recall": recall_score(y_test, y_pred),
#         "F1-score": f1_score(y_test, y_pred),
#         "AUC-ROC": roc_auc_score(y_test, y_prob)
#     }

#     print(classification_report(y_test, y_pred))

# resultados_df = pd.DataFrame(resultados).T
# print(resultados_df)


'''
Accuracy	Proporción de predicciones correctas sobre el total.
Precision	Proporción de positivos predichos que realmente son positivos.
Recall	Proporción de positivos reales que fueron identificados.
F1-score	Promedio armonizado de Precision y Recall.


en vista que los datos estan balanceados, se prioriza la eleccion segun el parametro Accuracy
dado que se deben interpretar los resultados, Precision sera clave para escoger el mejor modelo

en base a las dos metricas anteriores, se escoge Random Forest como el modelo a emplear para predecir la rotacion de puesto
'''


# Importancia de características con Random Forest
modelo_rf = modelos['Random Forest']
modelo_rf.fit(X_train2, y_train2)  # Usar datos balanceados (undersampled)

# Importancias directas
importancias = modelo_rf.feature_importances_

# Crear un DataFrame con las importancias directas
importancia_rf_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Agrupar importancias por variables originales
importancias_agrupadas_rf = {}

for var in categorical_vars:
    relacionadas = [col for col in X_train.columns if col.startswith(var)]
    importancia_total = importancia_rf_df[importancia_rf_df['Variable'].isin(relacionadas)]['Importancia'].sum()
    importancias_agrupadas_rf[var] = importancia_total

# Crear DataFrame agrupado
importancias_agrupadas_rf_df = pd.DataFrame({
    'Variable': list(importancias_agrupadas_rf.keys()),
    'Importancia': list(importancias_agrupadas_rf.values())
}).sort_values(by='Importancia', ascending=False)

# Visualizar importancias agrupadas (Random Forest)
plt.figure(figsize=(10, 6))
plt.barh(importancias_agrupadas_rf_df['Variable'][:10], importancias_agrupadas_rf_df['Importancia'][:10])
plt.gca().invert_yaxis()
plt.xlabel("Importancia")
plt.title("Características más importantes agrupadas según Random Forest")
plt.show()

# Importancia de características con SVM
modelo_svm = SVC(kernel='linear', random_state=42, probability=True)
modelo_svm.fit(X_train_scaled1, y_train1)  # Usar datos balanceados (oversampled)

coeficientes = modelo_svm.coef_[0]  # Coeficientes del hiperplano

# Crear un DataFrame con las importancias directas
importancia_svm_df = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': coeficientes
}).sort_values(by='Importancia', key=abs, ascending=False)

# Agrupar importancias por variables originales
importancias_agrupadas_svm = {}

for var in categorical_vars:
    relacionadas = [col for col in X_train.columns if col.startswith(var)]
    importancia_total = importancia_svm_df[importancia_svm_df['Variable'].isin(relacionadas)]['Importancia'].sum()
    importancias_agrupadas_svm[var] = importancia_total

# Crear DataFrame agrupado
importancias_agrupadas_svm_df = pd.DataFrame({
    'Variable': list(importancias_agrupadas_svm.keys()),
    'Importancia': list(importancias_agrupadas_svm.values())
}).sort_values(by='Importancia', key=abs, ascending=False)

# Visualizar importancias agrupadas (SVM)
# plt.figure(figsize=(10, 6))
# plt.barh(importancias_agrupadas_svm_df['Variable'][:10], importancias_agrupadas_svm_df['Importancia'][:10])
# plt.gca().invert_yaxis()
# plt.xlabel("Importancia")
# plt.title("Características más importantes agrupadas según SVM (Kernel Lineal)")
# plt.show()


# Cargar la base de datos actual
df_actuales = pd.read_csv('Data_empleados_actuales_2024_clean.csv')
df_actuales['id'] = range(1, len(df_actuales) + 1)


# Crear las dummies necesarias (como en la base de entrenamiento)
categorical_vars = ['satisfaccion_entorno', 'implicacion', 'satisfaccion_trabajo', 
                    'estado_civil', 'mayor_edad', 'evaluacion', 'satisfaccion_companeros', 
                    'carrera', 'educacion', 'departamento', 'viajes', 'puesto', 'horas_extra']

df_actuales = pd.get_dummies(df_actuales, columns=categorical_vars, drop_first=True)

# Manejar columnas faltantes o adicionales
missing_cols = set(X.columns) - set(df_actuales.columns)
for col in missing_cols:
    df_actuales[col] = 0  # Agregar columnas faltantes con valor 0

extra_cols = set(df_actuales.columns) - set(X.columns)
df_actuales = df_actuales.drop(extra_cols, axis=1)  # Quitar columnas no necesarias

# Separar 'id' de la base actual
id_actuales = df_actuales['id']
X_actuales = df_actuales.drop('id', axis=1)

# Escalar los datos usando el mismo escalador usado en entrenamiento
X_actuales = df_actuales
scaler = StandardScaler()
X_actuales_scaled = scaler.fit_transform(X_actuales)  # Ajustar y transformar

# Asegurarse de que las columnas sean las mismas
for col in X_train.columns:
    if col not in X_actuales.columns:
        X_actuales[col] = 0  # Agregar columna faltante con valor por defecto

# Eliminar columnas adicionales
X_actuales = X_actuales[X_train.columns]

# Confirmar que el orden es correcto
assert list(X_actuales.columns) == list(X_train.columns), "Las columnas no coinciden exactamente"



# Random Forest - Predicción de probabilidades
rf_predictions = modelo_rf.predict_proba(X_actuales)[:, 1]  # Probabilidad de abandono
df_actuales['probabilidad_abandono_rf'] = rf_predictions


# SVM - Predicción de probabilidades
svm_predictions = modelo_svm.decision_function(X_actuales_scaled)  # Puntuación del modelo
svm_probabilities = (svm_predictions - svm_predictions.min()) / (
    svm_predictions.max() - svm_predictions.min()
)  # Escalar a rango [0, 1]
df_actuales['probabilidad_abandono_svm'] = svm_probabilities


# Ordenar por probabilidad de abandono según Random Forest
df_actuales_sorted_rf = df_actuales.sort_values(by='probabilidad_abandono_rf', ascending=False)

# Mostrar los 10 empleados con mayor probabilidad de abandonar
print(df_actuales_sorted_rf[['id', 'probabilidad_abandono_rf']].head(10))

df_actuales.to_csv('Predicciones_abandono_empleados_actuales.csv', index=False)
