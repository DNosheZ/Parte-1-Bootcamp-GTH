import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Cargar el archivo
df = pd.read_csv('Data_empleados_historicos_clean.csv')

mean_distance = df.groupby('abandono')['distancia_casa'].mean()
std_distance = df.groupby('abandono')['distancia_casa'].std()
print("Promedio de distancia a casa por abandono:")
print(mean_distance)
print("\nDesviación estándar de distancia a casa por abandono:")
print(std_distance)

'''
Si el promedio de distancia a casa es significativamente mayor para 
quienes abandonaron, podría ser un factor importante.
'''

sns.boxplot(data=df, x='abandono', y='distancia_casa')
plt.title("Distribución de Distancia a Casa por Estado de Abandono")
plt.xlabel("Abandono")
plt.ylabel("Distancia a Casa")
plt.show()


#correlaciones
#print(f'Correlacion abandono y {df['abandono'].corr(df[''])}')