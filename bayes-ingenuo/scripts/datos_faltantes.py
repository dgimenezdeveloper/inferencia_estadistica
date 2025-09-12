import pandas as pd

df = pd.read_csv('ddbb-correr-bayes-ingenuo.csv')

# Mostrar cantidad de valores faltantes por columna
print("Valores faltantes por columna:")
print(df.isnull().sum())

# Mostrar las filas que tienen al menos un valor faltante
print("\nFilas con valores faltantes:")
print(df[df.isnull().any(axis=1)])