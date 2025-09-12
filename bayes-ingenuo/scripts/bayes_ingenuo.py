import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import os

# Get the directory of the current script and build the relative path to the data file
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir, '..', 'datos', 'ddbb-correr-bayes-ingenuo.csv')

try:
    # Cargar datos
    df = pd.read_csv(DATA_FILE)
    print(f"Datos cargados desde {DATA_FILE} (filas: {df.shape[0]}, columnas: {df.shape[1]})")
    print(df.head())
    # Imputar valores faltantes con la media de cada columna
    if df.isnull().values.any():
        print("\nSe encontraron valores faltantes. Imputando con la media de cada columna...")
        df = df.fillna(df.mean(numeric_only=True))
        print("Valores faltantes imputados.")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    sys.exit(1)

# Suponiendo que la última columna es la clase
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Si se necesitan seleccionar columnas específicas, descomentar y editar:
# X = df[['col1', 'col2', ...]]
# y = df['nombre_columna_clase']

# Separar en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Predecir
y_pred = modelo.predict(X_test)

# Métricas
print("\n--- Resultados ---")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.3f}")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Visualización de la matriz de confusión
plt.figure(figsize=(6,4))
plt.title('Matriz de Confusión')
plt.imshow(confusion_matrix(y_test, y_pred), cmap='Blues', interpolation='nearest')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.colorbar()
plt.show()
