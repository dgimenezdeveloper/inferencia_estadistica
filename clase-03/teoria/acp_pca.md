# Análisis de Componentes Principales (ACP / PCA)

## ¿Qué es PCA?
El Análisis de Componentes Principales (ACP o PCA por sus siglas en inglés) es una técnica de reducción de dimensiones que transforma un conjunto de variables originales en un nuevo conjunto de variables (componentes principales) que explican la mayor parte de la varianza de los datos.

## ¿Para qué sirve?
- Visualizar datos multidimensionales en 2D o 3D.
- Eliminar redundancia y correlación entre variables.
- Identificar patrones y grupos.
- Preprocesar datos para algoritmos de Machine Learning.

## Pasos básicos de PCA
1. **Estandarizar los datos** (media 0, varianza 1).
2. **Calcular la matriz de covarianza**.
3. **Obtener los eigenvectores y eigenvalores**.
4. **Seleccionar los componentes principales** (los que explican más varianza).
5. **Proyectar los datos** en el nuevo espacio.

## Ejemplo práctico con Wine.csv
```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Wine.csv')
X = df.drop('Class', axis=1)  # Suponiendo que 'Class' es la columna de clase
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Varianza explicada:", pca.explained_variance_ratio_)
```

## Interpretación
- Los componentes principales son combinaciones lineales de las variables originales.
- El primer componente explica la mayor varianza posible, el segundo la siguiente mayor, y así sucesivamente.
- Puedes graficar los datos proyectados para visualizar agrupamientos y patrones.

## Referencias
- ACPnotasdeclase.pdf
- [Scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
