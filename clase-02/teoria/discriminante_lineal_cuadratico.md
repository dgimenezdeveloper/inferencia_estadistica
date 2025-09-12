# Análisis Discriminante Lineal y Cuadrático (LDA & QDA)

## 1. Introducción

El Análisis Discriminante es una técnica de clasificación supervisada que busca encontrar una función que separe de manera óptima dos o más clases. Los dos métodos más conocidos son:

- **LDA (Linear Discriminant Analysis):** Asume que las clases tienen igual matriz de covarianza (varianza).
- **QDA (Quadratic Discriminant Analysis):** Permite que cada clase tenga su propia matriz de covarianza.

## 2. Fundamentos Teóricos

### 2.1. LDA

- **Supuestos:** Las clases tienen distribución normal multivariada con igual matriz de covarianza.
- **Función discriminante:** Es lineal respecto a las variables predictoras.
- **Ecuación general:**
  $$
  \delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)
  $$
  Donde:
  - $\mu_k$: media de la clase k
  - $\Sigma$: matriz de covarianza común
  - $\pi_k$: probabilidad a priori de la clase k

### 2.2. QDA

- **Supuestos:** Las clases tienen distribución normal multivariada, pero cada una con su propia matriz de covarianza.
- **Función discriminante:** Es cuadrática respecto a las variables predictoras.
- **Ecuación general:**
  $$
  \delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log(\pi_k)
  $$
  Donde:
  - $\Sigma_k$: matriz de covarianza de la clase k

## 3. Ejemplo Práctico: Dataset Iris

### 3.1. LDA con Python (scikit-learn)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                  columns = iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['species']

model = LinearDiscriminantAnalysis()
model.fit(X, y)
print(model.predict([[4, 1, 1.8, 1]]))  # Ejemplo de predicción
```

### 3.2. QDA con Python (scikit-learn)

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

model = QuadraticDiscriminantAnalysis()
model.fit(X, y)
print(model.predict([[5.5, 3, 1.0, 0.4]]))  # Ejemplo de predicción
```

## 4. Ventajas y Desventajas

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| LDA    | Simple, robusto, funciona bien si los supuestos se cumplen | No modela bien clases con diferente varianza |
| QDA    | Más flexible, modela clases con diferente varianza | Requiere más datos, puede sobreajustar |,m

## 5. ¿Cuándo usar cada uno?

- **LDA:** Cuando las clases tienen varianzas similares y los datos son aproximadamente normales.
- **QDA:** Cuando las clases tienen varianzas claramente diferentes.

## 6. Referencias

- [Statology: Linear Discriminant Analysis in Python](https://www.statology.org/linear-discriminant-analysis-in-python/)
- [Statology: Quadratic Discriminant Analysis in Python](https://www.statology.org/quadratic-discriminant-analysis-in-python/)
- Apuntes y PDF: `discriminante_l_q.pdf`
