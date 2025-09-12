from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import datasets
import pandas as pd
import numpy as np

# Cargar el dataset Iris
df = pd.DataFrame(data = np.c_[datasets.load_iris()['data'], datasets.load_iris()['target']],
                  columns = datasets.load_iris()['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(datasets.load_iris().target, datasets.load_iris().target_names)

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['species']

# Entrenar modelo QDA
model = QuadraticDiscriminantAnalysis()
model.fit(X, y)

# Predicción ejemplo
nueva_obs = [[5.5, 3, 1.0, 0.4]]
prediccion = model.predict(nueva_obs)
print(f"Predicción para {nueva_obs}: {prediccion[0]}")
