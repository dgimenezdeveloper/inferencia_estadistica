import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('../datos/Wine.csv')
X = df.drop('Class', axis=1)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Class'])
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('PCA - Wine Dataset')
plt.colorbar(label='Clase')
plt.show()
