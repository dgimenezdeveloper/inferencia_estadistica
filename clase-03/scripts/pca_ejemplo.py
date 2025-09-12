import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar dataset de ejemplo (puedes cambiar el nombre)
df = pd.read_csv('../Wine.csv')
X = df.drop('Class', axis=1)
X_scaled = StandardScaler().fit_transform(X)

# PCA a 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Varianza explicada:", pca.explained_variance_ratio_)

# Graficar
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Class'])
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('PCA - Wine Dataset')
plt.colorbar(label='Clase')
plt.savefig('../pca_wine_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado como '../pca_wine_plot.png'")
