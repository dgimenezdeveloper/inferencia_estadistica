def plot_flechas_componentes_principales(X, pca, var_x, var_y, feature_cols_pca, scale=5):
	"""
	Visualiza los datos originales en dos variables y las flechas de PC1 y PC2 sobre ese plano.
	"""
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(figsize=(7, 6))
	ax.scatter(X[var_x], X[var_y], alpha=0.6, color='gray', label='Datos')
	x_mean = X[var_x].mean()
	y_mean = X[var_y].mean()
	idx_x = feature_cols_pca.index(var_x)
	idx_y = feature_cols_pca.index(var_y)
	# PC1
	ax.arrow(x_mean, y_mean, pca.components_[0, idx_x]*scale, pca.components_[0, idx_y]*scale, width=0.05, color="purple", label="PC1")
	# PC2
	ax.arrow(x_mean, y_mean, pca.components_[1, idx_x]*scale, pca.components_[1, idx_y]*scale, width=0.05, color="black", label="PC2")
	ax.set_xlabel(var_x)
	ax.set_ylabel(var_y)
	ax.set_title(f"Flechas de PC1 y PC2 sobre {var_x} vs {var_y}")
	ax.legend(["Datos", "PC1", "PC2"])
	fig.tight_layout()
	return fig
def interpretar_componentes_principales(components, feature_names, top_n=3):
	"""
	Devuelve una lista de strings con la interpretación automática de cada componente principal.
	Ejemplo: 'PC1 está principalmente influenciado por: var1 (+), var2 (-), var3 (+)'
	"""
	interpretaciones = []
	for i, row in enumerate(components):
		pesos = np.abs(row)
		top_idx = np.argsort(pesos)[::-1][:top_n]
		top_vars = [(feature_names[j], row[j]) for j in top_idx]
		partes = []
		for var, peso in top_vars:
			sentido = "+" if peso > 0 else "-"
			partes.append(f"{var} ({sentido})")
		explicacion = f"PC{i+1} está principalmente influenciado por: " + ", ".join(partes)
		interpretaciones.append(explicacion)
	return interpretaciones

def ecuacion_componente_principal(components, feature_names):
	"""
	Devuelve una lista de strings con la ecuación formateada de cada componente principal.
	Ejemplo: 'PC1 = +0.45 × var1 -0.32 × var2 ...'
	"""
	ecuaciones = []
	for i, row in enumerate(components):
		ecuacion = f"PC{i+1} = "
		partes = []
		for peso, var in zip(row, feature_names):
			partes.append(f"{peso:+.2f} × {var}")
		ecuacion += " "+" ".join(partes)
		ecuaciones.append(ecuacion)
	return ecuaciones
def plot_heatmap_covarianza(df_cov, title='Heatmap matriz de covarianza (PCA)'):
	"""
	Dibuja un heatmap de la matriz de covarianza usando seaborn y matplotlib.
	"""
	import matplotlib.pyplot as plt
	import seaborn as sns
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(df_cov, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, annot_kws={"size":8})
	ax.set_title(title, fontsize=14)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
	ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
	fig.tight_layout()
	return fig

def plot_heatmap_correlacion(df_corr, title='Heatmap matriz de correlación (PCA)'):
	"""
	Dibuja un heatmap de la matriz de correlación usando seaborn y matplotlib.
	"""
	import matplotlib.pyplot as plt
	import seaborn as sns
	fig, ax = plt.subplots(figsize=(8, 6))
	sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0, annot_kws={"size":8})
	ax.set_title(title, fontsize=14)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
	ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
	fig.tight_layout()
	return fig

def plot_varianza_acumulada_pca(var_exp, n_comp):
	"""
	Gráfico de línea de varianza acumulada por componente principal (PCA).
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	var_acum = np.cumsum(var_exp)
	fig, ax = plt.subplots()
	x_vals = range(1, n_comp+1)
	ax.plot(x_vals, var_acum, color='red', marker='o')
	ax.set_xlabel('Componente principal')
	ax.set_ylabel('Varianza acumulada')
	ax.set_title('Varianza acumulada por componente')
	ax.set_ylim(0, 1.05)
	# Método del codo mejorado
	difs = np.diff(var_acum)
	codo_idx = None
	for i, d in enumerate(difs):
		if d < 0.02:
			codo_idx = i
			break
	if codo_idx is None:
		codo_idx = len(var_acum) - 1
	ax.axvline(codo_idx+1, color='blue', linestyle='--', label=f'Recomendado: {codo_idx+1} componentes')
	ax.legend()
	fig.tight_layout()
	return fig, codo_idx+1
def plot_varianza_explicada_pca(var_exp, n_comp):
	"""
	Gráfico de barras de varianza explicada por componente principal.
	"""
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.bar(range(1, n_comp+1), [v*100 for v in var_exp], color='dodgerblue')
	ax.set_xlabel('Componente principal')
	ax.set_ylabel('Varianza explicada (%)')
	ax.set_title('Porcentaje de varianza explicada por componente')
	for i, v in enumerate(var_exp):
		ax.text(i+1, v*100, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=9)
	fig.tight_layout()
	return fig
# Archivo de visualizaciones para la app integrada

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metricas_por_clase(metricas, class_names, y_true):
	"""
	Devuelve una figura matplotlib con:
	- Barras de precision/recall/F1 por clase
	- Barras de soporte (conteo) por clase
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

	# Gráfico de barras de métricas por clase
	x = np.arange(len(class_names))
	width = 0.25

	ax1.bar(x - width, metricas['precision_per_class'], width, label='Precision', alpha=0.8)
	ax1.bar(x, metricas['recall_per_class'], width, label='Recall', alpha=0.8)
	ax1.bar(x + width, metricas['f1_per_class'], width, label='F1-Score', alpha=0.8)

	ax1.set_xlabel('Clases')
	ax1.set_ylabel('Score')
	ax1.set_title('Métricas por Clase')
	ax1.set_xticks(x)
	ax1.set_xticklabels(class_names, rotation=45 if len(class_names) > 3 else 0)
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# Gráfico de soporte (cantidad de muestras por clase)
	class_counts = pd.Series(y_true).value_counts().sort_index()
	ax2.bar(range(len(class_counts)), class_counts.values, alpha=0.7, color='skyblue')
	ax2.set_xlabel('Clases')
	ax2.set_ylabel('Número de muestras')
	ax2.set_title('Distribución de clases en el dataset')
	ax2.set_xticks(range(len(class_names)))
	ax2.set_xticklabels(class_names, rotation=45 if len(class_names) > 3 else 0)
	ax2.grid(True, alpha=0.3)

	# Agregar valores en las barras
	maxv = max(class_counts.values) if len(class_counts.values) else 0
	for i, v in enumerate(class_counts.values):
		ax2.text(i, v + (maxv * 0.01 if maxv else 0.05), str(v), ha='center', va='bottom', fontsize=9)

	plt.tight_layout()
	return fig
