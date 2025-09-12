import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("Reducción de dimensiones: Análisis de Componentes Principales (PCA)")

with st.expander("¿Qué es PCA? (Explicación rápida)"):
    st.markdown("""
    El Análisis de Componentes Principales (PCA) es una técnica que permite reducir la cantidad de variables conservando la mayor información posible. Es útil para visualizar datos multidimensionales y eliminar redundancia.
    """)

# Selección de dataset para PCA
carpeta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos')
archivos_csv_pca = [f for f in os.listdir(carpeta_datos) if f.endswith('.csv')] if os.path.exists(carpeta_datos) else []
opcion_archivo_pca = st.selectbox("Seleccionar archivo CSV", archivos_csv_pca)
archivo_subido_pca = st.file_uploader("O sube tu propio archivo CSV para PCA", type=["csv"], key="pca")

# Cargar el dataset PCA
df_pca = None
if archivo_subido_pca is not None:
    df_pca = pd.read_csv(archivo_subido_pca)
    st.success("Archivo subido cargado correctamente.")
elif opcion_archivo_pca:
    df_pca = pd.read_csv(os.path.join(carpeta_datos, opcion_archivo_pca))
    st.success(f"Archivo '{opcion_archivo_pca}' cargado correctamente.")

if df_pca is not None:
    st.write("### Vista previa del dataset PCA")
    st.dataframe(df_pca.head())
    # Selección de columnas numéricas
    num_cols_pca = [c for c in df_pca.columns if pd.api.types.is_numeric_dtype(df_pca[c])]
    class_col_pca = st.selectbox("Columna de clase (opcional, para colorear)", ["(Ninguna)"] + num_cols_pca)
    feature_cols_pca = st.multiselect("Selecciona las columnas numéricas para PCA:", [c for c in num_cols_pca if c != class_col_pca], default=[c for c in num_cols_pca if c != class_col_pca])
    n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=min(10, len(feature_cols_pca)), value=2)
    if feature_cols_pca:
        X_pca = df_pca[feature_cols_pca]
        X_pca_scaled = StandardScaler().fit_transform(X_pca)
        pca = PCA(n_components=n_comp)
        X_proj = pca.fit_transform(X_pca_scaled)
        st.write(f"Varianza explicada por los {n_comp} componentes: {pca.explained_variance_ratio_}")
        # Graficar los dos primeros componentes
        st.write("### Visualización de los dos primeros componentes principales")
        fig_pca, ax_pca = plt.subplots()
        if class_col_pca != "(Ninguna)":
            scatter = ax_pca.scatter(X_proj[:,0], X_proj[:,1], c=df_pca[class_col_pca], cmap='viridis')
            plt.colorbar(scatter, ax=ax_pca, label=class_col_pca)
        else:
            ax_pca.scatter(X_proj[:,0], X_proj[:,1])
        ax_pca.set_xlabel('Componente 1')
        ax_pca.set_ylabel('Componente 2')
        ax_pca.set_title('PCA - Proyección 2D')
        st.pyplot(fig_pca)
