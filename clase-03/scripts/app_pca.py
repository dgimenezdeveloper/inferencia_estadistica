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
    max_comp = min(10, len(feature_cols_pca))
    n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=max_comp, value=2, key="slider_pca")
    # Guardar la selección en session_state para sugerir en otras vistas
    st.session_state["n_comp_pca"] = n_comp
    if feature_cols_pca:
        X_pca = df_pca[feature_cols_pca]
        X_pca_scaled = StandardScaler().fit_transform(X_pca)
        # PCA con todos los componentes para scree plot
        pca_full = PCA(n_components=max_comp)
        pca_full.fit(X_pca_scaled)
        varianza_explicada = pca_full.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        # Scree plot
        st.write("### Gráfico de varianza explicada (Scree plot)")
        fig_scree, ax_scree = plt.subplots()
        ax_scree.bar(range(1, max_comp+1), varianza_explicada, alpha=0.7, label='Varianza explicada')
        ax_scree.plot(range(1, max_comp+1), varianza_acumulada, marker='o', color='red', label='Varianza acumulada')
        ax_scree.set_xlabel('Componente principal')
        ax_scree.set_ylabel('Proporción de varianza')
        ax_scree.set_title('Scree plot y varianza acumulada')
        ax_scree.legend()
        st.pyplot(fig_scree)
        st.write(f"**Varianza acumulada para {n_comp} componentes:** {varianza_acumulada[n_comp-1]:.2%}")
        # PCA con n_comp seleccionados
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

        # Mostrar matriz de covarianza de los datos transformados por PCA (justo después de calcular X_proj)
        st.write("### Matriz de covarianza de los datos transformados por PCA")
        cov_pca = np.cov(X_proj.T)
        pc_names = [f"PC{i+1}" for i in range(X_proj.shape[1])]
        df_cov_pca = pd.DataFrame(cov_pca, index=pc_names, columns=pc_names)
        st.dataframe(df_cov_pca.style.format("{:.2e}"), use_container_width=True)
        st.caption("La matriz de covarianza de los componentes principales debe ser diagonal (o casi), con valores fuera de la diagonal cercanos a cero (en notación científica). Esto confirma que los PCs son ortogonales y no correlacionados.")
