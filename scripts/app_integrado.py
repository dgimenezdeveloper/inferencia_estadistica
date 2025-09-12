import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

st.title("Análisis Integrado: Discriminante (LDA/QDA) y PCA")

# Selección de tipo de análisis
analisis = st.sidebar.selectbox("Selecciona el tipo de análisis", ["Discriminante (LDA/QDA)", "Reducción de dimensiones (PCA)"])

carpeta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos')
archivos_csv = [f for f in os.listdir(carpeta_datos) if f.endswith('.csv')] if os.path.exists(carpeta_datos) else []
opcion_archivo = st.selectbox("Seleccionar archivo CSV", archivos_csv)
archivo_subido = st.file_uploader("O sube tu propio archivo CSV", type=["csv"])

# Cargar el dataset
df = None
if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.success("Archivo subido cargado correctamente.")
elif opcion_archivo:
    df = pd.read_csv(os.path.join(carpeta_datos, opcion_archivo))
    st.success(f"Archivo '{opcion_archivo}' cargado correctamente.")

if df is not None:
    st.write("### Vista previa del dataset")
    st.dataframe(df.head())

    if analisis == "Discriminante (LDA/QDA)":
        st.header("Clasificación Discriminante (LDA/QDA)")
        with st.expander("¿Qué es LDA y QDA? (Explicación teórica)"):
            st.markdown("""
            **LDA (Análisis Discriminante Lineal)** y **QDA (Análisis Discriminante Cuadrático)** son algoritmos de clasificación supervisada que buscan separar clases en función de sus características.
            - **LDA** asume que todas las clases tienen igual matriz de covarianza y genera fronteras lineales.
            - **QDA** permite que cada clase tenga su propia matriz de covarianza y genera fronteras cuadráticas.
            """)
        # Filtrar columnas válidas para target
        max_unique_target = 20
        columnas = df.columns.tolist()
        num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in columnas if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() <= max_unique_target]
        if not cat_cols:
            st.error("No hay columnas válidas para usar como variable de clase (target). Elige un dataset con una columna categórica o entera con pocos valores únicos.")
        else:
            target_col = st.selectbox("Selecciona la columna de clase (target):", cat_cols, index=max(0, len(cat_cols)-1))
            feature_cols = st.multiselect(
                "Selecciona las columnas de atributos (features):",
                [c for c in num_cols if c != target_col],
                default=[c for c in num_cols if c != target_col]
            )
            if feature_cols and target_col:
                X = df[feature_cols]
                y = df[target_col]
                if X.isnull().values.any():
                    st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
                    X = X.fillna(X.mean())
                algoritmo = st.selectbox("Selecciona el algoritmo", ["LDA", "QDA", "Bayes Ingenuo"])
                if algoritmo == "LDA":
                    model = LinearDiscriminantAnalysis(store_covariance=True)
                elif algoritmo == "QDA":
                    model = QuadraticDiscriminantAnalysis(store_covariance=True)
                else:
                    from sklearn.naive_bayes import GaussianNB
                    model = GaussianNB()
                model.fit(X, y)
                # Mostrar matriz de covarianza estimada por el modelo (solo LDA/QDA)
                if algoritmo in ["LDA", "QDA"]:
                    st.write(f"### Matriz de covarianza estimada por el modelo ({algoritmo})")
                    import seaborn as sns
                    if algoritmo == "LDA":
                        cov_matrix = model.covariance_
                        df_cov = pd.DataFrame(cov_matrix, index=feature_cols, columns=feature_cols)
                        st.write(df_cov)
                        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
                        sns.heatmap(df_cov, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax_cov, annot_kws={"size":8})
                        ax_cov.set_title('Heatmap matriz de covarianza (LDA)', fontsize=14)
                        ax_cov.set_xticklabels(ax_cov.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                        ax_cov.set_yticklabels(ax_cov.get_yticklabels(), rotation=0, fontsize=9)
                        fig_cov.tight_layout()
                        st.pyplot(fig_cov)
                        st.caption("La matriz de covarianza muestra cómo varían conjuntamente las variables. En LDA es única para todas las clases.")
                    else:
                        cov_matrices = model.covariance_
                        for i, cov in enumerate(cov_matrices):
                            st.write(f"Clase {model.classes_[i]}")
                            df_cov = pd.DataFrame(cov, index=feature_cols, columns=feature_cols)
                            st.write(df_cov)
                            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
                            sns.heatmap(df_cov, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax_cov, annot_kws={"size":8})
                            ax_cov.set_title(f'Heatmap matriz de covarianza (QDA) - Clase {model.classes_[i]}', fontsize=14)
                            ax_cov.set_xticklabels(ax_cov.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                            ax_cov.set_yticklabels(ax_cov.get_yticklabels(), rotation=0, fontsize=9)
                            fig_cov.tight_layout()
                            st.pyplot(fig_cov)
                        st.caption("En QDA, cada clase tiene su propia matriz de covarianza.")
                nueva_obs = []
                st.write("### Ingresa una observación para predecir la clase")
                for col in feature_cols:
                    val = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))
                    nueva_obs.append(val)
                nueva_obs = [nueva_obs]
                prediccion = model.predict(nueva_obs)
                if st.button("Predecir clase"):
                    st.success(f"Predicción para {nueva_obs}: {prediccion[0]}")
                    st.write(f"Algoritmo usado: {algoritmo}")
                    if algoritmo == "Bayes Ingenuo":
                        st.info("Bayes Ingenuo (Naive Bayes) es un clasificador probabilístico basado en la regla de Bayes y la independencia entre atributos.")
                min_samples_per_class = y.value_counts().min()
                cv_splits = min(5, min_samples_per_class) if min_samples_per_class >= 2 else None
                if cv_splits and cv_splits >= 2:
                    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv_splits)
                    st.write(f"Precisión promedio ({algoritmo}, cv={cv_splits}): {np.mean(scores):.2f}")
                else:
                    st.warning("No se puede calcular validación cruzada porque alguna clase tiene menos de 2 muestras.")
                st.write("### Matriz de confusión")
                y_pred = model.predict(X)
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicción')
                ax.set_ylabel('Real')
                st.pyplot(fig)
                st.write("### Visualización de separación de clases")
                show_proj = st.checkbox("Mostrar gráfico de componentes discriminantes (proyección LDA/QDA)")
                if show_proj:
                    try:
                        if algoritmo == "LDA":
                            X_proj = model.transform(X)
                        else:
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_proj = pca.fit_transform(X)
                        fig3, ax3 = plt.subplots()
                        for clase in np.unique(y):
                            ax3.scatter(X_proj[y == clase, 0], X_proj[y == clase, 1], label=str(clase))
                        ax3.set_xlabel('Componente 1')
                        ax3.set_ylabel('Componente 2')
                        ax3.legend()
                        st.pyplot(fig3)
                    except Exception as e:
                        st.error(f"No se pudo calcular la proyección: {e}")
                elif len(feature_cols) == 2:
                    fig2, ax2 = plt.subplots()
                    for clase in np.unique(y):
                        ax2.scatter(X[y == clase][feature_cols[0]], X[y == clase][feature_cols[1]], label=str(clase))
                    ax2.set_xlabel(feature_cols[0])
                    ax2.set_ylabel(feature_cols[1])
                    ax2.legend()
                    st.pyplot(fig2)
    elif analisis == "Reducción de dimensiones (PCA)":
        st.header("Reducción de dimensiones: PCA")
        with st.expander("¿Qué es PCA? (Explicación rápida)"):
            st.markdown("""
            El Análisis de Componentes Principales (PCA) es una técnica que permite reducir la cantidad de variables conservando la mayor información posible. Es útil para visualizar datos multidimensionales y eliminar redundancia.
            """)
        num_cols_pca = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        class_col_pca = st.selectbox("Columna de clase (opcional, para colorear)", ["(Ninguna)"] + [c for c in df.columns if df[c].nunique() <= 20 and c != "(Ninguna)"])
        feature_cols_pca = st.multiselect("Selecciona las columnas numéricas para PCA:", [c for c in num_cols_pca if c != class_col_pca], default=[c for c in num_cols_pca if c != class_col_pca])
        n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=len(feature_cols_pca), value=2)
        if feature_cols_pca:
            st.info("Las siglas 'PC' significan 'Principal Component' o 'Componente Principal'. Por ejemplo, PC1 es el primer componente principal, PC2 el segundo, y así sucesivamente. Cada uno es una combinación lineal de las variables originales que explica una parte de la varianza total del dataset.")
            X_pca = df[feature_cols_pca]
            X_pca_scaled = StandardScaler().fit_transform(X_pca)
            pca = PCA(n_components=n_comp)
            X_proj = pca.fit_transform(X_pca_scaled)
            var_exp = pca.explained_variance_ratio_
            st.write(f"Varianza explicada por los {n_comp} componentes: {var_exp}")
            st.write(f"Varianza acumulada: {np.sum(var_exp):.2f}")
            # Mostrar composición de cada componente principal
            st.write("### Composición de cada componente principal")
            comp_matrix = pca.components_
            for i, row in enumerate(comp_matrix):
                ecuacion = f"PC{i+1} = "
                partes = []
                for peso, var in zip(row, feature_cols_pca):
                    partes.append(f"{peso:+.2f} × {var}")
                ecuacion += " "+" ".join(partes)
                st.markdown(f"- {ecuacion}")
            # Matriz de componentes principales
            st.write("### Matriz de componentes principales (PCA)")
            st.write(pd.DataFrame(pca.components_, columns=feature_cols_pca, index=[f"PC{i+1}" for i in range(n_comp)]))
            st.caption("Cada fila representa un componente principal y cada columna el peso de la variable original en ese componente.")
            # Matriz de covarianza de los datos originales
            st.write("### Matriz de covarianza de los datos originales")
            cov_matrix = np.cov(X_pca.T)
            df_cov = pd.DataFrame(cov_matrix, index=feature_cols_pca, columns=feature_cols_pca)
            st.write(df_cov)
            import seaborn as sns
            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_cov, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax_cov, annot_kws={"size":8})
            ax_cov.set_title('Heatmap matriz de covarianza (PCA)', fontsize=14)
            ax_cov.set_xticklabels(ax_cov.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax_cov.set_yticklabels(ax_cov.get_yticklabels(), rotation=0, fontsize=9)
            fig_cov.tight_layout()
            st.pyplot(fig_cov)
            st.caption("La matriz de covarianza muestra cómo varían conjuntamente las variables originales.")
            # Matriz de correlación de los datos originales
            st.write("### Matriz de correlación de los datos originales")
            corr_matrix = np.corrcoef(X_pca.T)
            df_corr = pd.DataFrame(corr_matrix, index=feature_cols_pca, columns=feature_cols_pca)
            st.write(df_corr)
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr, center=0, annot_kws={"size":8})
            ax_corr.set_title('Heatmap matriz de correlación (PCA)', fontsize=14)
            ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=9)
            fig_corr.tight_layout()
            st.pyplot(fig_corr)
            st.caption("La matriz de correlación muestra la relación lineal entre variables, normalizada entre -1 y 1.")
            # Gráfico de barras de varianza explicada
            st.write("### Varianza explicada por cada componente")
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(range(1, n_comp+1), var_exp, color='dodgerblue')
            ax_bar.set_xlabel('Componente principal')
            ax_bar.set_ylabel('Varianza explicada')
            ax_bar.set_title('Varianza explicada por componente')
            st.pyplot(fig_bar)

            st.write("### Varianza acumulada por componente")
            var_acum = np.cumsum(var_exp)
            fig_line, ax_line = plt.subplots()
            ax_line.plot(range(1, n_comp+1), var_acum, color='red', marker='o')
            ax_line.set_xlabel('Componente principal')
            ax_line.set_ylabel('Varianza acumulada')
            ax_line.set_title('Varianza acumulada por componente')
            ax_line.set_ylim(0, 1.05)
            st.pyplot(fig_line)
            # Selección dinámica de componentes para visualización
            st.write("### Visualización de componentes principales seleccionados")
            comp_options = [f"PC{i+1}" for i in range(n_comp)]
            comp_x = st.selectbox("Componente para eje X", comp_options, index=0)
            comp_y = st.selectbox("Componente para eje Y", comp_options, index=1 if n_comp > 1 else 0)
            comp_z = None
            if n_comp >= 3:
                comp_z = st.selectbox("Componente para eje Z (opcional, para gráfico 3D)", ["(Ninguno)"] + comp_options, index=0)
            idx_x = comp_options.index(comp_x)
            idx_y = comp_options.index(comp_y)
            idx_z = comp_options.index(comp_z) if comp_z and comp_z != "(Ninguno)" else None
            # Gráfico 2D o 3D según selección
            if idx_z is not None:
                from mpl_toolkits.mplot3d import Axes3D
                fig_pca = plt.figure()
                ax_pca = fig_pca.add_subplot(111, projection='3d')
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    clases = pd.Categorical(df[class_col_pca]).categories
                    codigos = pd.Categorical(df[class_col_pca]).codes
                    cmap = plt.get_cmap('tab10') if len(clases) <= 10 else plt.get_cmap('tab20')
                    for idx, clase in enumerate(clases):
                        puntos = codigos == idx
                        ax_pca.scatter(X_proj[puntos, idx_x], X_proj[puntos, idx_y], X_proj[puntos, idx_z], label=str(clase), color=cmap(idx))
                    ax_pca.legend(title=class_col_pca)
                else:
                    ax_pca.scatter(X_proj[:, idx_x], X_proj[:, idx_y], X_proj[:, idx_z], color='dodgerblue')
                ax_pca.set_xlabel(f"{comp_x}")
                ax_pca.set_ylabel(f"{comp_y}")
                ax_pca.set_zlabel(f"{comp_z}")
                ax_pca.set_title(f"PCA - Proyección {comp_x} vs {comp_y} vs {comp_z}")
                st.pyplot(fig_pca)
            else:
                fig_pca, ax_pca = plt.subplots()
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    clases = pd.Categorical(df[class_col_pca]).categories
                    codigos = pd.Categorical(df[class_col_pca]).codes
                    cmap = plt.get_cmap('tab10') if len(clases) <= 10 else plt.get_cmap('tab20')
                    for idx, clase in enumerate(clases):
                        puntos = codigos == idx
                        ax_pca.scatter(X_proj[puntos, idx_x], X_proj[puntos, idx_y], label=str(clase), color=cmap(idx))
                    ax_pca.legend(title=class_col_pca)
                else:
                    ax_pca.scatter(X_proj[:, idx_x], X_proj[:, idx_y], color='dodgerblue')
                ax_pca.set_xlabel(f"{comp_x}")
                ax_pca.set_ylabel(f"{comp_y}")
                ax_pca.set_title(f"PCA - Proyección {comp_x} vs {comp_y}")
                st.pyplot(fig_pca)
            # Explicación automática
            with st.expander("¿Cómo interpretar los componentes principales?"):
                st.markdown("""
                - Los componentes principales son combinaciones lineales de las variables originales.
                - El primer componente explica la mayor varianza posible, el segundo la siguiente mayor, y así sucesivamente.
                - Si la varianza acumulada es alta (>0.8), los datos pueden representarse bien en menos dimensiones.
                - El gráfico de barras ayuda a decidir cuántos componentes usar.
                - La matriz de componentes muestra el peso de cada variable en cada componente.
                - Las matrices de covarianza y correlación ayudan a entender la relación entre variables originales.
                """)
