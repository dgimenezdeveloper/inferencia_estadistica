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
        # La sección completa de PCA (explicaciones, selección de variables y visualizaciones)
        # está implementada más abajo y usa las variables que se definen después de seleccionar las columnas para PCA.
        # Aquí solo se marca la rama para mostrar la sección correspondiente.
        st.header("Reducción de dimensiones: PCA")
        with st.expander("¿Qué es PCA? (Explicación visual y sencilla)"):
            st.markdown("""
            **PCA (Análisis de Componentes Principales)** es una técnica que transforma tus variables originales en nuevas variables llamadas *componentes principales*.
            
            - Cada componente principal es una combinación de las variables originales.
            - El **primer componente principal (PC1)** es la dirección donde los datos varían más.
            - El **segundo componente principal (PC2)** es la siguiente dirección de máxima variabilidad, perpendicular a la primera.
            - Así sucesivamente para los demás componentes.
            
            **¿Para qué sirve?**
            - Para reducir la cantidad de variables y simplificar el análisis.
            - Para visualizar datos multidimensionales en 2D o 3D.
            - Para eliminar redundancia si algunas variables están correlacionadas.
            
            **Ejemplo visual:**
            Si tienes datos de vinos con 10 características, PCA puede crear 2 componentes principales que expliquen el 80% de la variabilidad. Así puedes graficar y analizar los vinos en solo 2 dimensiones, sin perder casi nada de información.
            
            **¿En qué se basa?**
            - PCA calcula la matriz de covarianza de los datos y encuentra las direcciones (componentes) donde los datos varían más.
            - Cada componente tiene un porcentaje de varianza explicada: indica cuánta información conserva ese componente.
            
            **Interpretación de los porcentajes:**
            - Si el primer componente explica el 60% y el segundo el 20%, juntos explican el 80% de la variabilidad de los datos.
            - Puedes decidir cuántos componentes usar según la varianza acumulada.
            """)
        num_cols_pca = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        class_col_pca = st.selectbox("Columna de clase (opcional, para colorear)", ["(Ninguna)"] + [c for c in df.columns if df[c].nunique() <= 20 and c != "(Ninguna)"])
        feature_cols_pca = st.multiselect("Selecciona las columnas numéricas para PCA:", [c for c in num_cols_pca if c != class_col_pca], default=[c for c in num_cols_pca if c != class_col_pca])
        varianza_deseada = st.slider("Porcentaje mínimo de varianza acumulada a conservar", min_value=50, max_value=99, value=80, step=1)
        if feature_cols_pca:
            st.info("Las siglas 'PC' significan 'Principal Component' o 'Componente Principal'. Por ejemplo, PC1 es el primer componente principal, PC2 el segundo, y así sucesivamente. Cada uno es una combinación lineal de las variables originales que explica una parte de la varianza total del dataset.")
            X_pca = df[feature_cols_pca]
            X_pca_scaled = StandardScaler().fit_transform(X_pca)
            n_samples, n_features = X_pca_scaled.shape
            max_components = min(n_samples, n_features)
            # Calcula el número de componentes necesarios para alcanzar el porcentaje deseado
            pca_full = PCA(n_components=max_components)
            pca_full.fit(X_pca_scaled)
            var_exp_full = pca_full.explained_variance_ratio_
            var_acum_full = np.cumsum(var_exp_full)
            n_comp_auto = np.argmax(var_acum_full >= varianza_deseada/100) + 1
            st.write(f"Se requieren **{n_comp_auto}** componentes principales para alcanzar al menos {varianza_deseada}% de varianza acumulada.")
            n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=max_components, value=n_comp_auto)
            if max_components < n_comp:
                st.error(f"No se puede aplicar PCA con {n_comp} componentes. El dataset tiene solo {n_samples} muestra(s) y {n_features} característica(s). El número de componentes debe ser menor o igual a {max_components}.")
            else:
                pca = PCA(n_components=n_comp)
                X_proj = pca.fit_transform(X_pca_scaled)
                var_exp = pca.explained_variance_ratio_
                st.write("#### Porcentaje de varianza explicada por cada componente principal:")
                var_exp_pct = [f"PC{i+1}: {v*100:.2f}%" for i, v in enumerate(var_exp)]
                st.write(", ".join(var_exp_pct))
                st.write(f"**Varianza acumulada:** {np.sum(var_exp)*100:.2f}%")
                # Ejemplo de PCA sin escalado y comparación visual
                with st.expander("Comparación: PCA con y sin escalado de variables"):
                    st.markdown("""
                    Este ejemplo muestra la diferencia entre aplicar PCA con variables escaladas (recomendado) y sin escalar (como en el notebook de clase).
                    Si las variables tienen diferentes unidades o rangos, el PCA sin escalado puede estar dominado por las variables de mayor magnitud.
                    """)
                    if n_comp >= 2:
                        # PCA sin escalado
                        pca_noesc = PCA(n_components=n_comp)
                        X_noesc = X_pca.values
                        X_proj_noesc = pca_noesc.fit_transform(X_noesc)
                        var_exp_noesc = pca_noesc.explained_variance_ratio_
                        # Visualización comparativa de proyecciones
                        fig_comp, ax_comp = plt.subplots(1, 2, figsize=(12, 5))
                        # Con escalado
                        ax_comp[0].scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.6, color='dodgerblue')
                        ax_comp[0].set_title('PCA con escalado (PC1 vs PC2)')
                        ax_comp[0].set_xlabel('PC1')
                        ax_comp[0].set_ylabel('PC2')
                        # Sin escalado
                        ax_comp[1].scatter(X_proj_noesc[:, 0], X_proj_noesc[:, 1], alpha=0.6, color='orange')
                        ax_comp[1].set_title('PCA sin escalado (PC1 vs PC2)')
                        ax_comp[1].set_xlabel('PC1')
                        ax_comp[1].set_ylabel('PC2')
                        fig_comp.tight_layout()
                        st.pyplot(fig_comp)
                        st.caption("A la izquierda: PCA con escalado (StandardScaler). A la derecha: PCA sin escalado. Observa cómo cambia la orientación y agrupación de los datos.")
                        # Comparación de varianza explicada
                        st.write("#### Varianza explicada por los dos primeros componentes:")
                        st.write(f"Con escalado: PC1 = {var_exp[0]*100:.2f}%, PC2 = {var_exp[1]*100:.2f}%")
                        st.write(f"Sin escalado: PC1 = {var_exp_noesc[0]*100:.2f}%, PC2 = {var_exp_noesc[1]*100:.2f}%")
                        st.caption("El escalado permite que todas las variables aporten por igual al análisis. Sin escalado, las variables de mayor rango dominan el resultado.")
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
            st.info("La matriz de componentes principales te muestra cómo se construyen los nuevos ejes (componentes) a partir de tus variables originales. Los valores indican la importancia de cada variable en cada componente.")
            # Interpretación automática de cada componente principal
            st.write("### Interpretación automática de los componentes principales")
            for i, row in enumerate(pca.components_):
                pesos = np.abs(row)
                top_idx = np.argsort(pesos)[::-1][:3]  # top 3 variables
                top_vars = [(feature_cols_pca[j], row[j]) for j in top_idx]
                partes = []
                for var, peso in top_vars:
                    sentido = "+" if peso > 0 else "-"
                    partes.append(f"{var} ({sentido})")
                explicacion = f"PC{i+1} está principalmente influenciado por: " + ", ".join(partes)
                st.markdown(f"- {explicacion}")

            # Visualización de flechas de componentes principales sobre los datos originales
            with st.expander("Visualización: Flechas de componentes principales sobre los datos originales"):
                st.markdown("""
                Este gráfico muestra los datos originales en dos variables seleccionadas y las direcciones de los dos primeros componentes principales (PC1 y PC2) como flechas.
                Permite ver cómo se orientan los ejes principales respecto a las variables originales.
                """)
                if len(feature_cols_pca) >= 2:
                    var_x = st.selectbox("Variable para eje X", feature_cols_pca, index=0)
                    var_y = st.selectbox("Variable para eje Y", [v for v in feature_cols_pca if v != var_x], index=0)
                    fig_arrow, ax_arrow = plt.subplots(figsize=(7, 6))
                    # Graficar puntos
                    ax_arrow.scatter(X_pca[var_x], X_pca[var_y], alpha=0.6, color='gray', label='Datos')
                    # Calcular centro
                    x_mean = X_pca[var_x].mean()
                    y_mean = X_pca[var_y].mean()
                    # Flechas de PC1 y PC2
                    idx_x = feature_cols_pca.index(var_x)
                    idx_y = feature_cols_pca.index(var_y)
                    # PC1
                    ax_arrow.arrow(x_mean, y_mean, pca.components_[0, idx_x]*5, pca.components_[0, idx_y]*5, width=0.05, color="purple", label="PC1")
                    # PC2
                    ax_arrow.arrow(x_mean, y_mean, pca.components_[1, idx_x]*5, pca.components_[1, idx_y]*5, width=0.05, color="black", label="PC2")
                    ax_arrow.set_xlabel(var_x)
                    ax_arrow.set_ylabel(var_y)
                    ax_arrow.set_title(f"Flechas de PC1 y PC2 sobre {var_x} vs {var_y}")
                    ax_arrow.legend(["Datos", "PC1", "PC2"])
                    fig_arrow.tight_layout()
                    st.pyplot(fig_arrow)
                    st.caption("Las flechas muestran la dirección y sentido de los dos primeros componentes principales en el plano de las variables seleccionadas.")
            # Matriz de covarianza de los datos originales
            st.write("### Matriz de covarianza de los datos originales")
            cov_matrix = np.cov(X_pca.T)
            df_cov = pd.DataFrame(cov_matrix, index=feature_cols_pca, columns=feature_cols_pca)
            st.write(df_cov)
            st.caption("La matriz de covarianza muestra cómo varían conjuntamente las variables originales. Valores altos indican que dos variables tienden a aumentar o disminuir juntas.")
            st.info("¿Para qué sirve? Permite identificar relaciones y dependencias entre variables. Es la base matemática de PCA y ayuda a detectar variables redundantes o muy correlacionadas.")
            import seaborn as sns
            fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_cov, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax_cov, annot_kws={"size":8})
            ax_cov.set_title('Heatmap matriz de covarianza (PCA)', fontsize=14)
            ax_cov.set_xticklabels(ax_cov.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax_cov.set_yticklabels(ax_cov.get_yticklabels(), rotation=0, fontsize=9)
            fig_cov.tight_layout()
            st.pyplot(fig_cov)
            # Matriz de correlación de los datos originales
            st.write("### Matriz de correlación de los datos originales")
            corr_matrix = np.corrcoef(X_pca.T)
            df_corr = pd.DataFrame(corr_matrix, index=feature_cols_pca, columns=feature_cols_pca)
            st.write(df_corr)
            st.caption("La matriz de correlación muestra la relación lineal entre variables, normalizada entre -1 y 1. Valores cercanos a +1 o -1 indican fuerte relación positiva o negativa.")
            st.info("¿Para qué sirve? Permite identificar variables que están muy relacionadas (redundantes) y ayuda a decidir qué variables pueden ser eliminadas o combinadas.")
            # Recomendación automática de variables redundantes
            threshold = 0.85
            redundantes = []
            for i in range(len(feature_cols_pca)):
                for j in range(i+1, len(feature_cols_pca)):
                    corr = df_corr.iloc[i, j]
                    if abs(corr) > threshold:
                        redundantes.append((feature_cols_pca[i], feature_cols_pca[j], corr))
            if redundantes:
                st.warning("Variables potencialmente redundantes detectadas (correlación > 0.85 o < -0.85):")
                for var1, var2, corr in redundantes:
                    st.write(f"- **{var1}** y **{var2}** (correlación: {corr:.2f})")
                st.caption("Puedes considerar eliminar o combinar estas variables para simplificar el análisis, ya que aportan información muy similar.")
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax_corr, center=0, annot_kws={"size":8})
            ax_corr.set_title('Heatmap matriz de correlación (PCA)', fontsize=14)
            ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=9)
            fig_corr.tight_layout()
            st.pyplot(fig_corr)
            # Gráfico de barras de varianza explicada
            st.write("### Varianza explicada por cada componente")
            st.caption("La varianza explicada indica cuánta información conserva cada componente principal. Componentes con mayor varianza explicada son más importantes para representar los datos.")
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(range(1, n_comp+1), [v*100 for v in var_exp], color='dodgerblue')
            ax_bar.set_xlabel('Componente principal')
            ax_bar.set_ylabel('Varianza explicada (%)')
            ax_bar.set_title('Porcentaje de varianza explicada por componente')
            for i, v in enumerate(var_exp):
                ax_bar.text(i+1, v*100, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=9)
            st.pyplot(fig_bar)

            st.write("### Varianza acumulada por componente")
            var_acum = np.cumsum(var_exp)
            fig_line, ax_line = plt.subplots()
            x_vals = range(1, n_comp+1)
            ax_line.plot(x_vals, var_acum, color='red', marker='o')
            ax_line.set_xlabel('Componente principal')
            ax_line.set_ylabel('Varianza acumulada')
            ax_line.set_title('Varianza acumulada por componente')
            ax_line.set_ylim(0, 1.05)
            # Método del codo mejorado: siempre muestra la línea azul y el mensaje
            difs = np.diff(var_acum)
            codo_idx = None
            for i, d in enumerate(difs):
                if d < 0.02:
                    codo_idx = i + 1
                    break
            if codo_idx is None:
                codo_idx = len(var_acum) - 1  # último componente si no hay codo claro
            ax_line.axvline(codo_idx+1, color='blue', linestyle='--', label=f'Recomendado: {codo_idx+1} componentes')
            ax_line.legend()
            st.info(f"Recomendación automática: El método del codo sugiere usar **{codo_idx+1}** componentes principales. A partir de aquí, el incremento de varianza explicada es menor a 2%. Puedes ajustar según tu objetivo.")
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
            import plotly.express as px
            if idx_z is not None:
                # Mejor visualización 3D interactiva con Plotly
                import plotly.graph_objects as go
                hover_text = []
                for i in range(X_proj.shape[0]):
                    txt = f"{comp_x}: {X_proj[i, idx_x]:.2f}<br>{comp_y}: {X_proj[i, idx_y]:.2f}<br>{comp_z}: {X_proj[i, idx_z]:.2f}"
                    if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                        txt += f"<br>{class_col_pca}: {df[class_col_pca].iloc[i]}"
                    hover_text.append(txt)
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    clases = pd.Categorical(df[class_col_pca]).categories
                    codigos = pd.Categorical(df[class_col_pca]).codes
                    colors = px.colors.qualitative.Safe if len(clases) <= 10 else px.colors.qualitative.Light24
                    data = []
                    for idx, clase in enumerate(clases):
                        puntos = codigos == idx
                        data.append(go.Scatter3d(
                            x=X_proj[puntos, idx_x],
                            y=X_proj[puntos, idx_y],
                            z=X_proj[puntos, idx_z],
                            mode='markers',
                            marker=dict(size=7, opacity=0.7, color=colors[idx % len(colors)]),
                            name=str(clase),
                            text=[hover_text[i] for i in range(len(hover_text)) if puntos[i]],
                            hoverinfo='text'
                        ))
                else:
                    data = [go.Scatter3d(
                        x=X_proj[:, idx_x],
                        y=X_proj[:, idx_y],
                        z=X_proj[:, idx_z],
                        mode='markers',
                        marker=dict(size=7, opacity=0.7, color='dodgerblue'),
                        name='Datos',
                        text=hover_text,
                        hoverinfo='text'
                    )]
                layout = go.Layout(
                    title=f"PCA - Proyección {comp_x} vs {comp_y} vs {comp_z}",
                    scene=dict(
                        xaxis_title=comp_x,
                        yaxis_title=comp_y,
                        zaxis_title=comp_z,
                        bgcolor='white',
                        xaxis=dict(showbackground=True, backgroundcolor='white', gridcolor='lightgray'),
                        yaxis=dict(showbackground=True, backgroundcolor='white', gridcolor='lightgray'),
                        zaxis=dict(showbackground=True, backgroundcolor='white', gridcolor='lightgray'),
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                fig_pca = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig_pca, use_container_width=True)
                st.caption("Interactúa con el gráfico: puedes rotar, hacer zoom y ver detalles de cada punto. Cada color representa una clase (si está seleccionada). Los valores de los ejes corresponden a los componentes principales seleccionados.")
            else:
                # Gráfico 2D interactivo con Plotly
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    fig_pca = px.scatter(
                        x=X_proj[:, idx_x],
                        y=X_proj[:, idx_y],
                        color=df[class_col_pca].astype(str),
                        labels={
                            "x": comp_x,
                            "y": comp_y,
                            "color": class_col_pca
                        },
                        title=f"PCA - Proyección {comp_x} vs {comp_y}"
                    )
                else:
                    fig_pca = px.scatter(
                        x=X_proj[:, idx_x],
                        y=X_proj[:, idx_y],
                        labels={
                            "x": comp_x,
                            "y": comp_y
                        },
                        title=f"PCA - Proyección {comp_x} vs {comp_y}"
                    )
                st.plotly_chart(fig_pca, use_container_width=True)
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
            # Panel FAQ interactivo
            with st.expander("Preguntas frecuentes sobre PCA y matrices"):
                st.markdown("""
                **¿Por qué elegir PCA para mi análisis?**
                - PCA te ayuda a reducir la cantidad de variables, eliminar redundancia y visualizar datos complejos en menos dimensiones.

                **¿Qué significa 'varianza explicada'?**
                - Es el porcentaje de información que conserva cada componente principal respecto a los datos originales.

                **¿Cómo interpreto la matriz de componentes principales?**
                - Cada fila es un componente, cada columna es una variable original. Los valores indican la importancia de cada variable en ese componente.

                **¿Para qué sirve la matriz de covarianza?**
                - Permite ver cómo varían juntas las variables. Es útil para detectar relaciones y redundancias.

                **¿Y la matriz de correlación?**
                - Muestra la relación lineal entre variables, normalizada entre -1 y 1. Ayuda a identificar variables muy relacionadas.

                **¿Cuántos componentes debería elegir?**
                - Elige suficientes componentes para explicar al menos el 80% de la varianza acumulada, pero depende de tu objetivo.

                **¿Puedo usar PCA si tengo variables categóricas?**
                - No directamente. PCA requiere variables numéricas. Convierte las categóricas en numéricas si es necesario.
                """)
