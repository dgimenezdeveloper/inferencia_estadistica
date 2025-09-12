import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

carpeta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos')
st.write("---")
st.header("Reducción de dimensiones: Análisis de Componentes Principales (PCA)")

with st.expander("¿Qué es PCA? (Explicación rápida)"):
    st.markdown("""
El Análisis de Componentes Principales (PCA) es una técnica que permite reducir la cantidad de variables conservando la mayor información posible. Es útil para visualizar datos multidimensionales y eliminar redundancia.
    """)

# Selección de dataset para PCA
st.write("### Seleccionar archivo CSV para PCA")
archivos_csv_pca = [f for f in os.listdir(carpeta_datos) if f.endswith('.csv')] if os.path.exists(carpeta_datos) else []
opcion_archivo_pca = st.selectbox("Archivos disponibles en 'datos/':", ["Wine.csv"] + archivos_csv_pca)
archivo_subido_pca = st.file_uploader("O sube tu propio archivo CSV para PCA", type=["csv"], key="pca")

# Cargar el dataset PCA
if archivo_subido_pca is not None:
    df_pca = pd.read_csv(archivo_subido_pca)
    st.success("Archivo subido cargado correctamente.")
elif opcion_archivo_pca != "Wine.csv":
    df_pca = pd.read_csv(os.path.join(carpeta_datos, opcion_archivo_pca))
    st.success(f"Archivo '{opcion_archivo_pca}' cargado correctamente.")
else:
    try:
        df_pca = pd.read_csv(os.path.join(carpeta_datos, "Wine.csv"))
    except:
        df_pca = None

if df_pca is not None:
    st.write("### Vista previa del dataset PCA")
    st.dataframe(df_pca.head())
    # Selección de columnas numéricas
    num_cols_pca = [c for c in df_pca.columns if pd.api.types.is_numeric_dtype(df_pca[c])]
    class_col_pca = st.selectbox("Columna de clase (opcional, para colorear)", ["(Ninguna)"] + num_cols_pca)
    feature_cols_pca = st.multiselect("Selecciona las columnas numéricas para PCA:", [c for c in num_cols_pca if c != class_col_pca], default=[c for c in num_cols_pca if c != class_col_pca])
    n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=min(10, len(feature_cols_pca)), value=2)
    if feature_cols_pca:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
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



st.title("Clasificación: LDA y QDA - Carga de Dataset y Visualización Avanzada")

with st.expander("¿Qué es LDA y QDA? (Explicación teórica)"):
    st.markdown("""
**LDA (Análisis Discriminante Lineal)** y **QDA (Análisis Discriminante Cuadrático)** son algoritmos de clasificación supervisada que buscan separar clases en función de sus características.

- **LDA** asume que todas las clases tienen igual matriz de covarianza y genera fronteras lineales.
- **QDA** permite que cada clase tenga su propia matriz de covarianza y genera fronteras cuadráticas.

Ambos métodos son útiles para problemas donde las clases pueden modelarse como distribuciones normales multivariadas.
    """)

algoritmo = st.selectbox("Selecciona el algoritmo", ["LDA", "QDA"])

# Selección de dataset
st.write("### Seleccionar archivo CSV")
carpeta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos')
archivos_csv = [f for f in os.listdir(carpeta_datos) if f.endswith('.csv')] if os.path.exists(carpeta_datos) else []
opcion_archivo = st.selectbox("Archivos disponibles en 'datos/':", ["Iris (por defecto)"] + archivos_csv)
archivo_subido = st.file_uploader("O sube tu propio archivo CSV", type=["csv"])

# Cargar el dataset
if archivo_subido is not None:
    df = pd.read_csv(archivo_subido)
    st.success("Archivo subido cargado correctamente.")
elif opcion_archivo != "Iris (por defecto)":
    df = pd.read_csv(os.path.join(carpeta_datos, opcion_archivo))
    st.success(f"Archivo '{opcion_archivo}' cargado correctamente.")
else:
    from sklearn import datasets
    iris = datasets.load_iris()
    df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                     columns = iris['feature_names'] + ['target'])
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


st.write("### Vista previa del dataset")
st.dataframe(df.head())

# Explicación automática y recomendaciones según el dataset
with st.expander("Análisis automático y recomendaciones del dataset"):
    st.markdown("**Resumen estadístico:**")
    st.write(df.describe(include='all'))
    # Balance de clases
    if 'species' in df.columns or (len(df.columns) > 0 and df.columns[-1] == 'target'):
        target_col_auto = 'species' if 'species' in df.columns else 'target'
        class_counts = df[target_col_auto].value_counts()
        st.markdown(f"**Balance de clases en '{target_col_auto}':**")
        st.write(class_counts)
        if class_counts.min() / class_counts.max() < 0.5:
            st.warning("¡Advertencia! El dataset está desbalanceado. Considera técnicas de balanceo para mejorar el rendimiento del modelo.")
        elif class_counts.min() == 1:
            st.warning("¡Advertencia! Alguna clase tiene solo una muestra. Esto puede afectar la validación cruzada y el aprendizaje.")
    # NaN
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        st.warning(f"El dataset contiene {nan_count} valores faltantes (NaN). Se imputarán automáticamente en los atributos numéricos.")
    # Tamaño del dataset
    if len(df) < 30:
        st.warning("El dataset es muy pequeño. Los resultados pueden no ser representativos.")
    # Tipo de problema
    if 'species' in df.columns or (len(df.columns) > 0 and df.columns[-1] == 'target'):
        n_classes = df[target_col_auto].nunique()
        if n_classes == 2:
            st.info("Problema de clasificación binaria.")
        elif n_classes > 2:
            st.info(f"Problema de clasificación multiclase ({n_classes} clases).")


# Selección de columnas
columnas = df.columns.tolist()
num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
# Filtrar solo columnas categóricas o enteras con pocos valores únicos para target
max_unique_target = 20
cat_cols = [c for c in columnas if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() <= max_unique_target]
if not cat_cols:
    st.error("No hay columnas válidas para usar como variable de clase (target). Elige un dataset con una columna categórica o entera con pocos valores únicos.")
target_col = st.selectbox("Selecciona la columna de clase (target):", cat_cols, index=max(0, len(cat_cols)-1))
feature_cols = st.multiselect(
    "Selecciona las columnas de atributos (features):",
    [c for c in num_cols if c != target_col],
    default=[c for c in num_cols if c != target_col]
)

if feature_cols and target_col:
    X = df[feature_cols]
    y = df[target_col]

    # Imputar NaN en features numéricos
    if X.isnull().values.any():
        st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
        X = X.fillna(X.mean())

    # Ingreso manual de datos
    st.write("### Ingresa una observación para predecir la clase")
    nueva_obs = []
    for col in feature_cols:
        val = st.number_input(f"{col}", min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean()))
        nueva_obs.append(val)
    nueva_obs = [nueva_obs]

    # Entrenamiento y predicción
    if algoritmo == "LDA":
        model = LinearDiscriminantAnalysis()
    else:
        model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)
    prediccion = model.predict(nueva_obs)

    if st.button("Predecir clase"):
        st.success(f"Predicción para {nueva_obs}: {prediccion[0]}")
        st.write(f"Algoritmo usado: {algoritmo}")


    # Métrica de desempeño
    min_samples_per_class = y.value_counts().min()
    cv_splits = min(5, min_samples_per_class) if min_samples_per_class >= 2 else None
    if cv_splits and cv_splits >= 2:
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv_splits)
        st.write(f"Precisión promedio ({algoritmo}, cv={cv_splits}): {np.mean(scores):.2f}")
    else:
        st.warning("No se puede calcular validación cruzada (cross-validation) porque alguna clase tiene menos de 2 muestras.")

    # Visualización avanzada
    st.write("### Matriz de confusión")
    st.info("La matriz de confusión muestra cuántos elementos de cada clase fueron correctamente o incorrectamente clasificados. Idealmente, los valores altos deben estar en la diagonal.")
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig)

    # Opción para mostrar proyección LDA/QDA
    st.write("### Visualización de separación de clases")
    st.markdown("""
**¿Qué muestra este gráfico?**

La proyección discriminante (LDA/QDA) transforma los datos a un espacio donde las clases están lo más separadas posible. Si ves grupos bien diferenciados, el modelo está logrando separar las clases correctamente.

En el caso de QDA, si seleccionas la proyección, se usa PCA como alternativa visual.
    """)
    show_proj = st.checkbox("Mostrar gráfico de componentes discriminantes (proyección LDA/QDA)")

    if show_proj:
        st.info("La proyección LDA muestra los componentes discriminantes. Si usas QDA, se visualiza una reducción PCA como referencia.")
        # Proyección a 2D usando LDA/QDA
        try:
            if algoritmo == "LDA":
                X_proj = model.transform(X)
            else:
                # QDA no tiene método transform, usamos PCA como alternativa
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
        st.write("### Gráfico de dispersión de las clases (atributos seleccionados)")
        fig2, ax2 = plt.subplots()
        for clase in np.unique(y):
            ax2.scatter(X[y == clase][feature_cols[0]], X[y == clase][feature_cols[1]], label=str(clase))
        ax2.set_xlabel(feature_cols[0])
        ax2.set_ylabel(feature_cols[1])
        ax2.legend()
        st.pyplot(fig2)
