import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Tamaño estándar para todos los gráficos
DEFAULT_FIGSIZE = (5, 4)
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

st.set_page_config(page_title="Bayes Ingenuo - Interfaz Web", layout="wide")
st.title("Clasificador Bayes Ingenuo - Visualización y Explicación")

st.markdown("""
Esta aplicación permite cargar un archivo CSV, ejecutar el algoritmo de Bayes Ingenuo, visualizar resultados y obtener explicaciones automáticas.
""")

# Limitar el ancho máximo de los gráficos con CSS
st.markdown("""
<style>
.element-container img, .element-container svg {
    max-width: 500px !important;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)

# Subida de archivo
data_file = st.file_uploader("Sube tu archivo CSV de datos", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)
    st.subheader("Vista previa de los datos:")
    st.dataframe(df.head())

    # Mostrar valores faltantes
    st.write("**Valores faltantes por columna:**")
    st.write(df.isnull().sum())

    # Imputar NaN con la media
    if df.isnull().values.any():
        st.warning("Se encontraron valores faltantes. Imputando con la media de cada columna...")
        df = df.fillna(df.mean(numeric_only=True))
        st.success("Valores faltantes imputados.")

    # Filtrar solo columnas categóricas (tipo int y con pocos valores únicos)
    max_unique = 10  # Puedes ajustar este umbral
    cat_cols = [c for c in df.columns if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])) and df[c].nunique() <= max_unique]
    if not cat_cols:
        st.error("No se encontraron columnas categóricas adecuadas para usar como variable de clase (target). Asegúrate de tener una columna tipo entero con pocos valores únicos, como 0 y 1.")
    else:
        clase = st.selectbox("Selecciona la columna de la variable de clase (target):", cat_cols, index=0)
        columnas = df.columns.tolist()
        atributos = st.multiselect("Selecciona las columnas de atributos (features):", [c for c in columnas if c != clase], default=[c for c in columnas if c != clase])

        if atributos and clase:
            X = df[atributos]
            y = df[clase]

            # Mostrar balance de clases
            st.write("**Balance de clases (target):**")
            class_counts = y.value_counts()
            st.bar_chart(class_counts)
            if class_counts.min() / class_counts.max() < 0.5:
                st.warning("¡Tus datos están desbalanceados! Considera balancear las clases para mejorar la sensibilidad del modelo. Puedes usar técnicas como sobremuestreo, submuestreo o el parámetro class_weight en otros modelos.")

            test_size = st.slider("Proporción de test", 0.1, 0.5, 0.3)
            st.write("**Tamaño de los gráficos (pulgadas):**")
            fig_width = st.slider("Ancho", 2, 6, 3)
            fig_height = st.slider("Alto", 2, 6, 2)

            # Opciones de balanceo
            st.write("**¿Quieres balancear las clases antes de entrenar el modelo?**")
            balanceo = st.selectbox("Método de balanceo", ["Ninguno (original)", "Sobremuestreo clase minoritaria", "Submuestreo clase mayoritaria"])

            # Separar en train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Balanceo interactivo
            if balanceo == "Sobremuestreo clase minoritaria":
                from sklearn.utils import resample
                # Concatenar X_train y y_train para balancear
                train = pd.concat([X_train, y_train], axis=1)
                clase_min = train[clase].value_counts().idxmin()
                clase_maj = train[clase].value_counts().idxmax()
                df_min = train[train[clase]==clase_min]
                df_maj = train[train[clase]==clase_maj]
                df_min_upsampled = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
                train_bal = pd.concat([df_maj, df_min_upsampled])
                X_train_bal = train_bal[atributos]
                y_train_bal = train_bal[clase]
                modelo = GaussianNB()
                modelo.fit(X_train_bal, y_train_bal)
                y_pred = modelo.predict(X_test)
                st.info(f"Se aplicó sobremuestreo: ahora ambas clases tienen {len(df_maj)} ejemplos en el entrenamiento.")
            elif balanceo == "Submuestreo clase mayoritaria":
                from sklearn.utils import resample
                train = pd.concat([X_train, y_train], axis=1)
                clase_min = train[clase].value_counts().idxmin()
                clase_maj = train[clase].value_counts().idxmax()
                df_min = train[train[clase]==clase_min]
                df_maj = train[train[clase]==clase_maj]
                df_maj_downsampled = resample(df_maj, replace=False, n_samples=len(df_min), random_state=42)
                train_bal = pd.concat([df_min, df_maj_downsampled])
                X_train_bal = train_bal[atributos]
                y_train_bal = train_bal[clase]
                modelo = GaussianNB()
                modelo.fit(X_train_bal, y_train_bal)
                y_pred = modelo.predict(X_test)
                st.info(f"Se aplicó submuestreo: ahora ambas clases tienen {len(df_min)} ejemplos en el entrenamiento.")
            else:
                modelo = GaussianNB()
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                st.info("Entrenamiento con datos originales (sin balancear).")

            st.subheader("Resultados del Modelo")
            st.write(f"**Exactitud:** {accuracy_score(y_test, y_pred):.3f}")

            # Matriz de confusión
            st.write("**Matriz de confusión:**")
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            st.pyplot(fig)
            st.caption("La matriz de confusión muestra cuántos casos fueron correctamente clasificados y cuántos fueron errores. Es fundamental para evaluar el rendimiento de Bayes Ingenuo.")

            # Reporte de clasificación
            st.write("**Reporte de clasificación:**")
            st.text(classification_report(y_test, y_pred))
            st.caption("El reporte de clasificación incluye métricas como precisión, recall y F1-score, que permiten analizar la calidad de las predicciones del modelo.")

            # Gráficos de distribución de atributos por clase
            st.subheader("Distribución de atributos por clase")
            for atributo in atributos:
                fig_attr, ax_attr = plt.subplots(figsize=(fig_width, fig_height))
                if pd.api.types.is_numeric_dtype(df[atributo]):
                    for clase_val in sorted(df[clase].unique()):
                        sns.histplot(df[df[clase]==clase_val][atributo], label=f"Clase {clase_val}", kde=True, ax=ax_attr, bins=20, alpha=0.6)
                    ax_attr.set_title(f"Distribución de '{atributo}' por clase")
                    ax_attr.set_xlabel(atributo)
                    ax_attr.set_ylabel("Frecuencia")
                    ax_attr.legend()
                    st.pyplot(fig_attr)
                    st.caption(f"Este gráfico muestra cómo se distribuye el atributo '{atributo}' en cada clase. Si las distribuciones son diferentes, Bayes Ingenuo puede distinguir mejor entre clases usando este atributo.")
                else:
                    sns.countplot(x=atributo, hue=clase, data=df, ax=ax_attr)
                    ax_attr.set_title(f"Conteo de '{atributo}' por clase")
                    st.pyplot(fig_attr)
                    st.caption(f"Este gráfico muestra el conteo de cada valor del atributo '{atributo}' según la clase. Es útil para atributos categóricos.")

            # Explicación automática
            st.subheader("Explicación de los resultados")
            st.markdown("""
**Exactitud:** Proporción de aciertos sobre el total de casos.
**Matriz de confusión:**
    - Diagonal: aciertos.
    - Fuera de la diagonal: errores de predicción.
**Precision:** De los que predijo como positivos, ¿cuántos lo son realmente?
**Recall:** De los positivos reales, ¿cuántos detectó?
**F1-score:** Media armónica entre precision y recall.
            """)
            # Explicación personalizada según resultados
            if confusion_matrix(y_test, y_pred).shape == (2,2):
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                st.write(f"El modelo acertó {tn+tp} de {tn+fp+fn+tp} casos. Hay {fp} falsos positivos y {fn} falsos negativos.")
                sensibilidad = tp/(tp+fn+1e-9)
                if sensibilidad < 0.5:
                    st.warning("El modelo tiene baja sensibilidad para la clase positiva. Esto suele deberse a datos desbalanceados. Prueba balancear las clases, usar técnicas de muestreo o probar otros modelos más robustos.")
else:
    st.info("Sube un archivo CSV para comenzar.")
