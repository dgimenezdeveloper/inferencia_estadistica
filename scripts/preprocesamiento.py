def mostrar_ejemplos_por_clase(df, target_col, clase_labels_global, st, n=3):
    """
    Muestra ejemplos de filas para cada clase usando Streamlit.
    """
    clase_unicos = sorted(df[target_col].unique())
    for v in clase_unicos:
        nombre_clase = clase_labels_global.get(v, str(v))
        st.caption(f"Ejemplos para la clase '{nombre_clase}':")
        st.dataframe(df[df[target_col] == v].head(n), width='stretch')
def detectar_variables_redundantes(df_corr, feature_names, threshold=0.85):
    """
    Devuelve una lista de tuplas (var1, var2, corr) para pares de variables con |correlación| > threshold.
    """
    redundantes = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = df_corr.iloc[i, j]
            if abs(corr) > threshold:
                redundantes.append((feature_names[i], feature_names[j], corr))
    return redundantes
# Archivo de preprocesamiento de datos para la app integrada

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cargar_dataset(archivo_subido, opcion_archivo, carpeta_datos):
    """
    Carga un DataFrame desde un archivo subido o desde la carpeta de datos.
    """
    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        return df, 'Archivo subido cargado correctamente.'
    elif opcion_archivo:
        ruta = os.path.join(carpeta_datos, opcion_archivo)
        df = pd.read_csv(ruta)
        return df, f"Archivo '{opcion_archivo}' cargado correctamente."
    return None, None

def seleccionar_columnas(df, max_unique_target=20):
    """
    Devuelve listas de columnas numéricas y categóricas elegibles para target.
    """
    columnas = df.columns.tolist()
    num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in columnas if (
        pd.api.types.is_integer_dtype(df[c]) or
        pd.api.types.is_bool_dtype(df[c]) or
        pd.api.types.is_categorical_dtype(df[c]) or
        pd.api.types.is_object_dtype(df[c])
    ) and df[c].nunique() <= max_unique_target]
    return num_cols, cat_cols

def manejar_nulos(X, metodo='media'):
    """
    Imputa valores nulos en X usando la media (por defecto) o mediana.
    """
    if metodo == 'media':
        return X.fillna(X.mean())
    elif metodo == 'mediana':
        return X.fillna(X.median())
    else:
        raise ValueError('Método de imputación no soportado')

def escalar_datos(X):
    """
    Escala los datos usando StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def aplicar_pca(X_scaled, varianza_min=0.8):
    """
    Aplica PCA y retorna la proyección, el modelo PCA y la varianza explicada.
    """
    n_samples, n_features = X_scaled.shape
    max_components = min(n_samples, n_features)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X_scaled)
    var_exp_full = pca_full.explained_variance_ratio_
    var_acum_full = np.cumsum(var_exp_full)
    n_comp_auto = np.argmax(var_acum_full >= varianza_min) + 1
    pca = PCA(n_components=n_comp_auto)
    X_proj = pca.fit_transform(X_scaled)
    return X_proj, pca, pca.explained_variance_ratio_, n_comp_auto
