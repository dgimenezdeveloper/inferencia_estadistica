import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_curve, auc,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Función para calcular métricas de evaluación completas
def calcular_metricas_clasificacion(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calcula todas las métricas de evaluación para clasificación
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        y_prob: Probabilidades predichas (opcional, para ROC)
        class_names: Nombres de las clases (opcional)
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    metricas = {}
    
    # Métricas básicas
    metricas['accuracy'] = accuracy_score(y_true, y_pred)
    metricas['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metricas['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metricas['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Métricas por clase
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    # Soporte por clase (cantidad de muestras reales de cada clase)
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y_true, y_pred)
    support_per_class = np.array([(y_true == l).sum() for l in labels])
    metricas['precision_per_class'] = precision_per_class
    metricas['recall_per_class'] = recall_per_class
    metricas['f1_per_class'] = f1_per_class
    metricas['support_per_class'] = support_per_class
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    metricas['confusion_matrix'] = cm
    
    # ROC-AUC (si se proporcionan probabilidades)
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                # Clasificación binaria
                metricas['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Clasificación multiclase
                y_true_bin = label_binarize(y_true, classes=classes)
                metricas['roc_auc'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
        except:
            metricas['roc_auc'] = None
    
    # Nombres de clases
    # Usar nombres descriptivos si existen en session_state
    labels_map = st.session_state.get("clase_labels_global", {})
    if class_names is None:
        class_names = [labels_map.get(i, str(i)) for i in np.unique(y_true)]
    else:
        class_names = [labels_map.get(i, str(i)) for i in class_names]
    metricas['class_names'] = class_names
    
    return metricas

def mostrar_metricas_clasificacion(metricas, titulo="Métricas de Evaluación"):
    """
    Muestra las métricas de clasificación en Streamlit de forma organizada
    """
    st.write(f"### {titulo}")
    
    # Métricas generales en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metricas['accuracy']:.3f}")
    with col2:
        st.metric("Precision (Macro)", f"{metricas['precision_macro']:.3f}")
    with col3:
        st.metric("Recall (Macro)", f"{metricas['recall_macro']:.3f}")
    with col4:
        st.metric("F1-Score (Macro)", f"{metricas['f1_macro']:.3f}")
    
    if metricas.get('roc_auc') is not None:
        st.metric("ROC-AUC", f"{metricas['roc_auc']:.3f}")
    
    # Tabla detallada por clase
    st.write("#### Métricas por clase")
    
    class_labels_global = st.session_state.get("clase_labels_global", {})
    nombres_clase = [class_labels_global.get(c, str(c)) for c in metricas['class_names']]

    tabla_metricas = pd.DataFrame({
        'Clase': nombres_clase,
        'Precision': metricas['precision_per_class'],
        'Recall': metricas['recall_per_class'],
        'F1-Score': metricas['f1_per_class'],
        'Support': metricas.get('support_per_class', [np.nan]*len(nombres_clase))
    })
    # Formatear valores
    for c in ['Precision','Recall','F1-Score']:
        tabla_metricas[c] = tabla_metricas[c].apply(lambda x: float(x) if isinstance(x, (float, np.floating, int, np.integer)) else np.nan)
    # Formato condicional pastel solo para las métricas, no para Support
    def pastel_metric(val):
        if pd.isnull(val) or val == 0:
            return 'background-color: #ffffff; color: #111; font-weight: bold;'
        if val >= 0.8:
            color = '#81c784'  # verde pastel saturado
        elif val >= 0.6:
            color = '#fff176'  # amarillo pastel saturado
        elif val > 0.0:
            color = '#e57373'  # rojo pastel saturado
        else:
            color = '#ffffff'
        return f'background-color: {color}; color: #111; font-weight: bold;'
    styled = tabla_metricas.style.applymap(pastel_metric, subset=['Precision','Recall','F1-Score'])\
        .format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-Score': '{:.3f}', 'Support': '{:.0f}'})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Glosario de colores para la tabla de métricas por clase
    st.markdown("""
<div style='margin: 0.5em 0 1em 0; font-size: 0.95em;'>
<strong>Glosario de colores:</strong><br>
<span style='background-color:#81c784; color:#111; padding:2px 8px; border-radius:4px;'>🟩 Verde</span> = valor alto (≥ 0.8, buen desempeño)<br>
<span style='background-color:#fff176; color:#111; padding:2px 8px; border-radius:4px;'>🟨 Amarillo</span> = valor medio (≥ 0.6, aceptable)<br>
<span style='background-color:#e57373; color:#111; padding:2px 8px; border-radius:4px;'>🟥 Rojo</span> = valor bajo (> 0.0, necesita mejora)<br>
<span style='background-color:#fff; color:#111; padding:2px 8px; border-radius:4px;'>⬜ Blanco</span> = cero o nulo
</div>
""", unsafe_allow_html=True)

    # Interpretaciones automáticas
    with st.expander("💡 Interpretación de las métricas"):
        st.markdown(f"""
        **Accuracy ({metricas['accuracy']:.3f})**: Proporción de predicciones correctas del total.
        {'✅ Excelente' if metricas['accuracy'] > 0.9 else '✅ Buena' if metricas['accuracy'] > 0.8 else '⚠️ Regular' if metricas['accuracy'] > 0.7 else '❌ Necesita mejora'}
        
        **Precision (Macro) ({metricas['precision_macro']:.3f})**: Promedio de precisión por clase. 
        Indica qué tan exactas son las predicciones positivas.
        {'✅ Excelente' if metricas['precision_macro'] > 0.9 else '✅ Buena' if metricas['precision_macro'] > 0.8 else '⚠️ Regular' if metricas['precision_macro'] > 0.7 else '❌ Necesita mejora'}
        
        **Recall (Macro) ({metricas['recall_macro']:.3f})**: Promedio de sensibilidad por clase.
        Indica qué tan bien el modelo encuentra los casos positivos reales.
        {'✅ Excelente' if metricas['recall_macro'] > 0.9 else '✅ Buena' if metricas['recall_macro'] > 0.8 else '⚠️ Regular' if metricas['recall_macro'] > 0.7 else '❌ Necesita mejora'}
        
        **F1-Score (Macro) ({metricas['f1_macro']:.3f})**: Media armónica entre precision y recall.
        Balancea ambas métricas.
        {'✅ Excelente' if metricas['f1_macro'] > 0.9 else '✅ Buena' if metricas['f1_macro'] > 0.8 else '⚠️ Regular' if metricas['f1_macro'] > 0.7 else '❌ Necesita mejora'}
        """)

def visualizar_matriz_confusion_mejorada(cm, class_names, titulo="Matriz de Confusión"):
    """
    Crea una visualización mejorada de la matriz de confusión
    """
    # Calcular porcentajes por fila, manejando filas vacías
    row_sums = cm.sum(axis=1, keepdims=True)
    # Evitar división por cero: si la suma es 0, dejar la fila en 0
    row_sums[row_sums == 0] = 1
    cm_percent = cm.astype('float') / row_sums * 100
    # Opcional: redondear a 1 decimal para visualización
    cm_percent = np.round(cm_percent, 1)
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de confusión con valores absolutos
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Matriz de Confusión (Valores Absolutos)')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Real')
    
    # Matriz de confusión con porcentajes
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Matriz de Confusión (Porcentajes)')
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Real')
    # Mostrar suma de cada fila (debería ser 100%)
    percent_sums = cm_percent.sum(axis=1)
    for i, total in enumerate(percent_sums):
        ax2.text(len(class_names), i + 0.5, f"{total:.1f}", va='center', ha='left', color='black', fontsize=9, fontweight='bold')
    ax2.set_xlim(-0.5, len(class_names) + 0.5)
    # Etiqueta para la suma
    ax2.set_xticks(list(ax2.get_xticks()) + [len(class_names)])
    ax2.set_xticklabels(list(class_names) + ['Σ'], rotation=45 if len(class_names) > 3 else 0)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Interpretación automática
    diagonal_sum = np.trace(cm)
    total_sum = np.sum(cm)
    accuracy = diagonal_sum / total_sum
    
    st.write("#### Interpretación de la matriz de confusión:")
    st.write(f"- **Accuracy**: {accuracy:.3f} ({diagonal_sum}/{total_sum} predicciones correctas)")
    
    # Identificar clases con mayor confusión
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_confusion = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
    
    if cm_no_diag[max_confusion] > 0:
        st.write(f"- **Mayor confusión**: {class_names[max_confusion[0]]} confundida con {class_names[max_confusion[1]]} ({cm_no_diag[max_confusion]} casos)")

def crear_curvas_roc_interactivas(y_true, y_prob, class_names):
    """
    Crea curvas ROC interactivas con Plotly
    """
    fig = go.Figure()
    
    if len(class_names) == 2:
        # Clasificación binaria
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Curva ROC
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Línea diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Clasificador aleatorio',
            line=dict(color='navy', dash='dash', width=2),
            hovertemplate='Línea de referencia<extra></extra>'
        ))
        
    else:
        # Clasificación multiclase
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        colors = px.colors.qualitative.Set1
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{class_name} (AUC = {roc_auc:.3f})',
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f'{class_name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
            ))
        
        # Línea diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Clasificador aleatorio',
            line=dict(color='black', dash='dash', width=2),
            hovertemplate='Línea de referencia<extra></extra>'
        ))
    
    fig.update_layout(
        title='Curvas ROC Interactivas',
        xaxis_title='Tasa de Falsos Positivos (FPR)',
        yaxis_title='Tasa de Verdaderos Positivos (TPR)',
        width=800,
        height=600,
        hovermode='closest'
    )
    
    return fig

def calcular_metricas_regresion(y_true, y_pred, titulo="Métricas de Regresión"):
    """
    Calcula y muestra métricas para problemas de regresión
    """
    metricas = {}
    metricas['mse'] = mean_squared_error(y_true, y_pred)
    metricas['rmse'] = np.sqrt(metricas['mse'])
    metricas['mae'] = mean_absolute_error(y_true, y_pred)
    metricas['r2'] = r2_score(y_true, y_pred)
    
    # Mostrar métricas
    st.write(f"### {titulo}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{metricas['r2']:.3f}")
    with col2:
        st.metric("RMSE", f"{metricas['rmse']:.3f}")
    with col3:
        st.metric("MAE", f"{metricas['mae']:.3f}")
    with col4:
        st.metric("MSE", f"{metricas['mse']:.3f}")
    
    # Interpretaciones
    with st.expander("💡 Interpretación de métricas de regresión"):
        st.markdown(f"""
        **R² Score ({metricas['r2']:.3f})**: Proporción de varianza explicada por el modelo.
        - 1.0 = Perfecto ajuste
        - 0.0 = Modelo no mejor que predecir la media
        - <0.0 = Modelo peor que predecir la media
        {'✅ Excelente' if metricas['r2'] > 0.9 else '✅ Bueno' if metricas['r2'] > 0.7 else '⚠️ Regular' if metricas['r2'] > 0.5 else '❌ Pobre'}
        
        **RMSE ({metricas['rmse']:.3f})**: Error cuadrático medio. Mismas unidades que la variable objetivo.
        Penaliza más los errores grandes.
        
        **MAE ({metricas['mae']:.3f})**: Error absoluto medio. Mismas unidades que la variable objetivo.
        Menos sensible a valores atípicos que RMSE.
        
        **MSE ({metricas['mse']:.3f})**: Error cuadrático medio. Unidades al cuadrado.
        Base matemática para RMSE.
        """)
    
    # Gráfico de valores reales vs predichos
    fig_reg, ax_reg = plt.subplots(figsize=(8, 6))
    ax_reg.scatter(y_true, y_pred, alpha=0.6)
    ax_reg.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax_reg.set_xlabel('Valores Reales')
    ax_reg.set_ylabel('Valores Predichos')
    ax_reg.set_title('Valores Reales vs Predichos')
    ax_reg.grid(True, alpha=0.3)
    
    # Añadir línea de mejor ajuste
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax_reg.plot(y_true, p(y_true), "b--", alpha=0.8, linewidth=1, label=f'Ajuste lineal')
    ax_reg.legend()
    
    st.pyplot(fig_reg)
    
    # Gráfico de residuos
    residuos = y_true - y_pred
    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuos vs predichos
    ax1.scatter(y_pred, residuos, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Valores Predichos')
    ax1.set_ylabel('Residuos')
    ax1.set_title('Residuos vs Valores Predichos')
    ax1.grid(True, alpha=0.3)
    
    # Histograma de residuos
    ax2.hist(residuos, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuos')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Residuos')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_res)
    
    return metricas


# NUEVO: Título y selección de tipo de análisis (ahora incluye Bayes Ingenuo)
st.title("Análisis Integrado: Discriminante (LDA/QDA), Bayes Ingenuo y PCA")
analisis = st.sidebar.selectbox(
    "Selecciona el tipo de análisis",
    ["Discriminante (LDA/QDA)", "Bayes Ingenuo", "Reducción de dimensiones (PCA)"]
)

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

    # Detectar columnas categóricas elegibles para target
    max_unique_target = 20
    columnas = df.columns.tolist()
    num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in columnas if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() <= max_unique_target]
    # Asignar nombres descriptivos a las clases antes de la vista previa
    if cat_cols:
        st.write("### Asignación de nombres descriptivos a las clases")
        target_col = st.selectbox("Selecciona la columna de clase (target):", cat_cols, index=max(0, len(cat_cols)-1), key="target_global")
        clase_unicos = sorted(df[target_col].unique())
        conteo_clase = df[target_col].value_counts().sort_index()
        st.write("#### Valores únicos de la clase:")
        # Inputs para nombres descriptivos
        clase_labels_global = st.session_state.get("clase_labels_global", {})
        for v in clase_unicos:
            label = st.text_input(f"Nombre descriptivo para la clase '{v}'", value=clase_labels_global.get(v, str(v)), key=f"label_global_{v}")
            clase_labels_global[v] = label if label.strip() else str(v)
        st.session_state["clase_labels_global"] = clase_labels_global
        st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global[v] for v in conteo_clase.index], "Cantidad": conteo_clase.values}), use_container_width=True)
    st.write("### Vista previa del dataset")
    st.info("""
    **¿Qué es esto?**
    Aquí puedes ver las primeras filas del dataset cargado. Esto te permite:
    - Verificar que los datos se han cargado correctamente.
    - Observar los nombres de las columnas y el tipo de datos.
    - Identificar posibles valores nulos o atípicos.
    
    **¿Cómo interpretarlo?**
    - Cada fila es una observación (ejemplo, individuo, muestra).
    - Cada columna es una variable (característica, atributo, o la clase a predecir).
    - Revisa que los nombres de columnas sean claros y que los datos tengan sentido antes de continuar.
    """)
    st.dataframe(df.head())

    # ================= VISTA DEDICADA: BAYES INGENUO =================
    if analisis == "Bayes Ingenuo":
        st.header("Clasificación Bayes Ingenuo")
        with st.expander("¿Qué es Bayes Ingenuo? (Explicación teórica, práctica y predicción por clase)", expanded=True):
            st.markdown(r'''
**Bayes Ingenuo** es un algoritmo de clasificación supervisada basado en el Teorema de Bayes, con el supuesto de que las características son independientes entre sí dado la clase.

**Teorema de Bayes:**
$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

- $P(C|X)$: Probabilidad de la clase $C$ dado los atributos $X$.
- $P(X|C)$: Probabilidad de observar $X$ si la clase es $C$.
- $P(C)$: Probabilidad previa de la clase $C$.
- $P(X)$: Probabilidad de observar $X$.

**Supuesto ingenuo:**
$$
P(X|C) = \prod_{i=1}^n P(x_i|C)
$$
Esto simplifica el cálculo, aunque en la práctica las variables pueden estar correlacionadas.

**¿Cómo funciona?**
1. Calcula la probabilidad previa de cada clase y la probabilidad condicional de cada atributo dado la clase.
2. Para una nueva observación, multiplica las probabilidades y elige la clase con mayor probabilidad posterior.

**Ventajas:**
- Muy rápido y eficiente.
- Funciona bien incluso con pocos datos.
- Fácil de implementar.

**Desventajas:**
- El supuesto de independencia rara vez se cumple totalmente.
- No modela relaciones entre variables.

**Aplicaciones:**
- Clasificación de correos (spam/no spam), análisis de sentimientos, diagnóstico médico, etc.

''')
        # Selección de variables igual que en LDA/QDA
        max_unique_target = 20
        columnas = df.columns.tolist()
        num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in columnas if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() <= max_unique_target]
        st.caption("""
        **¿Qué es la columna de clase (target)?**
        Es la variable que quieres predecir. Debe ser categórica (por ejemplo: 'especie', 'tipo de vino', 'aprobado/suspendido').
        El modelo aprenderá a predecir esta columna usando las demás variables.
        """)
        if not cat_cols:
            st.error("No hay columnas válidas para usar como variable de clase (target). Elige un dataset con una columna categórica o entera con pocos valores únicos.")
        else:
            with st.expander("1️⃣ Selección de columna de clase y significado de valores", expanded=True):
                target_col = st.selectbox("Selecciona la columna de clase (target):", cat_cols, index=max(0, len(cat_cols)-1), key="target_bayes")
                # Mostrar valores únicos y conteo
                st.info("""
                **¿Qué significa cada valor de la clase?**
                Aquí puedes ver las clases únicas y su cantidad. Esto te ayuda a saber a qué hace referencia cada clase (por ejemplo, 0 = no potable, 1 = potable).
                """)
                clase_unicos = sorted(df[target_col].unique())
                conteo_clase = df[target_col].value_counts().sort_index()
                # Obtener mapeo de nombres descriptivos (si ya existe)
                clase_labels_global = st.session_state.get("clase_labels_global", {})
                for v in clase_unicos:
                    label = st.session_state.get(f"label_global_{v}", str(v))
                    clase_labels_global[v] = label if label.strip() else str(v)
                st.session_state["clase_labels_global"] = clase_labels_global
                st.write("#### Valores únicos de la clase:")
                st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global[v] for v in conteo_clase.index], "Cantidad": conteo_clase.values}), use_container_width=True)
                # Mostrar ejemplos de filas para cada valor de clase
                st.write("#### Ejemplos para cada clase:")
                for v in clase_unicos:
                    nombre_clase = clase_labels_global[v]
                    st.caption(f"Ejemplos para la clase '{nombre_clase}':")
                    st.dataframe(df[df[target_col] == v].head(3), use_container_width=True)
                
                st.caption("""
                **¿Por qué es importante?**
                - Interpretar correctamente los resultados depende de saber qué significa cada clase.
                - Si tienes dudas, consulta la fuente del dataset o pregunta a tu docente.
                """)
            feature_cols = st.multiselect(
                "Selecciona las columnas de atributos (features):",
                [c for c in num_cols if c != target_col],
                default=[c for c in num_cols if c != target_col],
                key="features_bayes"
            )
            st.caption("""
            **¿Qué son las columnas de atributos (features)?**
            Son las variables que el modelo usará para predecir la clase. Deben ser numéricas (por ejemplo: 'edad', 'alcohol', 'longitud').
            Elige aquellas que creas relevantes para la predicción. Puedes seleccionar varias.
            """)
            # Opción de preprocesamiento PCA
            usar_pca = st.checkbox("Aplicar reducción de dimensiones (PCA) como preprocesamiento", value=False, key="usar_pca_bayes")
            if usar_pca and feature_cols:
                st.info("PCA es una técnica de preprocesamiento no supervisado que transforma las variables originales en componentes principales, conservando la mayor parte de la información. Úsalo para reducir la dimensionalidad antes de entrenar el modelo.")
                varianza_pca = st.slider("Porcentaje mínimo de varianza acumulada a conservar (PCA)", min_value=50, max_value=99, value=80, step=1, key="varianza_pca_bayes")
            if feature_cols and target_col:
                X = df[feature_cols]
                y = df[target_col]
                if X.isnull().values.any():
                    st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
                    X = X.fillna(X.mean())
                # Preprocesamiento PCA si corresponde
                if usar_pca:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    pca_full = PCA(n_components=min(X.shape[0], X.shape[1]))
                    pca_full.fit(X_scaled)
                    var_acum = np.cumsum(pca_full.explained_variance_ratio_)
                    n_comp = np.argmax(var_acum >= varianza_pca/100) + 1
                    pca = PCA(n_components=n_comp)
                    X_pca = pca.fit_transform(X_scaled)
                    X_model = X_pca
                    st.success(f"PCA aplicado: {n_comp} componentes principales conservan al menos {varianza_pca}% de la varianza.")
                else:
                    X_model = X
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB()
                model.fit(X_model, y)
                with st.expander("2️⃣ Predicción interactiva y probabilidades por clase", expanded=True):
                    st.write("### Ingresa una observación para predecir la clase")
                    st.info("""
                    **¿Qué es esto?**
                    Aquí puedes ingresar valores para cada atributo y predecir a qué clase pertenecería una nueva observación según el modelo entrenado.
                    
                    **¿Cómo se usa?**
                    - Ingresa valores numéricos para cada feature.
                    - Puedes usar ejemplos reales o probar valores hipotéticos.
                    - El modelo calculará la clase más probable y las probabilidades asociadas.
                    
                    **¿Cómo interpretarlo?**
                    - Útil para ver cómo el modelo clasifica nuevos casos y entender la influencia de cada variable.
                    """)
                    col1, col2 = st.columns([2,1])
                    with col2:
                        if st.button("Cargar ejemplo aleatorio", key="btn_cargar_ejemplo_bayes"):
                            ejemplo = df[feature_cols].sample(1).iloc[0].to_dict()
                            for col in feature_cols:
                                st.session_state[f"input_{col}_bayes"] = float(ejemplo[col])
                            st.rerun()
                    nueva_obs = []
                    for col in feature_cols:
                        minv = float(df[col].min())
                        maxv = float(df[col].max())
                        meanv = float(df[col].mean())
                        col3, col4 = st.columns([3,2])
                        with col3:
                            val = st.number_input(f"{col}", min_value=minv, max_value=maxv, value=st.session_state.get(f"input_{col}_bayes", meanv), key=f"input_{col}_bayes")
                            nueva_obs.append(val)
                        with col4:
                            st.caption(f"mín: {minv:.2f}\nmedia: {meanv:.2f}\nmáx: {maxv:.2f}")
                    nueva_obs = [nueva_obs]
                    # Si se usó PCA, transformar la observación
                    if usar_pca:
                        nueva_obs_scaled = scaler.transform(nueva_obs)
                        nueva_obs_pca = pca.transform(nueva_obs_scaled)
                        obs_model = nueva_obs_pca
                    else:
                        obs_model = nueva_obs
                    prediccion = model.predict(obs_model)
                    probas = model.predict_proba(obs_model)[0]
                    class_names = [str(c) for c in model.classes_]
                    # Botón solo para feedback visual, pero la predicción es reactiva
                    st.button("Predecir clase", key="btn_pred_bayes")
                    # Visualización y lógica siempre activas
                    class_labels_global = st.session_state.get("clase_labels_global", {})
                    nombre_prediccion = class_labels_global.get(prediccion[0], str(prediccion[0]))
                    st.success(f"Predicción: {nombre_prediccion}")
                    with st.expander("Ver probabilidades por clase", expanded=True):
                        st.write("#### Probabilidades por clase para la observación ingresada:")
                        st.caption("""
                        **¿Qué significa esto?**
                        Aquí se muestran las probabilidades calculadas para cada clase posible, dadas las características ingresadas.
                        
                        **¿Cómo interpretarlo?**
                        - La clase con mayor probabilidad es la predicción del modelo.
                        - Si varias clases tienen probabilidades similares, el modelo está menos seguro.
                        - Útil para analizar la confianza y la ambigüedad en la predicción.
                        """)
                        # Usar nombres descriptivos en la tabla y gráfico
                        nombres_clase_descriptivos = [class_labels_global.get(c, str(c)) for c in model.classes_]
                        df_proba = pd.DataFrame({
                            'Clase': nombres_clase_descriptivos,
                            'Probabilidad': [f"{p:.3f}" for p in probas]
                        })
                        st.dataframe(df_proba, use_container_width=True)
                        # Gráfico de barras
                        import plotly.graph_objects as go
                        fig_proba = go.Figure(go.Bar(
                            x=nombres_clase_descriptivos,
                            y=probas,
                            marker_color='royalblue',
                            text=[f"{p:.2%}" for p in probas],
                            textposition='auto'))
                        fig_proba.update_layout(
                            title="Probabilidad de pertenencia a cada clase",
                            xaxis_title="Clase",
                            yaxis_title="Probabilidad",
                            yaxis=dict(range=[0,1]),
                            width=600, height=400
                        )
                        st.plotly_chart(fig_proba, use_container_width=True)
                        # Interpretación automática
                        max_idx = int(np.argmax(probas))
                        max_prob = probas[max_idx]
                        clase_max_nombre = nombres_clase_descriptivos[max_idx]
                        if max_prob > 0.9:
                            st.info(f"El modelo está **muy seguro** de que la observación pertenece a la clase **{clase_max_nombre}** (probabilidad {max_prob:.1%}).")
                        elif max_prob > 0.7:
                            st.warning(f"El modelo predice la clase **{clase_max_nombre}** con **confianza moderada** (probabilidad {max_prob:.1%}).")
                        else:
                            st.error(f"La predicción es **incierta**: la clase más probable es **{clase_max_nombre}** pero con baja confianza ({max_prob:.1%}). Revisa las probabilidades por clase.")
                    # Umbral de decisión interactivo
                    with st.expander("⚙️ Opcional: Ajustar umbral de decisión"):
                        st.markdown("Puedes modificar el umbral mínimo de probabilidad para asignar una clase. Si ninguna clase supera el umbral, la predicción se considera incierta.")
                        threshold = st.slider("Umbral mínimo de probabilidad", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="thresh_bayes")
                        clases_superan = [(c, p) for c, p in zip(class_names, probas) if p >= threshold]
                        if len(clases_superan) == 0:
                            st.error(f"Con umbral {threshold:.2f}, **ninguna clase supera el umbral**. La predicción es incierta.")
                        elif len(clases_superan) == 1:
                            st.success(f"Con umbral {threshold:.2f}, la clase predicha es **{clases_superan[0][0]}** (probabilidad {clases_superan[0][1]:.1%}).")
                        else:
                            st.warning(f"Con umbral {threshold:.2f}, varias clases superan el umbral: " + ", ".join([f"{c} ({p:.1%})" for c, p in clases_superan]))
                # ======== EVALUACIÓN COMPLETA DEL MODELO ========
                with st.expander("3️⃣ Evaluación completa del modelo", expanded=True):
                    st.write("## 📊 Evaluación del Modelo")
                    st.info("""
                    **¿Qué es esto?**
                    Aquí se evalúa el desempeño general del modelo Bayes Ingenuo usando todo el dataset.
                    
                    **¿Para qué sirve?**
                    - Permite ver qué tan bien clasifica el modelo en promedio.
                    - Incluye métricas globales y por clase, matriz de confusión y reportes detallados.
                    - Útil para comparar con otros modelos y detectar posibles problemas.
                    """)
                    y_pred = model.predict(X_model)
                    y_prob = None
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_prob = model.predict_proba(X_model)
                    except:
                        y_prob = None
                    class_labels_global = st.session_state.get("clase_labels_global", {})
                    class_names = [class_labels_global.get(c, str(c)) for c in model.classes_]
                    metricas = calcular_metricas_clasificacion(y, y_pred, y_prob, class_names)
                    mostrar_metricas_clasificacion(metricas, "Métricas de Bayes Ingenuo")
                    with st.expander("Ver matriz de confusión", expanded=True):
                        st.write("### 🎯 Matriz de Confusión Detallada")
                        st.caption("""
                        **¿Qué es la matriz de confusión?**
                        Es una tabla que muestra cuántas veces el modelo predijo correctamente cada clase y cuántas veces se confundió.
                        
                        **¿Cómo interpretarla?**
                        - La diagonal muestra los aciertos (predicciones correctas).
                        - Los valores fuera de la diagonal son errores de clasificación.
                        - Permite identificar patrones de confusión entre clases.
                        """)
                        visualizar_matriz_confusion_mejorada(metricas['confusion_matrix'], class_names)
                    if y_prob is not None and len(class_names) > 1:
                        with st.expander("Ver curvas ROC", expanded=True):
                            st.write("### Curvas ROC (si hay probabilidades disponibles)")
                            fig_roc = crear_curvas_roc_interactivas(y, y_prob, class_names)
                            st.plotly_chart(fig_roc, use_container_width=True)
                    with st.expander("📋 Reporte de Clasificación Completo", expanded=True):
                        st.caption("""
                        **¿Qué es esto?**
                        Es un reporte detallado con métricas de precisión, recall y F1-score para cada clase, mostrado como tabla interactiva.
                        
                        **¿Para qué sirve?**
                        - Permite analizar el rendimiento del modelo en cada clase específica.
                        - Útil para detectar clases difíciles de predecir o desbalanceadas.
                        """)
                        # Mostrar como DataFrame con formato condicional pastel legible
                        from sklearn.metrics import classification_report
                        import pandas as pd
                        import numpy as np
                        report_dict = classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
                        df_report = pd.DataFrame(report_dict).T
                        # Reordenar columnas si existen
                        cols = [c for c in ['precision','recall','f1-score','support'] if c in df_report.columns]
                        df_report = df_report[cols]
                        # Formatear valores
                        for c in ['precision','recall','f1-score']:
                            if c in df_report.columns:
                                df_report[c] = df_report[c].apply(lambda x: float(x) if isinstance(x, (float, np.floating, int, np.integer)) else np.nan)
                        # Formatear soporte como int
                        if 'support' in df_report.columns:
                            df_report['support'] = df_report['support'].astype(int)
                        # Formato condicional pastel MUY claro y texto en negrita/negro
                        def pastel_metric(val):
                            if pd.isnull(val) or val == 0:
                                return 'background-color: #ffffff; color: #111; font-weight: bold;'
                            if val >= 0.8:
                                color = '#81c784'  # verde pastel saturado
                            elif val >= 0.6:
                                color = '#fff176'  # amarillo pastel saturado
                            elif val > 0.0:
                                color = '#e57373'  # rojo pastel saturado
                            else:
                                color = '#ffffff'
                            return f'background-color: {color}; color: #111; font-weight: bold;'
                        styled = df_report.style.applymap(pastel_metric, subset=['precision','recall','f1-score'])\
                            .format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:d}'})
                        st.dataframe(styled, use_container_width=True, hide_index=False)
                        st.markdown("""
                        <div style='margin-bottom: 0.5em;'>
                        <strong>Glosario de colores:</strong><br>
                        <span style='background-color:#81c784; color:#111; padding:2px 8px; border-radius:4px;'>🟩 Verde</span> = valor alto (≥ 0.8, buen desempeño)<br>
                        <span style='background-color:#fff176; color:#111; padding:2px 8px; border-radius:4px;'>🟨 Amarillo</span> = valor medio (≥ 0.6, aceptable)<br>
                        <span style='background-color:#e57373; color:#111; padding:2px 8px; border-radius:4px;'>🟥 Rojo</span> = valor bajo (> 0.0, necesita mejora)<br>
                        <span style='background-color:#fff; color:#111; padding:2px 8px; border-radius:4px;'>⬜ Blanco</span> = cero o nulo
                        </div>
                        """, unsafe_allow_html=True)
                    with st.expander("Ver rendimiento por clase y distribución", expanded=True):
                        st.write("### 📊 Rendimiento por Clase")
                        st.caption("""
                        **¿Qué muestra este gráfico?**
                        Aquí puedes comparar visualmente la precisión, recall y F1-score de cada clase, así como la cantidad de ejemplos por clase.
                        
                        **¿Cómo interpretarlo?**
                        - Barras altas indican buen rendimiento para esa clase.
                        - Si una clase tiene pocas muestras o bajo score, puede ser más difícil de predecir.
                        - Útil para identificar clases desbalanceadas o problemáticas.
                        """)
                        fig_class, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
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
                        class_counts = pd.Series(y).value_counts().sort_index()
                        ax2.bar(range(len(class_counts)), class_counts.values, alpha=0.7, color='skyblue')
                        ax2.set_xlabel('Clases')
                        ax2.set_ylabel('Número de muestras')
                        ax2.set_title('Distribución de clases en el dataset')
                        ax2.set_xticks(range(len(class_names)))
                        ax2.set_xticklabels(class_names, rotation=45 if len(class_names) > 3 else 0)
                        ax2.grid(True, alpha=0.3)
                        for i, v in enumerate(class_counts.values):
                            ax2.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig_class)
                        st.info("Puedes comparar el rendimiento de Bayes Ingenuo con LDA/QDA seleccionando el mismo dataset en las otras vistas.")
    # ================= FIN VISTA BAYES INGENUO =================
    elif analisis == "Discriminante (LDA/QDA)":
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
        st.caption("""
        **¿Qué es la columna de clase (target)?**
        Es la variable que quieres predecir. Debe ser categórica (por ejemplo: 'especie', 'tipo de vino', 'aprobado/suspendido').
        El modelo aprenderá a predecir esta columna usando las demás variables.
        """)
        if not cat_cols:
            st.error("No hay columnas válidas para usar como variable de clase (target). Elige un dataset con una columna categórica o entera con pocos valores únicos.")
        else:
            target_col = st.selectbox("Selecciona la columna de clase (target):", cat_cols, index=max(0, len(cat_cols)-1))
            st.caption("""
            **¿Qué son las columnas de atributos (features)?**
            Son las variables que el modelo usará para predecir la clase. Deben ser numéricas (por ejemplo: 'edad', 'alcohol', 'longitud').
            Elige aquellas que creas relevantes para la predicción. Puedes seleccionar varias.
            """)
            feature_cols = st.multiselect(
                "Selecciona las columnas de atributos (features):",
                [c for c in num_cols if c != target_col],
                default=[c for c in num_cols if c != target_col]
            )
            # Opción de preprocesamiento PCA
            usar_pca = st.checkbox("Aplicar reducción de dimensiones (PCA) como preprocesamiento", value=False, key="usar_pca_ldaqda")
            if usar_pca and feature_cols:
                st.info("PCA es una técnica de preprocesamiento no supervisado que transforma las variables originales en componentes principales, conservando la mayor parte de la información. Úsalo para reducir la dimensionalidad antes de entrenar el modelo.")
                varianza_pca = st.slider("Porcentaje mínimo de varianza acumulada a conservar (PCA)", min_value=50, max_value=99, value=80, step=1, key="varianza_pca_ldaqda")
            if feature_cols and target_col:
                X = df[feature_cols]
                y = df[target_col]
                if X.isnull().values.any():
                    st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
                    X = X.fillna(X.mean())
                # Preprocesamiento PCA si corresponde
                if usar_pca:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    pca_full = PCA(n_components=min(X.shape[0], X.shape[1]))
                    pca_full.fit(X_scaled)
                    var_acum = np.cumsum(pca_full.explained_variance_ratio_)
                    n_comp = np.argmax(var_acum >= varianza_pca/100) + 1
                    pca = PCA(n_components=n_comp)
                    X_pca = pca.fit_transform(X_scaled)
                    X_model = X_pca
                    st.success(f"PCA aplicado: {n_comp} componentes principales conservan al menos {varianza_pca}% de la varianza.")
                else:
                    X_model = X
                algoritmo = st.selectbox("Selecciona el algoritmo", ["LDA", "QDA"])
                if algoritmo == "LDA":
                    model = LinearDiscriminantAnalysis(store_covariance=True)
                elif algoritmo == "QDA":
                    model = QuadraticDiscriminantAnalysis(store_covariance=True)
                model.fit(X_model, y)
                # Mostrar matriz de covarianza estimada por el modelo (solo LDA/QDA)
                if algoritmo in ["LDA", "QDA"]:
                    st.write(f"### Matriz de covarianza estimada por el modelo ({algoritmo})")
                    import seaborn as sns
                    # Determinar nombres de columnas para la matriz de covarianza
                    if usar_pca and 'n_comp' in locals():
                        nombres_cov = [f"PC{i+1}" for i in range(n_comp)]
                    else:
                        nombres_cov = feature_cols
                    if algoritmo == "LDA":
                        cov_matrix = model.covariance_
                        df_cov = pd.DataFrame(cov_matrix, index=nombres_cov, columns=nombres_cov)
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
                            df_cov = pd.DataFrame(cov, index=nombres_cov, columns=nombres_cov)
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
                st.info("""
                **¿Qué es esto?**
                Aquí puedes ingresar valores para cada atributo y predecir a qué clase pertenecería una nueva observación según el modelo entrenado (LDA/QDA/Bayes Ingenuo).
                
                **¿Cómo se usa?**
                - Ingresa valores numéricos para cada feature.
                - Puedes usar ejemplos reales o probar valores hipotéticos.
                - El modelo calculará la clase más probable y, si es posible, las probabilidades asociadas.
                
                **¿Cómo interpretarlo?**
                - Útil para ver cómo el modelo clasifica nuevos casos y entender la influencia de cada variable.
                """)
                for col in feature_cols:
                    minv = float(df[col].min())
                    maxv = float(df[col].max())
                    meanv = float(df[col].mean())
                    col3, col4 = st.columns([3,2])
                    with col3:
                        val = st.number_input(f"{col}", min_value=minv, max_value=maxv, value=meanv)
                        nueva_obs.append(val)
                    with col4:
                        st.caption(f"mín: {minv:.2f}\nmedia: {meanv:.2f}\nmáx: {maxv:.2f}")
                nueva_obs = [nueva_obs]
                # Si se usó PCA, transformar la observación
                if usar_pca:
                    nueva_obs_scaled = scaler.transform(nueva_obs)
                    nueva_obs_pca = pca.transform(nueva_obs_scaled)
                    obs_model = nueva_obs_pca
                else:
                    obs_model = nueva_obs
                prediccion = model.predict(obs_model)
                probas = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probas = model.predict_proba(obs_model)[0]
                    except Exception:
                        probas = None
                if st.button("Predecir clase"):
                    st.caption("""
                    **¿Qué significa esto?**
                    El modelo predice la clase más probable para la observación ingresada. Si el modelo soporta probabilidades, también se mostrarán.
                    
                    **¿Cómo interpretarlo?**
                    - La clase mostrada es la predicción del modelo.
                    - Si hay probabilidades, puedes ver la confianza del modelo en su predicción.
                    """)
                    class_labels_global = st.session_state.get("clase_labels_global", {})
                    nombre_prediccion = class_labels_global.get(prediccion[0], str(prediccion[0]))
                    st.success(f"Predicción: {nombre_prediccion}")
                    st.write(f"Algoritmo usado: {algoritmo}")
                    # Mostrar probabilidades por clase si existen
                    if probas is not None:
                        class_names = [str(c) for c in model.classes_]
                        class_labels_global = st.session_state.get("clase_labels_global", {})
                        nombres_clase = [class_labels_global.get(c, str(c)) for c in model.classes_]
                        with st.expander("Ver probabilidades por clase", expanded=True):
                            st.write("#### Probabilidades por clase para la observación ingresada:")
                            st.caption("""
                            **¿Qué significa esto?**
                            Aquí se muestran las probabilidades calculadas para cada clase posible, dadas las características ingresadas.
                            
                            **¿Cómo interpretarlo?**
                            - La clase con mayor probabilidad es la predicción del modelo.
                            - Si varias clases tienen probabilidades similares, el modelo está menos seguro.
                            - Útil para analizar la confianza y la ambigüedad en la predicción.
                            """)
                            # Tabla de probabilidades
                            df_proba = pd.DataFrame({
                                'Clase': nombres_clase,
                                'Probabilidad': [f"{p:.3f}" for p in probas]
                            })
                            st.dataframe(df_proba, use_container_width=True)
                            # Gráfico de barras
                            import plotly.graph_objects as go
                            fig_proba = go.Figure(go.Bar(
                                x=nombres_clase,
                                y=probas,
                                marker_color='royalblue',
                                text=[f"{p:.2%}" for p in probas],
                                textposition='auto'))
                            fig_proba.update_layout(
                                title="Probabilidad de pertenencia a cada clase",
                                xaxis_title="Clase",
                                yaxis_title="Probabilidad",
                                yaxis=dict(range=[0,1]),
                                width=600, height=400
                            )
                            st.plotly_chart(fig_proba, use_container_width=True)
                    if algoritmo == "Bayes Ingenuo":
                        st.info("Bayes Ingenuo (Naive Bayes) es un clasificador probabilístico basado en la regla de Bayes y la independencia entre atributos.")
                # ======== EVALUACIÓN COMPLETA DEL MODELO ========
                st.write("## 📊 Evaluación del Modelo")
                st.info("""
                **¿Qué es esto?**
                Aquí se evalúa el desempeño general del modelo seleccionado usando todo el dataset.
                
                **¿Para qué sirve?**
                - Permite ver qué tan bien clasifica el modelo en promedio.
                - Incluye métricas globales y por clase, matriz de confusión y reportes detallados.
                - Útil para comparar con otros modelos y detectar posibles problemas.
                """)
                
                # Predicciones
                y_pred = model.predict(X_model)
                
                # Obtener probabilidades si es posible
                y_prob = None
                try:
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_model)
                    elif hasattr(model, 'decision_function'):
                        # Para modelos como SVM que usan decision_function
                        decision_scores = model.decision_function(X_model)
                        if decision_scores.ndim == 1:
                            # Clasificación binaria
                            y_prob = np.column_stack([1-decision_scores, decision_scores])
                        else:
                            # Clasificación multiclase - convertir a probabilidades aproximadas
                            from scipy.special import softmax
                            y_prob = softmax(decision_scores, axis=1)
                except:
                    y_prob = None
                
                # Calcular todas las métricas
                class_labels_global = st.session_state.get("clase_labels_global", {})
                class_names = [class_labels_global.get(c, str(c)) for c in model.classes_]
                metricas = calcular_metricas_clasificacion(y, y_pred, y_prob, class_names)
                
                # Mostrar métricas principales
                mostrar_metricas_clasificacion(metricas, f"Métricas de {algoritmo}")
                
                # Validación cruzada
                st.write("### 🔄 Validación Cruzada")
                min_samples_per_class = y.value_counts().min()
                cv_splits = min(5, min_samples_per_class) if min_samples_per_class >= 2 else None
                
                if cv_splits and cv_splits >= 2:
                    # Múltiples métricas con validación cruzada
                    cv_scores = {}
                    for metric_name, metric_str in [('Accuracy', 'accuracy'), 
                                                   ('Precision', 'precision_macro'), 
                                                   ('Recall', 'recall_macro'), 
                                                   ('F1-Score', 'f1_macro')]:
                        scores = cross_val_score(model, X_model, y, scoring=metric_str, cv=cv_splits)
                        cv_scores[metric_name] = scores
                    
                    # Mostrar resultados de CV en columnas
                    col1, col2, col3, col4 = st.columns(4)
                    cols = [col1, col2, col3, col4]
                    
                    for i, (metric_name, scores) in enumerate(cv_scores.items()):
                        with cols[i]:
                            st.metric(
                                f"{metric_name} (CV={cv_splits})",
                                f"{np.mean(scores):.3f}",
                                f"±{np.std(scores):.3f}"
                            )
                    
                    # Gráfico de distribución de scores de CV
                    st.write("#### Distribución de scores en validación cruzada")
                    fig_cv, ax_cv = plt.subplots(figsize=(10, 6))
                    
                    positions = np.arange(len(cv_scores))
                    box_data = [scores for scores in cv_scores.values()]
                    
                    bp = ax_cv.boxplot(box_data, positions=positions, patch_artist=True)
                    ax_cv.set_xticklabels(cv_scores.keys())
                    ax_cv.set_ylabel('Score')
                    ax_cv.set_title(f'Distribución de métricas - Validación Cruzada (CV={cv_splits})')
                    ax_cv.grid(True, alpha=0.3)
                    
                    # Colorear las cajas
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    
                    st.pyplot(fig_cv)
                    
                else:
                    st.warning("No se puede calcular validación cruzada porque alguna clase tiene menos de 2 muestras.")
                
                # Matriz de confusión mejorada
                st.write("### 🎯 Matriz de Confusión Detallada")
                st.caption("""
                **¿Qué es la matriz de confusión?**
                Es una tabla que muestra cuántas veces el modelo predijo correctamente cada clase y cuántas veces se confundió.
                
                **¿Cómo interpretarla?**
                - La diagonal muestra los aciertos (predicciones correctas).
                - Los valores fuera de la diagonal son errores de clasificación.
                - Permite identificar patrones de confusión entre clases.
                """)
                visualizar_matriz_confusion_mejorada(metricas['confusion_matrix'], class_names)
                
                # Curvas ROC (si hay probabilidades disponibles)
                # Visualización de curvas ROC con Plotly (interactivo)
                if y_prob is not None and len(class_names) > 1:
                    st.write("### 📈 Curvas ROC Interactivas")
                    fig_roc = crear_curvas_roc_interactivas(y, y_prob, class_names)
                    st.plotly_chart(fig_roc, use_container_width=True)
                    # Interpretación del AUC
                    if metricas.get('roc_auc'):
                        auc_val = metricas['roc_auc']
                        st.write("#### Interpretación del AUC:")
                        if auc_val >= 0.9:
                            st.success(f"🎉 AUC = {auc_val:.3f} - Excelente capacidad discriminativa")
                        elif auc_val >= 0.8:
                            st.success(f"✅ AUC = {auc_val:.3f} - Buena capacidad discriminativa")
                        elif auc_val >= 0.7:
                            st.warning(f"⚠️ AUC = {auc_val:.3f} - Capacidad discriminativa aceptable")
                        elif auc_val >= 0.6:
                            st.warning(f"⚠️ AUC = {auc_val:.3f} - Capacidad discriminativa pobre")
                        else:
                            st.error(f"❌ AUC = {auc_val:.3f} - Capacidad discriminativa muy pobre")
                
                # Reporte de clasificación detallado
                with st.expander("📋 Reporte de Clasificación Completo"):
                    st.caption("""
                    **¿Qué es esto?**
                    Es un reporte detallado con métricas de precisión, recall y F1-score para cada clase.
                    
                    **¿Para qué sirve?**
                    - Permite analizar el rendimiento del modelo en cada clase específica.
                    - Útil para detectar clases difíciles de predecir o desbalanceadas.
                    """)
                    report = classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    # Reordenar columnas si existen
                    cols = [c for c in ['precision','recall','f1-score','support'] if c in report_df.columns]
                    report_df = report_df[cols]
                    # Formatear valores
                    for c in ['precision','recall','f1-score']:
                        if c in report_df.columns:
                            report_df[c] = report_df[c].apply(lambda x: float(x) if isinstance(x, (float, np.floating, int, np.integer)) else np.nan)
                    if 'support' in report_df.columns:
                        report_df['support'] = report_df['support'].astype(int)
                    def pastel_metric(val):
                        if pd.isnull(val) or val == 0:
                            return 'background-color: #ffffff; color: #111; font-weight: bold;'
                        if val >= 0.8:
                            color = '#81c784'  # verde pastel saturado
                        elif val >= 0.6:
                            color = '#fff176'  # amarillo pastel saturado
                        elif val > 0.0:
                            color = '#e57373'  # rojo pastel saturado
                        else:
                            color = '#ffffff'
                        return f'background-color: {color}; color: #111; font-weight: bold;'
                    styled = report_df.style.applymap(pastel_metric, subset=['precision','recall','f1-score'])\
                        .format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:d}'})
                    st.dataframe(styled, use_container_width=True, hide_index=False)
                
                # Comparación de rendimiento por clase
                st.write("### 📊 Rendimiento por Clase")
                st.caption("""
                **¿Qué muestra este gráfico?**
                Aquí puedes comparar visualmente la precisión, recall y F1-score de cada clase, así como la cantidad de ejemplos por clase.
                
                **¿Cómo interpretarlo?**
                - Barras altas indican buen rendimiento para esa clase.
                - Si una clase tiene pocas muestras o bajo score, puede ser más difícil de predecir.
                - Útil para identificar clases desbalanceadas o problemáticas.
                """)
                fig_class, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
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
                class_counts = pd.Series(y).value_counts().sort_index()
                ax2.bar(range(len(class_counts)), class_counts.values, alpha=0.7, color='skyblue')
                ax2.set_xlabel('Clases')
                ax2.set_ylabel('Número de muestras')
                ax2.set_title('Distribución de clases en el dataset')
                ax2.set_xticks(range(len(class_names)))
                ax2.set_xticklabels(class_names, rotation=45 if len(class_names) > 3 else 0)
                ax2.grid(True, alpha=0.3)
                
                # Agregar valores en las barras
                for i, v in enumerate(class_counts.values):
                    ax2.text(i, v + max(class_counts.values) * 0.01, str(v), 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig_class)
                
                # ======== COMPARACIÓN DE MODELOS ========
                st.write("## ⚖️ Comparación de Algoritmos Estudiados")
                
                if st.checkbox("Ejecutar comparación de algoritmos estudiados (LDA, QDA, Bayes Ingenuo)"):
                    with st.spinner("Comparando algoritmos..."):
                        # Lista de algoritmos a comparar
                        algoritmos_comparar = {
                            'LDA': LinearDiscriminantAnalysis(),
                            'QDA': QuadraticDiscriminantAnalysis(),
                            'Bayes Ingenuo': None  # Se define abajo
                        }
                        
                        # Agregar Bayes Ingenuo
                        from sklearn.naive_bayes import GaussianNB
                        algoritmos_comparar['Bayes Ingenuo'] = GaussianNB()
                        
                        # Resultados de comparación
                        resultados_comparacion = {}
                        
                        # Comparar cada algoritmo
                        for nombre, modelo in algoritmos_comparar.items():
                            try:
                                # Entrenar modelo
                                modelo.fit(X_model, y)
                                y_pred_comp = modelo.predict(X_model)
                                
                                # Obtener probabilidades
                                y_prob_comp = None
                                try:
                                    if hasattr(modelo, 'predict_proba'):
                                        y_prob_comp = modelo.predict_proba(X_model)
                                except:
                                    pass
                                
                                # Calcular métricas
                                metricas_comp = calcular_metricas_clasificacion(y, y_pred_comp, y_prob_comp, class_names)
                                
                                # Validación cruzada si es posible
                                cv_accuracy = None
                                if cv_splits and cv_splits >= 2:
                                    try:
                                        cv_scores = cross_val_score(modelo, X_model, y, cv=cv_splits, scoring='accuracy')
                                        cv_accuracy = np.mean(cv_scores)
                                    except:
                                        pass
                                
                                # Guardar resultados
                                resultados_comparacion[nombre] = {
                                    'accuracy': metricas_comp['accuracy'],
                                    'precision': metricas_comp['precision_macro'],
                                    'recall': metricas_comp['recall_macro'],
                                    'f1': metricas_comp['f1_macro'],
                                    'roc_auc': metricas_comp.get('roc_auc', None),
                                    'cv_accuracy': cv_accuracy
                                }
                                
                            except Exception as e:
                                st.warning(f"Error al evaluar {nombre}: {str(e)}")
                                continue
                        
                        # Mostrar tabla comparativa
                        if resultados_comparacion:
                            st.write("### 📋 Tabla Comparativa de Algoritmos")
                            
                            # Crear DataFrame para la comparación
                            df_comparacion = pd.DataFrame(resultados_comparacion).T
                            
                            # Formatear números
                            for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_accuracy']:
                                if col in df_comparacion.columns:
                                    df_comparacion[col] = df_comparacion[col].apply(
                                        lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
                                    )
                            
                            # Renombrar columnas para mejor presentación
                            columnas_nombres = {
                                'accuracy': 'Accuracy',
                                'precision': 'Precision',
                                'recall': 'Recall',
                                'f1': 'F1-Score',
                                'roc_auc': 'ROC-AUC',
                                'cv_accuracy': f'CV Accuracy ({cv_splits}-fold)' if cv_splits else 'CV Accuracy'
                            }
                            
                            df_comparacion = df_comparacion.rename(columns=columnas_nombres)
                            
                            # Destacar el mejor modelo para cada métrica
                            def highlight_best(s):
                                if s.name == 'ROC-AUC' or 'CV Accuracy' in s.name or s.name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                                    # Convertir a float para comparar, ignorando N/A
                                    numeric_values = []
                                    indices = []
                                    for i, val in enumerate(s):
                                        try:
                                            if val != "N/A":
                                                numeric_values.append(float(val))
                                                indices.append(i)
                                        except:
                                            continue
                                    
                                    if numeric_values:
                                        max_val = max(numeric_values)
                                        max_idx = indices[numeric_values.index(max_val)]
                                        colors = [''] * len(s)
                                        colors[max_idx] = 'background-color: lightgreen'
                                        return colors
                                return [''] * len(s)
                            
                            # Aplicar formato
                            styled_df = df_comparacion.style.apply(highlight_best, axis=0)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Gráfico comparativo
                            st.write("### 📊 Comparación Visual de Algoritmos")
                            
                            # Convertir datos para el gráfico
                            metricas_para_grafico = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                            datos_grafico = {}
                            
                            for metrica in metricas_para_grafico:
                                if metrica in df_comparacion.columns:
                                    datos_grafico[metrica] = []
                                    for algoritmo in df_comparacion.index:
                                        try:
                                            val = df_comparacion.loc[algoritmo, metrica]
                                            if val != "N/A":
                                                datos_grafico[metrica].append(float(val))
                                            else:
                                                datos_grafico[metrica].append(0)
                                        except:
                                            datos_grafico[metrica].append(0)
                            
                            # Crear gráfico de barras comparativo
                            fig_comp, ax_comp = plt.subplots(figsize=(12, 8))
                            
                            x = np.arange(len(df_comparacion.index))
                            width = 0.2
                            
                            colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
                            
                            for i, (metrica, valores) in enumerate(datos_grafico.items()):
                                ax_comp.bar(x + i*width, valores, width, label=metrica, 
                                           alpha=0.8, color=colors[i % len(colors)])
                            
                            ax_comp.set_xlabel('Algoritmos')
                            ax_comp.set_ylabel('Score')
                            ax_comp.set_title('Comparación de Algoritmos - Todas las Métricas')
                            ax_comp.set_xticks(x + width * 1.5)
                            ax_comp.set_xticklabels(df_comparacion.index, rotation=45, ha='right')
                            ax_comp.legend()
                            ax_comp.grid(True, alpha=0.3)
                            ax_comp.set_ylim(0, 1.1)
                            
                            plt.tight_layout()
                            st.pyplot(fig_comp)
                            
                            # Recomendaciones automáticas
                            st.write("### 🎯 Recomendaciones")
                            
                            # Encontrar el mejor algoritmo por métrica
                            recomendaciones = []
                            
                            for metrica in metricas_para_grafico:
                                if metrica in df_comparacion.columns:
                                    try:
                                        # Convertir a numérico y encontrar el máximo
                                        valores_numericos = {}
                                        for algoritmo in df_comparacion.index:
                                            val = df_comparacion.loc[algoritmo, metrica]
                                            if val != "N/A":
                                                valores_numericos[algoritmo] = float(val)
                                        
                                        if valores_numericos:
                                            mejor_algoritmo = max(valores_numericos, key=valores_numericos.get)
                                            mejor_valor = valores_numericos[mejor_algoritmo]
                                            recomendaciones.append(f"**{metrica}**: {mejor_algoritmo} ({mejor_valor:.3f})")
                                    except:
                                        continue
                            
                            if recomendaciones:
                                st.write("Mejores algoritmos por métrica:")
                                for rec in recomendaciones:
                                    st.write(f"- {rec}")
                            
                            # Recomendación general
                            if 'Accuracy' in df_comparacion.columns:
                                try:
                                    accuracy_vals = {}
                                    for algoritmo in df_comparacion.index:
                                        val = df_comparacion.loc[algoritmo, 'Accuracy']
                                        if val != "N/A":
                                            accuracy_vals[algoritmo] = float(val)
                                    
                                    if accuracy_vals:
                                        mejor_general = max(accuracy_vals, key=accuracy_vals.get)
                                        st.success(f"🏆 **Recomendación general**: {mejor_general} tiene el mejor rendimiento global (Accuracy: {accuracy_vals[mejor_general]:.3f})")
                                except:
                                    pass
                
                st.write("### Visualización de separación de clases")
                show_proj = st.checkbox("Mostrar gráfico de componentes discriminantes (proyección LDA/QDA)")
                if show_proj:
                    try:
                        # Determinar número de componentes posibles
                        n_clases = len(np.unique(y))
                        n_comp = n_clases - 1 if algoritmo == "LDA" else min(len(feature_cols), n_clases)
                        # Proyección LDA/QDA
                        if algoritmo == "LDA":
                            X_proj = model.transform(X)
                        else:
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=min(3, X.shape[1]))
                            X_proj = pca.fit_transform(X)
                        class_labels_global = st.session_state.get("clase_labels_global", {})
                        # Opción para elegir 2D o 3D si hay suficientes componentes
                        plot_mode = "2D"
                        if X_proj.shape[1] >= 3:
                            plot_mode = st.radio("Tipo de gráfico", ["2D", "3D"], index=0, horizontal=True)
                        if plot_mode == "3D" and X_proj.shape[1] >= 3:
                            import plotly.express as px
                            df_plot = pd.DataFrame({
                                'Componente 1': X_proj[:, 0],
                                'Componente 2': X_proj[:, 1],
                                'Componente 3': X_proj[:, 2],
                                'Clase': [class_labels_global.get(val, str(val)) for val in y]
                            })
                            fig_plotly = px.scatter_3d(
                                df_plot,
                                x='Componente 1',
                                y='Componente 2',
                                z='Componente 3',
                                color='Clase',
                                title='Separación de clases (proyección LDA/QDA 3D)',
                                width=900,
                                height=700,
                                opacity=0.7
                            )
                            fig_plotly.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
                            st.plotly_chart(fig_plotly, use_container_width=True)
                        else:
                            import plotly.express as px
                            df_plot = pd.DataFrame({
                                'Componente 1': X_proj[:, 0],
                                'Componente 2': X_proj[:, 1],
                                'Clase': [class_labels_global.get(val, str(val)) for val in y]
                            })
                            fig_plotly = px.scatter(
                                df_plot,
                                x='Componente 1',
                                y='Componente 2',
                                color='Clase',
                                title='Separación de clases (proyección LDA/QDA)',
                                width=800,
                                height=600,
                                opacity=0.7
                            )
                            fig_plotly.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
                            st.plotly_chart(fig_plotly, use_container_width=True)
                    except Exception as e:
                        st.error(f"No se pudo calcular la proyección: {e}")
                elif len(feature_cols) == 2:
                    class_labels_global = st.session_state.get("clase_labels_global", {})
                    df_plot = pd.DataFrame({
                        feature_cols[0]: X[feature_cols[0]],
                        feature_cols[1]: X[feature_cols[1]],
                        'Clase': [class_labels_global.get(val, str(val)) for val in y]
                    })
                    import plotly.express as px
                    fig_plotly = px.scatter(
                        df_plot,
                        x=feature_cols[0],
                        y=feature_cols[1],
                        color='Clase',
                        title=f'Separación de clases ({feature_cols[0]} vs {feature_cols[1]})',
                        width=800,
                        height=600,
                        opacity=0.7
                    )
                    fig_plotly.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
                    st.plotly_chart(fig_plotly, use_container_width=True)
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
            if X_pca.isnull().values.any():
                st.warning("Se encontraron valores faltantes en las columnas seleccionadas para PCA. Imputando con la media de cada columna...")
                X_pca = X_pca.fillna(X_pca.mean())
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
                # Visualización 3D mejorada e interactiva con Plotly
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Opciones avanzadas de visualización 3D
                st.write("#### 🎨 Opciones avanzadas de visualización 3D")
                col1_3d, col2_3d, col3_3d, col4_3d = st.columns(4)
                
                with col1_3d:
                    st.write("**Puntos**")
                    marker_size = st.slider("Tamaño de puntos", 3, 15, 8, key="marker_size_3d")
                    marker_opacity = st.slider("Transparencia", 0.1, 1.0, 0.8, 0.1, key="marker_opacity_3d")
                
                with col2_3d:
                    st.write("**Ejes y rejilla**")
                    show_grid = st.checkbox("Mostrar rejilla", True, key="show_grid_3d")
                    show_axes = st.checkbox("Mostrar ejes", True, key="show_axes_3d")
                
                with col3_3d:
                    st.write("**Colores**")
                    color_scheme = st.selectbox("Esquema de colores", 
                                              ["Plotly", "Viridis", "Plasma", "Inferno", "Set1", "Pastel"],
                                              key="color_scheme_3d")
                
                with col4_3d:
                    st.write("**Fondo**")
                    background_style = st.selectbox("Estilo de fondo", 
                                                   ["Claro", "Oscuro", "Neutro", "Científico"],
                                                   key="background_style_3d")
                    axis_style = st.selectbox("Estilo de ejes", 
                                            ["Moderno", "Clásico", "Minimalista"],
                                            key="axis_style_3d")
                
                # Configurar esquema de colores
                if color_scheme == "Viridis":
                    colors = px.colors.sequential.Viridis
                elif color_scheme == "Plasma":
                    colors = px.colors.sequential.Plasma
                elif color_scheme == "Inferno":
                    colors = px.colors.sequential.Inferno
                elif color_scheme == "Set1":
                    colors = px.colors.qualitative.Set1
                elif color_scheme == "Pastel":
                    colors = px.colors.qualitative.Pastel
                else:  # Plotly default
                    colors = px.colors.qualitative.Plotly
                
                # Configurar estilos de fondo
                if background_style == "Oscuro":
                    scene_bgcolor = 'rgba(45,55,72,1)'
                    paper_bgcolor = 'rgba(26,32,44,1)'
                    plot_bgcolor = 'rgba(45,55,72,1)'
                    grid_color = 'rgba(203,213,224,0.3)'
                    background_color = 'rgba(74,85,104,0.2)' if show_grid else 'rgba(45,55,72,0)'
                    zeroline_color = 'rgba(203,213,224,0.6)'
                    line_color = 'rgba(203,213,224,0.4)'
                    font_color = '#E2E8F0'
                    title_color = '#E2E8F0'
                    tick_color = '#CBD5E0'
                elif background_style == "Neutro":
                    scene_bgcolor = 'rgba(247,250,252,1)'
                    paper_bgcolor = 'rgba(255,255,255,1)'
                    plot_bgcolor = 'rgba(247,250,252,1)'
                    grid_color = 'rgba(148,163,184,0.4)'
                    background_color = 'rgba(226,232,240,0.5)' if show_grid else 'rgba(255,255,255,0)'
                    zeroline_color = 'rgba(71,85,105,0.7)'
                    line_color = 'rgba(71,85,105,0.5)'
                    font_color = '#374151'
                    title_color = '#1F2937'
                    tick_color = '#4B5563'
                elif background_style == "Científico":
                    scene_bgcolor = 'rgba(249,250,251,1)'
                    paper_bgcolor = 'rgba(255,255,255,1)'
                    plot_bgcolor = 'rgba(249,250,251,1)'
                    grid_color = 'rgba(156,163,175,0.5)'
                    background_color = 'rgba(243,244,246,0.8)' if show_grid else 'rgba(255,255,255,0)'
                    zeroline_color = 'rgba(55,65,81,0.8)'
                    line_color = 'rgba(55,65,81,0.6)'
                    font_color = '#111827'
                    title_color = '#111827'
                    tick_color = '#374151'
                else:  # Claro (default)
                    scene_bgcolor = 'rgba(248,251,253,1)'
                    paper_bgcolor = 'rgba(255,255,255,1)'
                    plot_bgcolor = 'rgba(248,251,253,1)'
                    grid_color = 'rgba(149,165,166,0.3)'
                    background_color = 'rgba(248,249,250,0.8)' if show_grid else 'rgba(255,255,255,0)'
                    zeroline_color = 'rgba(52,73,94,0.6)'
                    line_color = 'rgba(52,73,94,0.4)'
                    font_color = '#2C3E50'
                    title_color = '#2E4057'
                    tick_color = '#34495E'
                
                # Configurar estilos de ejes
                if axis_style == "Clásico":
                    axis_line_width = 3
                    zeroline_width = 4
                    grid_width = 2
                    tick_font_size = 12
                    title_font_size = 15
                elif axis_style == "Minimalista":
                    axis_line_width = 1
                    zeroline_width = 2
                    grid_width = 1
                    tick_font_size = 10
                    title_font_size = 13
                else:  # Moderno (default)
                    axis_line_width = 2
                    zeroline_width = 3
                    grid_width = 1
                    tick_font_size = 11
                    title_font_size = 14
                
                # Crear texto mejorado para hover
                hover_text = []
                for i in range(X_proj.shape[0]):
                    # Información básica de componentes
                    txt = f"<b>Punto {i+1}</b><br>"
                    txt += f"{comp_x}: <b>{X_proj[i, idx_x]:.3f}</b><br>"
                    txt += f"{comp_y}: <b>{X_proj[i, idx_y]:.3f}</b><br>"
                    txt += f"{comp_z}: <b>{X_proj[i, idx_z]:.3f}</b><br>"
                    
                    # Información de clase si está disponible
                    if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                        class_labels_pca = st.session_state.get("clase_labels_global", {})
                        nombre_clase = class_labels_pca.get(df[class_col_pca].iloc[i], str(df[class_col_pca].iloc[i]))
                        txt += f"<br><b>{class_col_pca}:</b> {nombre_clase}<br>"
                    
                    # Información adicional de variables originales más influyentes
                    txt += "<br><b>Variables originales influyentes:</b><br>"
                    # Calcular contribución de variables originales a este punto
                    contribuciones = {}
                    for j, var in enumerate(feature_cols_pca):
                        contrib = (pca.components_[idx_x, j] * X_pca_scaled[i, j] + 
                                 pca.components_[idx_y, j] * X_pca_scaled[i, j] + 
                                 pca.components_[idx_z, j] * X_pca_scaled[i, j])
                        contribuciones[var] = abs(contrib)
                    
                    # Top 3 variables más influyentes
                    top_vars = sorted(contribuciones.items(), key=lambda x: x[1], reverse=True)[:3]
                    for var, contrib in top_vars:
                        txt += f"• {var}: {X_pca[var].iloc[i]:.2f}<br>"
                    
                    hover_text.append(txt)
                
                # Crear visualización según si hay clases o no
                data = []
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    class_labels_pca = st.session_state.get("clase_labels_global", {})
                    clases = pd.Categorical(df[class_col_pca]).categories
                    codigos = pd.Categorical(df[class_col_pca]).codes
                    for idx, clase in enumerate(clases):
                        puntos = codigos == idx
                        nombre_clase = class_labels_pca.get(clase, str(clase))
                        if np.any(puntos):  # Solo agregar si hay puntos de esta clase
                            # Calcular estadísticas por clase
                            n_puntos = np.sum(puntos)
                            centroide_x = np.mean(X_proj[puntos, idx_x])
                            centroide_y = np.mean(X_proj[puntos, idx_y])
                            centroide_z = np.mean(X_proj[puntos, idx_z])
                            data.append(go.Scatter3d(
                                x=X_proj[puntos, idx_x],
                                y=X_proj[puntos, idx_y],
                                z=X_proj[puntos, idx_z],
                                mode='markers',
                                marker=dict(
                                    size=marker_size,
                                    opacity=marker_opacity,
                                    color=colors[idx % len(colors)],
                                    line=dict(width=0.5, color='darkgray')
                                ),
                                name=f'{nombre_clase} (n={n_puntos})',
                                text=[hover_text[i].replace(str(clase), nombre_clase) for i in range(len(hover_text)) if puntos[i]],
                                hoverinfo='text',
                                legendgroup=f'grupo_{idx}'
                            ))
                            # Agregar centroide de cada clase
                            data.append(go.Scatter3d(
                                x=[centroide_x],
                                y=[centroide_y],
                                z=[centroide_z],
                                mode='markers',
                                marker=dict(
                                    size=marker_size * 2,
                                    opacity=1.0,
                                    color=colors[idx % len(colors)],
                                    symbol='diamond',
                                    line=dict(width=2, color='black')
                                ),
                                name=f'Centroide {nombre_clase}',
                                text=f'<b>Centroide de {nombre_clase}</b><br>{comp_x}: {centroide_x:.3f}<br>{comp_y}: {centroide_y:.3f}<br>{comp_z}: {centroide_z:.3f}',
                                hoverinfo='text',
                                legendgroup=f'grupo_{idx}',
                                showlegend=False
                            ))
                else:
                    # Sin clases, usar gradiente de colores basado en distancia al origen
                    distancias = np.sqrt(X_proj[:, idx_x]**2 + X_proj[:, idx_y]**2 + X_proj[:, idx_z]**2)
                    
                    data.append(go.Scatter3d(
                        x=X_proj[:, idx_x],
                        y=X_proj[:, idx_y],
                        z=X_proj[:, idx_z],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            opacity=marker_opacity,
                            color=distancias,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Distancia al origen", thickness=15),
                            line=dict(width=0.5, color='darkgray')
                        ),
                        name='Datos PCA',
                        text=hover_text,
                        hoverinfo='text'
                    ))
                
                # Agregar vectores de carga (loadings) si se desea
                show_loadings = st.checkbox("Mostrar vectores de carga (loadings)", False, key="show_loadings_3d")
                if show_loadings:
                    # Escalar los vectores para visualización
                    scale_factor = st.slider("Factor de escala para vectores", 1, 10, 5, key="loading_scale")
                    
                    for i, var in enumerate(feature_cols_pca):
                        loading_x = pca.components_[idx_x, i] * scale_factor
                        loading_y = pca.components_[idx_y, i] * scale_factor
                        loading_z = pca.components_[idx_z, i] * scale_factor
                        
                        # Vector desde el origen
                        hover_loading_text = f'<b>Vector de carga: {var}</b><br>Contribución a {comp_x}: {pca.components_[idx_x, i]:.3f}<br>Contribución a {comp_y}: {pca.components_[idx_y, i]:.3f}<br>Contribución a {comp_z}: {pca.components_[idx_z, i]:.3f}'
                        
                        data.append(go.Scatter3d(
                            x=[0, loading_x],
                            y=[0, loading_y],
                            z=[0, loading_z],
                            mode='lines+text',
                            line=dict(color='red', width=3),
                            text=['', var],
                            textposition='top center',
                            name=f'Loading {var}',
                            showlegend=False,
                            hovertext=[hover_loading_text, hover_loading_text],
                            hoverinfo='text'
                        ))
                
                # Configuración avanzada del layout con estilos personalizables
                layout = go.Layout(
                    title={
                        'text': f"<b>PCA - Proyección 3D Interactiva</b><br><sub>{comp_x} vs {comp_y} vs {comp_z}</sub>",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': title_color}
                    },
                    scene=dict(
                        xaxis=dict(
                            title=dict(
                                text=f"<b>{comp_x}</b><br>({var_exp[idx_x]*100:.1f}% varianza)",
                                font=dict(size=title_font_size, color=font_color)
                            ),
                            showbackground=show_grid,
                            backgroundcolor=background_color,
                            gridcolor=grid_color,
                            gridwidth=grid_width,
                            zeroline=True,
                            zerolinecolor=zeroline_color,
                            zerolinewidth=zeroline_width,
                            showspikes=False,
                            visible=show_axes,
                            tickfont=dict(size=tick_font_size, color=tick_color),
                            showline=True,
                            linecolor=line_color,
                            linewidth=axis_line_width,
                            mirror=False,
                            showgrid=show_grid,
                            tickmode='auto',
                            nticks=6
                        ),
                        yaxis=dict(
                            title=dict(
                                text=f"<b>{comp_y}</b><br>({var_exp[idx_y]*100:.1f}% varianza)",
                                font=dict(size=title_font_size, color=font_color)
                            ),
                            showbackground=show_grid,
                            backgroundcolor=background_color,
                            gridcolor=grid_color,
                            gridwidth=grid_width,
                            zeroline=True,
                            zerolinecolor=zeroline_color,
                            zerolinewidth=zeroline_width,
                            showspikes=False,
                            visible=show_axes,
                            tickfont=dict(size=tick_font_size, color=tick_color),
                            showline=True,
                            linecolor=line_color,
                            linewidth=axis_line_width,
                            mirror=False,
                            showgrid=show_grid,
                            tickmode='auto',
                            nticks=6
                        ),
                        zaxis=dict(
                            title=dict(
                                text=f"<b>{comp_z}</b><br>({var_exp[idx_z]*100:.1f}% varianza)",
                                font=dict(size=title_font_size, color=font_color)
                            ),
                            showbackground=show_grid,
                            backgroundcolor=background_color,
                            gridcolor=grid_color,
                            gridwidth=grid_width,
                            zeroline=True,
                            zerolinecolor=zeroline_color,
                            zerolinewidth=zeroline_width,
                            showspikes=False,
                            visible=show_axes,
                            tickfont=dict(size=tick_font_size, color=tick_color),
                            showline=True,
                            linecolor=line_color,
                            linewidth=axis_line_width,
                            mirror=False,
                            showgrid=show_grid,
                            tickmode='auto',
                            nticks=6
                        ),
                        bgcolor=scene_bgcolor,
                        camera=dict(
                            eye=dict(x=1.8, y=1.8, z=1.5),
                            center=dict(x=0, y=0, z=0),
                            up=dict(x=0, y=0, z=1)
                        ),
                        aspectmode='cube',
                        dragmode='orbit'
                    ),
                    legend=dict(
                        bgcolor=f'rgba(255,255,255,0.9)' if background_style != "Oscuro" else f'rgba(45,55,72,0.9)',
                        bordercolor=line_color,
                        borderwidth=1,
                        x=0.02,
                        y=0.98,
                        font=dict(size=11, color=font_color),
                        itemsizing='constant',
                        itemwidth=30
                    ),
                    margin=dict(l=10, r=10, b=10, t=80),
                    font=dict(size=12, color=font_color, family="Arial, sans-serif"),
                    paper_bgcolor=paper_bgcolor,
                    plot_bgcolor=plot_bgcolor,
                    hoverlabel=dict(
                        bgcolor=f'rgba(255,255,255,0.95)' if background_style != "Oscuro" else f'rgba(45,55,72,0.95)',
                        bordercolor=zeroline_color,
                        font_size=12,
                        font_family="Arial",
                        font_color=font_color
                    )
                )
                
                fig_pca = go.Figure(data=data, layout=layout)
                
                # Mostrar el gráfico
                st.plotly_chart(fig_pca, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'pca_3d_{comp_x}_{comp_y}_{comp_z}',
                        'height': 800,
                        'width': 1200,
                        'scale': 2
                    }
                })
                
                # Información detallada sobre la visualización
                with st.expander("🎯 Guía de interpretación del gráfico 3D"):
                    col1_info, col2_info = st.columns(2)
                    
                    with col1_info:
                        st.markdown("""
                        **📊 Elementos del gráfico:**
                        - **Puntos**: Cada observación proyectada en el espacio 3D
                        - **Colores**: Representan diferentes clases (si están definidas)
                        - **Diamantes**: Centroides de cada clase
                        - **Vectores rojos**: Direcciones de las variables originales (loadings)
                        
                        **🔄 Interactividad:**
                        - **Rotar**: Arrastra para rotar la vista
                        - **Zoom**: Rueda del ratón o pinch
                        - **Pan**: Shift + arrastrar
                        - **Hover**: Información detallada de cada punto
                        """)
                    
                    with col2_info:
                        st.markdown(f"""
                        **📈 Información estadística:**
                        - **Varianza explicada total**: {(var_exp[idx_x] + var_exp[idx_y] + var_exp[idx_z])*100:.1f}%
                        - **{comp_x}**: {var_exp[idx_x]*100:.1f}% de varianza
                        - **{comp_y}**: {var_exp[idx_y]*100:.1f}% de varianza
                        - **{comp_z}**: {var_exp[idx_z]*100:.1f}% de varianza
                        
                        **🎨 Personalización:**
                        - **Puntos**: Ajusta tamaño y transparencia
                        - **Colores**: 6 esquemas diferentes disponibles
                        - **Fondo**: 4 estilos (Claro, Oscuro, Neutro, Científico)
                        - **Ejes**: 3 estilos (Moderno, Clásico, Minimalista)
                        - **Rejilla**: Activar/desactivar según preferencia
                        - **Vectores**: Mostrar direcciones de variables originales
                        """)
                
                # Análisis automático de clustering visual
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    st.write("#### 🔍 Análisis automático de separación de clases")
                    
                    class_labels_pca = st.session_state.get("clase_labels_global", {})
                    clases_unicas = df[class_col_pca].unique()
                    n_clases = len(clases_unicas)
                    # Calcular distancias entre centroides
                    centroides = {}
                    for clase in clases_unicas:
                        mask = df[class_col_pca] == clase
                        centroides[clase] = [
                            np.mean(X_proj[mask, idx_x]),
                            np.mean(X_proj[mask, idx_y]),
                            np.mean(X_proj[mask, idx_z])
                        ]
                    # Matriz de distancias entre centroides
                    if n_clases > 1:
                        distancias_inter = []
                        pares_clases = []
                        for i, clase1 in enumerate(clases_unicas):
                            for clase2 in clases_unicas[i+1:]:
                                c1 = np.array(centroides[clase1])
                                c2 = np.array(centroides[clase2])
                                dist = np.linalg.norm(c1 - c2)
                                distancias_inter.append(dist)
                                nombre_clase1 = class_labels_pca.get(clase1, str(clase1))
                                nombre_clase2 = class_labels_pca.get(clase2, str(clase2))
                                pares_clases.append(f"{nombre_clase1} - {nombre_clase2}")
                        # Mostrar análisis
                        col1_analysis, col2_analysis = st.columns(2)
                        with col1_analysis:
                            st.metric("Número de clases", n_clases)
                            st.metric("Distancia promedio entre centroides", f"{np.mean(distancias_inter):.2f}")
                        with col2_analysis:
                            max_dist_idx = np.argmax(distancias_inter)
                            min_dist_idx = np.argmin(distancias_inter)
                            st.metric("Clases más separadas", pares_clases[max_dist_idx])
                            st.metric("Clases más cercanas", pares_clases[min_dist_idx])
                        
                        # Interpretación automática
                        separacion_promedio = np.mean(distancias_inter)
                        if separacion_promedio > 3:
                            st.success("✅ **Excelente separación**: Las clases están bien diferenciadas en el espacio PCA.")
                        elif separacion_promedio > 2:
                            st.info("ℹ️ **Buena separación**: Las clases son distinguibles pero con cierto solapamiento.")
                        elif separacion_promedio > 1:
                            st.warning("⚠️ **Separación moderada**: Existe solapamiento considerable entre clases.")
                        else:
                            st.error("❌ **Separación pobre**: Las clases están muy mezcladas en el espacio PCA.")
                
                # Información sobre estilos disponibles
                with st.expander("🎨 Guía de estilos de visualización"):
                    col_style1, col_style2 = st.columns(2)
                    
                    with col_style1:
                        st.markdown("""
                        **� Estilos de fondo:**
                        - **Claro**: Fondo suave ideal para presentaciones
                        - **Oscuro**: Reduce fatiga visual, perfecto para análisis largos
                        - **Neutro**: Profesional y neutral para reportes
                        - **Científico**: Estilo académico con máximo contraste
                        """)
                    
                    with col_style2:
                        st.markdown("""
                        **📐 Estilos de ejes:**
                        - **Moderno**: Balance entre claridad y estética
                        - **Clásico**: Líneas gruesas, máxima visibilidad
                        - **Minimalista**: Líneas finas, interfaz limpia
                        """)
                
                st.markdown("""
                **💡 Consejos para la interpretación:**
                - **Proximidad**: Puntos cercanos tienen características similares
                - **Separación**: Distancia entre grupos indica diferencias entre clases
                - **Vectores de carga**: Muestran influencia de variables originales
                - **Varianza por eje**: Indica importancia de cada dimensión
                - **Centroides**: Diamantes muestran el centro de cada clase
                """)
                
                st.caption("🔄 **Gráfico interactivo**: Rota, haz zoom y explora los datos. Pasa el cursor sobre los puntos para ver información detallada.")
            else:
                # Gráfico 2D interactivo con Plotly
                if class_col_pca != "(Ninguna)" and class_col_pca in df.columns:
                    class_labels_pca = st.session_state.get("clase_labels_global", {})
                    color_labels = df[class_col_pca].map(lambda c: class_labels_pca.get(c, str(c)))
                    fig_pca = px.scatter(
                        x=X_proj[:, idx_x],
                        y=X_proj[:, idx_y],
                        color=color_labels,
                        labels={
                            "x": comp_x,
                            "y": comp_y,
                            "color": "Clase"
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

# ======== SECCIÓN DE AYUDA Y DOCUMENTACIÓN ========
st.sidebar.markdown("---")
st.sidebar.write("## 📚 Ayuda y Documentación")

with st.sidebar.expander("📖 Guía de uso"):
    st.markdown("""
    ### 🚀 Cómo usar esta aplicación
    
    **1. Selecciona el tipo de análisis:**
    - **Discriminante (LDA/QDA)**: Para clasificación supervisada lineal o cuadrática
    - **Bayes Ingenuo**: Para clasificación supervisada basada en el teorema de Bayes
    - **PCA**: Para reducción de dimensiones
    
    **2. Carga tus datos:**
    - Selecciona un archivo CSV de la carpeta de datos
    - O sube tu propio archivo CSV
    
    **3. Configura el análisis:**
    - Selecciona las variables target y features
    - Ajusta los parámetros según tus necesidades
    
    **4. Interpreta los resultados:**
    - Revisa las métricas y visualizaciones
    - Usa las interpretaciones automáticas
    - Explora las secciones expandibles para más detalles
    """)

with st.sidebar.expander("🎯 Métricas explicadas"):
    st.markdown("""
    ### 📊 Métricas de Clasificación
    
    **Accuracy**: Proporción de predicciones correctas
    - > 0.9: Excelente
    - 0.8-0.9: Buena
    - 0.7-0.8: Regular
    - < 0.7: Necesita mejora
    
    **Precision**: Exactitud de predicciones positivas
    - Pregunta: "De los que predije positivos, ¿cuántos son realmente positivos?"
    
    **Recall (Sensibilidad)**: Capacidad de encontrar casos positivos
    - Pregunta: "De todos los casos positivos reales, ¿cuántos encontré?"
    
    **F1-Score**: Media armónica entre precision y recall
    - Balancea ambas métricas
    
    **ROC-AUC**: Área bajo la curva ROC
    - Mide capacidad discriminativa del modelo
    - > 0.9: Excelente
    - 0.8-0.9: Buena
    - 0.7-0.8: Aceptable
    - < 0.7: Pobre
    """)


with st.sidebar.expander("🧮 Bayes Ingenuo explicado"):
    st.markdown("""
    ### 🤖 Bayes Ingenuo
    
    - Algoritmo de clasificación supervisada basado en el Teorema de Bayes.
    - Supone independencia entre las variables dado la clase.
    - Muy eficiente y fácil de implementar.
    - Útil para clasificación de texto, spam, análisis de sentimientos, etc.
    
    **¿Cómo funciona?**
    - Calcula la probabilidad de cada clase dado los atributos.
    - Asigna la clase con mayor probabilidad posterior.
    
    **Ventajas:**
    - Rápido, robusto y funciona bien con pocos datos.
    
    **Desventajas:**
    - El supuesto de independencia rara vez se cumple totalmente.
    """)

with st.sidebar.expander("🔍 Interpretación de PCA"):
    st.markdown("""
    ### 📈 Componentes Principales
    
    **PC1, PC2, PC3...**: Nuevas variables creadas
    - PC1 explica la mayor varianza
    - PC2 explica la segunda mayor varianza
    - Son perpendiculares entre sí
    
    **Varianza explicada**: Información conservada
    - 80%+ acumulada es generalmente buena
    - Ayuda a decidir cuántos componentes usar
    
    **Matriz de componentes**: Cómo se construyen
    - Cada fila = un componente
    - Cada columna = una variable original
    - Valores = importancia de cada variable
    
    **Escalado**: Siempre recomendado
    - Evita que variables de mayor rango dominen
    - Permite comparación justa entre variables
    """)

with st.sidebar.expander("⚙️ Configuración avanzada"):
    st.markdown("""
    ### 🛠️ Opciones Avanzadas
    
    **Validación Cruzada**:
    - Evalúa el modelo en diferentes subconjuntos
    - Proporciona estimación más robusta
    - K-fold típicamente entre 3-10
    
    **Comparación de Modelos**:
    - Evalúa los algoritmos estudiados (LDA, QDA, Bayes Ingenuo)
    - Compara métricas lado a lado
    - Proporciona recomendaciones
    
    **Visualizaciones Interactivas**:
    - Gráficos 3D con Plotly
    - Hover para detalles
    - Zoom y rotación disponibles
    
    **Interpretaciones Automáticas**:
    - Análisis de resultados
    - Recomendaciones basadas en métricas
    - Identificación de problemas comunes
    """)

# Información sobre los datos de ejemplo si existen
if os.path.exists(carpeta_datos) and archivos_csv:
    with st.sidebar.expander("📁 Datos de ejemplo"):
        st.markdown("### 📊 Datasets disponibles:")
        for archivo in archivos_csv:
            st.write(f"- {archivo}")
        st.markdown("""
        **Formato requerido**:
        - Archivos CSV con headers
        - Variables numéricas para features
        - Variable categórica/entera para target (clasificación)
        - Sin espacios en nombres de columnas (recomendado)
        """)

# Footer con información adicional
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>📊 <strong>Análisis Integrado v2.1</strong></p>
    <p>Incluye métricas avanzadas para LDA, QDA, Naive Bayes y PCA</p>
    <p>🎓 Para inferencia estadística - Contenido según programa de la materia</p>
</div>
""", unsafe_allow_html=True)
