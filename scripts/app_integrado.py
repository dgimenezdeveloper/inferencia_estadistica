import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

# Función helper para suprimir warnings de Plotly en Streamlit
def safe_plotly_chart(fig, **kwargs):
    """Wrapper para st.plotly_chart que suprime warnings de deprecación y agrupa opciones en 'config'"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The keyword arguments have been deprecated")
        # Extraer argumentos de configuración conocidos
        config_keys = [
            'displayModeBar', 'displaylogo', 'modeBarButtonsToRemove', 'toImageButtonOptions',
            'scrollZoom', 'responsive', 'staticPlot', 'editable', 'edits', 'autosizable', 'frameMargins',
            'showTips', 'doubleClick', 'showAxisDragHandles', 'showAxisRangeEntryBoxes', 'showLink', 'sendData',
            'linkText', 'showSources', 'locale', 'locales'
        ]
        config = kwargs.pop('config', {}) or {}
        # Mover argumentos conocidos a config
        for key in config_keys:
            if key in kwargs:
                config[key] = kwargs.pop(key)
        return st.plotly_chart(fig, config=config, **kwargs)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_curve, auc,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from metricas import (
    calcular_metricas_clasificacion,
    mostrar_metricas_clasificacion,
    visualizar_matriz_confusion_mejorada,
    crear_curvas_roc_interactivas,
    calcular_metricas_regresion
)
from plotly.subplots import make_subplots
from preprocesamiento import (
    cargar_dataset,
    seleccionar_columnas,
    manejar_nulos,
    escalar_datos,
    aplicar_pca,
    elegir_n_pca_cv,
)
from modelos import (
    entrenar_lda,
    entrenar_qda, 
    entrenar_bayes,
    predecir,
    predecir_lda,
    predecir_qda,
    predecir_bayes
)
from textos_ayuda import (
    GUIA_USO,
    METRICAS_EXPLICADAS,
    BAYES_EXPLICADO,
    PCA_SIDEBAR,
    CONFIG_AVANZADA
)
from visualizaciones import plot_metricas_por_clase
###############################################################
# === Definición de temas y selección antes de cualquier uso de 'colores' ===
COLORES_TEMA = {
    "Oscuro": {
        "fondo": "#181c25",
        "fondo_secundario": "#23293a",
        "texto": "#f3f6fa",
        "texto_secundario": "#b0b8c9",
        "acento": "#4f8cff",
        "exito": "#2ecc71",
        "error": "#e74c3c",
        "info": "#3498db",
        "warning": "#f1c40f",
        "borde": "#313a4d",
        "tabla_header": "#23293a",
        "tabla_row": "#23293a",
        "tabla_row_alt": "#1a1e29",
        "input_bg": "#23293a",
        "input_border": "#4f8cff",
        "plot_bg": "#23293a",
        "plot_grid": "#313a4d",
        "plot_line": "#4f8cff",
        "plot_font": "#f3f6fa"
    },
    "Claro": {
        "fondo": "#f7f9fb",
        "fondo_secundario": "#ffffff",
        "texto": "#23293a",
        "texto_secundario": "#4f5b6b",
        "acento": "#0056d6",
        "exito": "#27ae60",
        "error": "#c0392b",
        "info": "#2980b9",
        "warning": "#e67e22",
        "borde": "#e1e8ed",
        "tabla_header": "#eaf0f6",
        "tabla_row": "#ffffff",
        "tabla_row_alt": "#f7f9fb",
        "input_bg": "#ffffff",
        "input_border": "#0056d6",
        "plot_bg": "#ffffff",
        "plot_grid": "#e1e8ed",
        "plot_line": "#0056d6",
        "plot_font": "#23293a"
    }
}

# ================== TEXTOS TEÓRICOS PLACEHOLDER ==================
TEXTO_LDA_QDA = """
El Análisis Discriminante Lineal (LDA) y Cuadrático (QDA) son técnicas de clasificación supervisada que buscan encontrar 
las mejores fronteras de decisión entre clases mediante proyecciones lineales o cuadráticas del espacio de características.
"""

TEXTO_BAYES = """
El Clasificador Bayesiano Ingenuo (Naive Bayes) es un algoritmo de clasificación probabilística basado en el teorema de Bayes,
que asume independencia condicional entre las características dadas las clases.
"""

TEXTO_PCA = """
El Análisis de Componentes Principales (PCA) es una técnica de reducción de dimensionalidad que encuentra las direcciones 
de máxima varianza en los datos para representarlos en un espacio de menor dimensión.
"""

# Inicializar el tema en session_state si no existe
if 'tema' not in st.session_state:
    st.session_state['tema'] = 'Oscuro'

tema_actual = st.sidebar.radio(
    "🌗 Tema de la interfaz",
    ["Oscuro", "Claro"],
    index=0 if st.session_state['tema'] == 'Oscuro' else 1,
    key="selector_tema"
)
if st.session_state['tema'] != tema_actual:
    st.session_state['tema'] = tema_actual
    st.rerun()  # Forzar refresco para aplicar el tema
colores = COLORES_TEMA[st.session_state['tema']]
###############################################################
# CSS dinámico para tema y elementos
st.markdown(f"""
    <style>
    /* Ajustes específicos para select/multiselect: fondo del menú overlay y contraste de opciones */
    /* Inputs y selects (control principal) */
    .stTextInput > div > input,
    .stNumberInput > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stSelectbox select,
    .stMultiSelect select,
    select,
    input[type="text"],
    input[type="number"] {{
        background-color: {colores['input_bg']} !important;
        color: {colores['texto']} !important;
        border: 1.5px solid {colores['input_border']} !important;
        border-radius: 7px;
        padding: 7px 10px;
        transition: border 0.2s, box-shadow 0.2s;
        box-shadow: 0 1px 4px {colores['borde']}22;
        font-size: 1.05em;
    }}
    /* Focus visual */
    .stTextInput > div > input:focus,
    .stNumberInput > div > input:focus,
    .stSelectbox > div > div:focus,
    .stMultiSelect > div > div:focus,
    .stSelectbox select:focus,
    .stMultiSelect select:focus,
    select:focus,
    input[type="text"]:focus,
    input[type="number"]:focus {{
        outline: none !important;
        border-color: {colores['acento']} !important;
        box-shadow: 0 0 0 2px {colores['acento']}33;
    }}

    /* Menu overlay (lista de opciones) - asegurar fondo y texto legible */
    .stSelectbox .css-1n6sfyn-MenuList,
    .stSelectbox .css-1n6sfyn-Menu,
    .stSelectbox [role="listbox"],
    .stMultiSelect .css-1n6sfyn-MenuList,
    .stMultiSelect .css-1n6sfyn-Menu,
    .stMultiSelect [role="listbox"] {{
        background-color: {colores['fondo_secundario']} !important;
        color: {colores['texto']} !important;
        border: 1px solid {colores['borde']} !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 18px {colores['borde']}44 !important;
        z-index: 9999 !important;
    }}

    /* Opciones dentro del menú: asegurar contraste y espaciado */
    .stSelectbox .css-1n7v3ny-option,
    .stSelectbox [role="option"],
    .stMultiSelect .css-1n7v3ny-option,
    .stMultiSelect [role="option"] {{
        color: {colores['texto']} !important;
        background-color: transparent !important;
        padding: 8px 12px !important;
        font-size: 1.03em !important;
    }}

    /* Hover y seleccionado: fondo con acento y texto legible */
    .stSelectbox .css-1n7v3ny-option:hover,
    .stSelectbox [role="option"]:hover,
    .stMultiSelect .css-1n7v3ny-option:hover,
    .stMultiSelect [role="option"]:hover {{
        background-color: {colores['acento']}22 !important;
        color: {colores['texto']} !important;
    }}
    .stSelectbox .css-1n7v3ny-option[aria-selected="true"],
    .stSelectbox [role="option"][aria-selected="true"],
    .stMultiSelect .css-1n7v3ny-option[aria-selected="true"],
    .stMultiSelect [role="option"][aria-selected="true"] {{
        background-color: {colores['acento']}33 !important;
        color: {colores['texto']} !important;
        font-weight: 700 !important;
    }}
    
    /* Asegurar altura mínima y alineación del valor seleccionado (evitar recorte de texto) */
    /* Usar altura flexible para que el control pueda crecer cuando los chips ocupen varias líneas */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stSelectbox .css-1pahdxg-control, .stMultiSelect .css-1pahdxg-control {{
        min-height: 56px !important;
        height: auto !important;
        display: flex !important;
        align-items: center !important;
        padding: 6px 8px !important;
        line-height: 1.25 !important;
        gap: 6px !important;
        box-sizing: border-box !important;
    }}

    /* Controlar la altura y overflow del valor mostrado (singleValue) */
    /* Para select simple mantener el recorte en una línea; para multiselect permitir wrapping */
    .stSelectbox .css-1uccc91-singleValue, .stSelectbox .css-1dimb5e-singleValue {{
        height: auto !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 6px 8px !important;
    }}
    /* En multiselect permitir que el valor mostrado (cuando hay muchas selecciones) se muestre en varias líneas */
    .stMultiSelect .css-1uccc91-singleValue, .stMultiSelect .css-1dimb5e-singleValue {{
        height: auto !important;
        line-height: 1.1 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
        padding: 4px 6px !important;
        max-width: 100% !important;
        display: block !important;
    }}

    /* Asegurar que el elemento <select> nativo también tenga altura adecuada */
    .stSelectbox select, .stMultiSelect select {{
        min-height: 40px !important;
        height: auto !important;
    }}

    /* Mejorar apariencia de los 'chips' (valores seleccionados) y permitir que hagan wrap */
    /* Compatibilidad con varias versiones de Streamlit (nombres de clases variables) */
    .stMultiSelect .css-1rhbuit-multiValue, .stSelectbox .css-1rhbuit-multiValue,
    .stMultiSelect .css-1rhbuit-multiValueLabel, .stSelectbox .css-1rhbuit-multiValueLabel,
    .stMultiSelect .css-12jo7m5, .stSelectbox .css-12jo7m5 {{
        background-color: {colores['acento']} !important;
        color: {colores['texto']} !important;
        border-radius: 12px !important;
        padding: 6px 10px !important;
        margin: 4px 6px !important;
        display: inline-flex !important;
        align-items: center !important;
        max-width: 100% !important;
        box-shadow: none !important;
        font-size: 0.95em !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }}

    .stMultiSelect .css-1rhbuit-multiValueLabel, .stSelectbox .css-1rhbuit-multiValueLabel,
    .stMultiSelect .css-12jo7m5 > span, .stSelectbox .css-12jo7m5 > span {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        max-width: 100% !important;
        padding-right: 6px !important;
        color: {colores['texto']} !important;
        display: inline-block !important;
        line-height: 1.1 !important;
    }}

    /* Icono de cierre (x) en chips */
    .stMultiSelect .css-1rhbuit-multiValue__remove, .stSelectbox .css-1rhbuit-multiValue__remove,
    .stMultiSelect .css-1rhbuit-multiValueRemove, .stSelectbox .css-1rhbuit-multiValueRemove {{
        color: {colores['texto']} !important;
        opacity: 0.95 !important;
        margin-left: 6px !important;
    }}

    /* Permitir que los valores seleccionados ocupen varias líneas y el control se expanda */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stSelectbox .css-1pahdxg-control, .stMultiSelect .css-1pahdxg-control {{
        flex-wrap: wrap !important;
        gap: 6px !important;
        align-items: flex-start !important;
        align-content: flex-start !important;
        max-height: none !important;
    }}

    /* Asegurar overflow visible para que los chips no queden recortados por contenedores y permitir scroll interno si hay exceso */
    .stSelectbox, .stMultiSelect {{
        overflow: visible !important;
        max-width: 100% !important;
    }}

    /* Si hay muchísimos chips, permitir que la zona de chips tenga un máximo y scroll vertical interno */
    .stMultiSelect .css-1pahdxg-control .css-1rhbuit-multiValue, .stMultiSelect .css-1pahdxg-control .css-12jo7m5 {{
        max-height: calc(3 * 1.6em) !important;
        overflow-y: auto !important;
    }}

    /* Placeholder de selectbox/multiselect */
    ::placeholder {{
        color: {colores['texto_secundario']} !important;
        opacity: 1 !important;
    }}

    /* Evitar forzar fondo transparente en todos los descendientes (provocaba pérdida de contraste) */
    /* Mantener regla general para botones/tablas/títulos más abajo */
    /* Botones */
    .stButton > button {{
        background-color: {colores['acento']} !important;
        color: #fff !important;
        border-radius: 7px;
        border: none;
        font-weight: 600;
        transition: background 0.2s, color 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px {colores['borde']}22;
        padding: 8px 22px;
        font-size: 1.05em;
        letter-spacing: 0.2px;
    }}
    .stButton > button:hover, .stButton > button:focus {{
        background-color: {colores['texto_secundario']} !important;
        color: {colores['acento']} !important;
        box-shadow: 0 4px 16px {colores['acento']}33;
        cursor: pointer;
    }}
    /* Tablas */
    .stDataFrame thead tr th {{
        background-color: {colores['tabla_header']} !important;
        color: {colores['acento']} !important;
        font-weight: 700;
        font-size: 1.01em;
        border-bottom: 2px solid {colores['borde']} !important;
    }}
    .stDataFrame tbody tr {{
        background-color: {colores['tabla_row']} !important;
        color: {colores['texto']} !important;
        transition: background 0.2s;
    }}
    .stDataFrame tbody tr:nth-child(even) {{
        background-color: {colores['tabla_row_alt']} !important;
    }}
    .stDataFrame tbody tr:hover {{
        background-color: {colores['acento']}22 !important;
        color: {colores['acento']} !important;
        cursor: pointer;
    }}
    /* Bordes y detalles */
    .stDataFrame, .stTextInput, .stSelectbox, .stMultiSelect, .stNumberInput {{
        border-color: {colores['borde']} !important;
        border-radius: 7px;
    }}
    /* Mensajes de feedback */
    .stSuccess {{ background-color: {colores['exito']}33 !important; color: {colores['exito']} !important; border-left: 5px solid {colores['exito']} !important; box-shadow: 0 2px 8px {colores['exito']}22; }}
    .stError {{ background-color: {colores['error']}33 !important; color: {colores['error']} !important; border-left: 5px solid {colores['error']} !important; box-shadow: 0 2px 8px {colores['error']}22; }}
    .stInfo {{ background-color: {colores['info']}33 !important; color: {colores['info']} !important; border-left: 5px solid {colores['info']} !important; box-shadow: 0 2px 8px {colores['info']}22; }}
    .stWarning {{ background-color: {colores['warning']}33 !important; color: {colores['warning']} !important; border-left: 5px solid {colores['warning']} !important; box-shadow: 0 2px 8px {colores['warning']}22; }}
    /* Scrollbar moderno */
    ::-webkit-scrollbar {{ width: 10px; background: {colores['fondo_secundario']}; }}
    ::-webkit-scrollbar-thumb {{ background: {colores['borde']}99; border-radius: 6px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {colores['acento']}99; }}
    </style>
""", unsafe_allow_html=True)





st.title("Inferencia Estadística y Reconocimiento de Patrones")

analisis = st.sidebar.selectbox(
    "Selecciona el tipo de análisis",
    ["Discriminante (LDA/QDA)", "Bayes Ingenuo", "Reducción de dimensiones (PCA)"]
)

carpeta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos')
archivos_csv = [f for f in os.listdir(carpeta_datos) if f.endswith('.csv')] if os.path.exists(carpeta_datos) else []
opcion_archivo = st.selectbox("Seleccionar archivo CSV", archivos_csv)
archivo_subido = st.file_uploader("O sube tu propio archivo CSV", type=["csv"])

# Cargar el dataset (modularizado)
df, _msg_carga = cargar_dataset(archivo_subido, opcion_archivo, carpeta_datos)
if _msg_carga:
    st.success(_msg_carga)

if df is not None:

    # Detectar columnas categóricas elegibles para target
    max_unique_target = 20
    num_cols, cat_cols = seleccionar_columnas(df, max_unique_target=max_unique_target)
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
        st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global[v] for v in conteo_clase.index], "Cantidad": conteo_clase.values}), width='stretch')
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
            st.markdown(TEXTO_BAYES)
        # Selección de variables igual que en LDA/QDA
        max_unique_target = 20
        columnas = df.columns.tolist()
        num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in columnas if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() <= max_unique_target]
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
                st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global[v] for v in conteo_clase.index], "Cantidad": conteo_clase.values}), width='stretch')
                # Mostrar ejemplos de filas para cada valor de clase
                st.write("#### Ejemplos para cada clase:")
                from preprocesamiento import mostrar_ejemplos_por_clase
                mostrar_ejemplos_por_clase(df, target_col, clase_labels_global, st, n=3)
                
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

            # === MATRIZ DE CORRELACIÓN DE FEATURES ===
            if feature_cols and len(feature_cols) >= 2:
                st.markdown("---")
                st.write("### 🔗 Matriz de correlación entre variables seleccionadas")
                st.caption("El clasificador Bayes Ingenuo asume independencia entre variables. Si ves correlaciones fuertes (valores cercanos a 1 o -1 fuera de la diagonal), considera aplicar PCA o eliminar variables redundantes para mejorar el modelo.")
                import matplotlib.pyplot as plt
                import seaborn as sns
                corr_bayes = df[feature_cols].corr()
                fig_corr, ax_corr = plt.subplots(figsize=(min(0.7*len(feature_cols)+2, 10), min(0.7*len(feature_cols)+2, 10)))
                sns.heatmap(corr_bayes, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr, annot_kws={"size":9})
                ax_corr.set_title("Matriz de correlación (features seleccionadas)")
                plt.tight_layout()
                st.pyplot(fig_corr)
                st.markdown("---")

                # --- EXPLICACIÓN AUTOMÁTICA SI SE ELIMINAN VARIABLES CORRELACIONADAS Y BAJAN LAS MÉTRICAS ---
                # Guardar en session_state el set de features y la métrica anterior SOLO si no existe
                if 'last_feature_cols' not in st.session_state:
                    st.session_state['last_feature_cols'] = feature_cols.copy()
                if 'last_bayes_metrics' not in st.session_state:
                    st.session_state['last_bayes_metrics'] = None
                # Detectar si se quitó una o más variables correlacionadas
                prev_cols = set(st.session_state['last_feature_cols'])
                curr_cols = set(feature_cols)
                removed = list(prev_cols - curr_cols)
                correlaciones_altas = []
                try:
                    all_corr = df[list(prev_cols)].corr().abs()
                    for removed_var in removed:
                        if removed_var in all_corr:
                            corrs = all_corr[removed_var].drop(removed_var)
                            if not corrs.empty:
                                for var in curr_cols:
                                    if var in corrs and corrs[var] > 0.8:
                                        correlaciones_altas.append((removed_var, var, corrs[var]))
                    if correlaciones_altas:
                        st.session_state['last_removed_corr_var'] = correlaciones_altas
                except Exception as e:
                    st.info(f"[Depuración] No se pudo calcular correlación previa: {e}")
                st.session_state['last_feature_cols'] = feature_cols.copy()
            elif feature_cols:
                st.info("Selecciona al menos dos variables numéricas para ver la matriz de correlación.")
            # Opción de preprocesamiento PCA
            usar_pca = st.checkbox("Aplicar reducción de dimensiones (PCA) como preprocesamiento", value=False, key="usar_pca_bayes")
            if usar_pca and feature_cols:
                st.info("PCA es una técnica de preprocesamiento no supervisado que transforma las variables originales en componentes principales, conservando la mayor parte de la información. Úsalo para reducir la dimensionalidad antes de entrenar el modelo.")
                varianza_pca = st.slider("Porcentaje mínimo de varianza acumulada a conservar (PCA)", min_value=50, max_value=99, value=80, step=1, key="varianza_pca_bayes")
            if feature_cols and target_col:
                X = df[feature_cols]
                y = df[target_col]
                # Manejo de nulos
                if X.isnull().values.any():
                    st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
                    X = manejar_nulos(X, metodo='media')
                # Preprocesamiento PCA si corresponde
                scaler = None
                pca = None
                if usar_pca:
                    X_scaled, scaler = escalar_datos(X)
                    # Calcular número de componentes para alcanzar la varianza requerida
                    _, _pca_auto, _var_exp_auto, n_comp = aplicar_pca(X_scaled, varianza_min=varianza_pca/100)
                    # Reentrenar PCA limitado a n_comp
                    from sklearn.decomposition import PCA as _PCA
                    pca = _PCA(n_components=n_comp)
                    X_model = pca.fit_transform(X_scaled)
                    st.success(f"PCA aplicado: {n_comp} componentes principales conservan al menos {varianza_pca}% de la varianza.")

                    # === MATRIZ DE CORRELACIÓN DE COMPONENTES PRINCIPALES ===
                    if n_comp >= 2:
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        st.markdown("---")
                        st.write("### 📐 Matriz de correlación de los componentes principales (PCA)")
                        st.caption("Tras aplicar PCA, los componentes principales deberían ser ortogonales (independientes). La matriz debe ser casi diagonal. Si ves valores altos fuera de la diagonal, revisa el preprocesamiento.")
                        corr_pca = pd.DataFrame(X_model).corr()
                        fig_corr_pca, ax_corr_pca = plt.subplots(figsize=(min(0.7*n_comp+2, 10), min(0.7*n_comp+2, 10)))
                        sns.heatmap(corr_pca, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr_pca, annot_kws={"size":9})
                        ax_corr_pca.set_title("Matriz de correlación (componentes principales)")
                        plt.tight_layout()
                        st.pyplot(fig_corr_pca)
                        st.markdown("---")
                else:
                    X_model = X

                # Entrenar Bayes ingenuo
                model = entrenar_bayes(X_model, y)

                # Calcular y guardar métricas actuales para comparación automática
                from modelos import predecir
                y_pred, y_prob = predecir(model, X_model)
                class_labels_global = st.session_state.get("clase_labels_global", {})
                class_names = [class_labels_global.get(c, str(c)) for c in model.classes_]
                from metricas import calcular_metricas_clasificacion
                metricas_actuales = calcular_metricas_clasificacion(y, y_pred, y_prob, class_names, st=None)
                acc_actual = metricas_actuales['accuracy'] if metricas_actuales and 'accuracy' in metricas_actuales else None
                # Comparar con la métrica anterior si corresponde
                acc_anterior = st.session_state.get('last_bayes_metrics', None)
                if acc_actual is not None and acc_anterior is not None and acc_actual < acc_anterior - 0.003:
                    if 'last_removed_corr_var' in st.session_state and st.session_state['last_removed_corr_var']:
                        correlaciones_altas = st.session_state['last_removed_corr_var']
                        for removed_var, max_corr_var, max_corr_val in correlaciones_altas:
                            st.warning(f"\n**Atención:** Has eliminado la variable '{removed_var}', que estaba altamente correlacionada con '{max_corr_var}' (correlación = {max_corr_val:.2f}). Aunque la correlación era alta, ambas variables aportaban información útil para la clasificación. Si al quitar una variable correlacionada bajan las métricas, es mejor dejarlas ambas. La correlación es una guía, pero lo importante es el rendimiento final del modelo.\n\nPuedes dejar ambas variables si mejoran el desempeño, aunque no cumplan la independencia total.")
                        st.session_state['last_removed_corr_var'] = None
                # Guardar la métrica actual para la próxima comparación
                st.session_state['last_bayes_metrics'] = acc_actual
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
                    if usar_pca and scaler is not None and pca is not None:
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
                        st.dataframe(df_proba, width='stretch')
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
                        safe_plotly_chart(fig_proba, width='stretch')
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
                    y_pred, y_prob = predecir(model, X_model)
                    class_labels_global = st.session_state.get("clase_labels_global", {})
                    class_names = [class_labels_global.get(c, str(c)) for c in model.classes_]
                    metricas = calcular_metricas_clasificacion(y, y_pred, y_prob, class_names, st=st)
                    mostrar_metricas_clasificacion(metricas, st, "Métricas de Bayes Ingenuo")
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
                        visualizar_matriz_confusion_mejorada(metricas['confusion_matrix'], class_names, st)
                    if y_prob is not None and len(class_names) > 1:
                        with st.expander("Ver curvas ROC", expanded=True):
                            st.write("### Curvas ROC (si hay probabilidades disponibles)")
                            fig_roc = crear_curvas_roc_interactivas(y, y_prob, class_names, go, px)
                            safe_plotly_chart(fig_roc, width='stretch')
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
                        styled = df_report.style.map(pastel_metric, subset=['precision','recall','f1-score'])\
                            .format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:d}'})
                        st.dataframe(styled, width='stretch', hide_index=False)
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
                        fig_class = plot_metricas_por_clase(metricas, class_names, y)
                        st.pyplot(fig_class)
                        st.info("Puedes comparar el rendimiento de Bayes Ingenuo con LDA/QDA seleccionando el mismo dataset en las otras vistas.")
    # ================= FIN VISTA BAYES INGENUO =================
    elif analisis == "Discriminante (LDA/QDA)":
        st.header("Clasificación Discriminante (LDA/QDA)")
        with st.expander("¿Qué es LDA y QDA? (Explicación teórica)"):
            st.markdown(TEXTO_LDA_QDA)
        # Filtrar columnas válidas para target
        max_unique_target = 20
        columnas = df.columns.tolist()
        num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in columnas if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() <= max_unique_target]
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
            # Mostrar ejemplos de filas por clase (modularizado)
            st.write("#### Ejemplos para cada clase:")
            from preprocesamiento import mostrar_ejemplos_por_clase
            clase_labels_global = st.session_state.get("clase_labels_global", {})
            mostrar_ejemplos_por_clase(df, target_col, clase_labels_global, st, n=3)
            # Opción de preprocesamiento PCA
            usar_pca = st.checkbox("Aplicar reducción de dimensiones (PCA) como preprocesamiento", value=False, key="usar_pca_ldaqda")
            if usar_pca and feature_cols:
                st.info("PCA es una técnica de preprocesamiento no supervisado que transforma las variables originales en componentes principales, conservando la mayor parte de la información. Úsalo para reducir la dimensionalidad antes de entrenar el modelo.")
                varianza_pca = st.slider("Porcentaje mínimo de varianza acumulada a conservar (PCA)", min_value=50, max_value=99, value=80, step=1, key="varianza_pca_ldaqda")
            if feature_cols and target_col:
                X = df[feature_cols]
                y = df[target_col]
                # Manejo de nulos
                if X.isnull().values.any():
                    st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
                    X = manejar_nulos(X, metodo='media')
                # Preprocesamiento PCA si corresponde
                scaler = None
                pca = None
                if usar_pca:
                    X_scaled, scaler = escalar_datos(X)
                    _, _pca_auto, _var_exp_auto, n_comp = aplicar_pca(X_scaled, varianza_min=varianza_pca/100)
                    from sklearn.decomposition import PCA as _PCA
                    pca = _PCA(n_components=n_comp)
                    X_model = pca.fit_transform(X_scaled)
                    st.success(f"PCA aplicado: {n_comp} componentes principales conservan al menos {varianza_pca}% de la varianza.")
                else:
                    X_model = X
                algoritmo = st.selectbox("Selecciona el algoritmo", ["LDA", "QDA"])
                if algoritmo == "LDA":
                    model = entrenar_lda(X_model, y, store_covariance=True)
                elif algoritmo == "QDA":
                    model = entrenar_qda(X_model, y, store_covariance=True)
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
                prediccion, probas_all = predecir(model, obs_model)
                probas = probas_all[0] if probas_all is not None else None
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
                            st.dataframe(df_proba, width='stretch')
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
                            safe_plotly_chart(fig_proba, width='stretch')
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
                y_pred, y_prob = predecir(model, X_model)
                
                # Calcular todas las métricas
                class_labels_global = st.session_state.get("clase_labels_global", {})
                class_names = [class_labels_global.get(c, str(c)) for c in model.classes_]
                metricas = calcular_metricas_clasificacion(y, y_pred, y_prob, class_names, st=st)
                
                # Mostrar métricas principales
                mostrar_metricas_clasificacion(metricas, st, f"Métricas de {algoritmo}")
                
                # Validación cruzada
                st.write("### 🔄 Validación Cruzada")
                min_samples_per_class = y.value_counts().min()
                cv_splits = min(5, min_samples_per_class) if min_samples_per_class >= 2 else None
                
                if cv_splits and cv_splits >= 2:
                    from sklearn.metrics import make_scorer
                    import warnings
                    
                    # Crear scorers personalizados con zero_division=0 para evitar warnings
                    def precision_scorer(y_true, y_pred):
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Precision is ill-defined")
                            return precision_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    def recall_scorer(y_true, y_pred):
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Recall is ill-defined")
                            return recall_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    def f1_scorer(y_true, y_pred):
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="F-score is ill-defined")
                            return f1_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    custom_precision = make_scorer(precision_scorer)
                    custom_recall = make_scorer(recall_scorer)
                    custom_f1 = make_scorer(f1_scorer)
                    
                    # Múltiples métricas con validación cruzada
                    cv_scores = {}
                    for metric_name, metric_str in [('Accuracy', 'accuracy'), 
                                                   ('Precision', custom_precision), 
                                                   ('Recall', custom_recall), 
                                                   ('F1-Score', custom_f1)]:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="X does not have valid feature names")
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
                visualizar_matriz_confusion_mejorada(metricas['confusion_matrix'], class_names, st)
                
                # Curvas ROC (si hay probabilidades disponibles)
                # Visualización de curvas ROC con Plotly (interactivo)
                if y_prob is not None and len(class_names) > 1:
                    st.write("### 📈 Curvas ROC Interactivas")
                    fig_roc = crear_curvas_roc_interactivas(y, y_prob, class_names, go, px)
                    safe_plotly_chart(fig_roc, width='stretch')
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
                    styled = report_df.style.map(pastel_metric, subset=['precision','recall','f1-score'])\
                        .format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:d}'})
                    st.dataframe(styled, width='stretch', hide_index=False)
                
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
                fig_class = plot_metricas_por_clase(metricas, class_names, y)
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
                                metricas_comp = calcular_metricas_clasificacion(y, y_pred_comp, y_prob_comp, class_names, st=st)
                                
                                # Validación cruzada si es posible
                                cv_accuracy = None
                                if cv_splits and cv_splits >= 2:
                                    try:
                                        with warnings.catch_warnings():
                                            warnings.filterwarnings("ignore", message="X does not have valid feature names")
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
                            st.dataframe(styled_df, width='stretch')
                            
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
                            safe_plotly_chart(fig_plotly, width='stretch')
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
                            safe_plotly_chart(fig_plotly, width='stretch')
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
                    safe_plotly_chart(fig_plotly, width='stretch')
    elif analisis == "Reducción de dimensiones (PCA)":
        # La sección completa de PCA (explicaciones, selección de variables y visualizaciones)
        # está implementada más abajo y usa las variables que se definen después de seleccionar las columnas para PCA.
        # Aquí solo se marca la rama para mostrar la sección correspondiente.
        st.header("Reducción de dimensiones: PCA")
        with st.expander("¿Qué es PCA? (Explicación visual y sencilla)"):
            st.markdown(TEXTO_PCA)
        num_cols_pca = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        class_col_pca = st.selectbox("Columna de clase (opcional, para colorear)", ["(Ninguna)"] + [c for c in df.columns if df[c].nunique() <= 20 and c != "(Ninguna)"])
        feature_cols_pca = st.multiselect("Selecciona las columnas numéricas para PCA:", [c for c in num_cols_pca if c != class_col_pca], default=[c for c in num_cols_pca if c != class_col_pca])
        varianza_deseada = st.slider("Porcentaje mínimo de varianza acumulada a conservar", min_value=50, max_value=99, value=80, step=1)
        if feature_cols_pca:
            st.info("Las siglas 'PC' significan 'Principal Component' o 'Componente Principal'. Por ejemplo, PC1 es el primer componente principal, PC2 el segundo, y así sucesivamente. Cada uno es una combinación lineal de las variables originales que explica una parte de la varianza total del dataset.")
            X_pca = df[feature_cols_pca]
            if X_pca.isnull().values.any():
                st.warning("Se encontraron valores faltantes en las columnas seleccionadas para PCA. Imputando con la media de cada columna...")
                X_pca = manejar_nulos(X_pca, metodo='media')
            X_pca_scaled, scaler_pca = escalar_datos(X_pca)
            n_samples, n_features = X_pca_scaled.shape
            max_components = min(n_samples, n_features)
            # Usar aplicar_pca para sugerir n_comp_auto y obtener proyección completa
            X_proj_auto, pca_auto, var_exp_full, n_comp_auto = aplicar_pca(X_pca_scaled, varianza_min=varianza_deseada/100)
            st.write(f"Se requieren **{n_comp_auto}** componentes principales para alcanzar al menos {varianza_deseada}% de varianza acumulada.")
            # Opción de auto-selección por validación cruzada
            st.markdown("---")
            st.write("#### Auto-selección de número de componentes vía CV (clasificación)")
            col_auto1, col_auto2, col_auto3 = st.columns([2,2,1])
            with col_auto1:
                classifier_choice = st.selectbox("Clasificador (para CV)", ["LogisticRegression"], index=0, key="cv_classifier")
            with col_auto2:
                metric_choice = st.selectbox("Métrica CV", ["f1_macro", "accuracy"], index=0, key="cv_metric")
            with col_auto3:
                tol_cv = st.number_input("Tolerancia (rel)", min_value=0.0, max_value=0.2, value=0.01, step=0.01, key="cv_tol")
            run_auto = st.button("Auto-seleccionar n (CV)", key="btn_auto_pca")

            n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=max_components, value=n_comp_auto)

            if run_auto:
                # --- OPTIMIZACIÓN DE AUTOSELECCIÓN DE N COMPONENTES PCA ---
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                import numpy as np
                import time
                st.info("Optimizando: se probarán menos valores de n y el cálculo será paralelo y más rápido.")
                cv_splits = min(3, max(2, n_samples // 10))
                if max_components <= 15:
                    n_range = list(range(1, max_components+1))
                else:
                    n_range = list(range(1, 11)) + list(range(12, max_components+1, 2))
                clf = LogisticRegression(max_iter=200, solver='saga', n_jobs=-1, random_state=42)
                metric = metric_choice
                scores = []
                t0 = time.time()
                # Validar y_cv: si no hay columna de clase válida, mostrar error y abortar
                if class_col_pca == "(Ninguna)" or class_col_pca not in df.columns:
                    st.error("Debes seleccionar una columna de clase válida para la auto-selección por CV.")
                else:
                    y_cv = df[class_col_pca]
                    for n in n_range:
                        pca = PCA(n_components=n, random_state=42)
                        X_proj = pca.fit_transform(X_pca_scaled)
                        try:
                            score = cross_val_score(clf, X_proj, y_cv, cv=cv_splits, scoring=metric, n_jobs=-1)
                            scores.append(np.mean(score))
                        except Exception as e:
                            scores.append(np.nan)
                    t1 = time.time()
                    st.success(f"Auto-selección completada en {t1-t0:.2f} segundos. Prueba menos valores de n para mayor velocidad.")
                    scores_arr = np.array(scores)
                    if np.all(np.isnan(scores_arr)):
                        st.error("No se pudo calcular ningún score de validación cruzada. Verifica que la columna de clase tenga al menos dos clases y suficientes muestras.")
                    else:
                        best_idx = np.nanargmax(scores_arr)
                        best_score = scores_arr[best_idx]
                        tol = tol_cv
                        n_recomendado = None
                        for i, s in enumerate(scores_arr):
                            if not np.isnan(s) and s >= best_score * (1-tol):
                                n_recomendado = n_range[i]
                                break
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(6,3))
                        ax.plot(n_range, scores_arr, marker='o', label=f"{metric}")
                        ax.axvline(n_range[best_idx], color='r', linestyle='--', label=f'Máximo score (n={n_range[best_idx]})')
                        if n_recomendado is not None:
                            ax.axvline(n_recomendado, color='g', linestyle=':', label=f'Recomendado (n={n_recomendado})')
                        ax.set_xlabel('n componentes')
                        ax.set_ylabel(metric)
                        ax.set_title('Score CV vs n componentes')
                        ax.legend()
                        st.pyplot(fig)
                        st.info(f"Mejor score: {best_score:.3f} con n={n_range[best_idx]}. Recomendado: n={n_recomendado} (tolerancia {tol*100:.1f}%)")
                        n_comp = st.slider("Cantidad de componentes principales", min_value=2, max_value=max_components, value=int(n_recomendado if n_recomendado is not None else n_range[best_idx]))
            if max_components < n_comp:
                st.error(f"No se puede aplicar PCA con {n_comp} componentes. El dataset tiene solo {n_samples} muestra(s) y {n_features} característica(s). El número de componentes debe ser menor o igual a {max_components}.")
            else:
                from sklearn.decomposition import PCA as _PCA
                pca = _PCA(n_components=n_comp)
                X_proj = pca.fit_transform(X_pca_scaled)
                var_exp = pca.explained_variance_ratio_
                # Mostrar porcentaje de varianza explicada por cada componente antes del gráfico de barras
                st.write("#### Porcentaje de varianza explicada por cada componente principal:")
                var_exp_pct = [f"PC{i+1}: {v*100:.2f}%" for i, v in enumerate(var_exp)]
                st.write(", ".join(var_exp_pct))
                # Gráfico de barras de varianza explicada
                from visualizaciones import plot_varianza_explicada_pca
                st.write("### Varianza explicada por cada componente")
                st.caption("La varianza explicada indica cuánta información conserva cada componente principal. Componentes con mayor varianza explicada son más importantes para representar los datos.")
                fig_bar = plot_varianza_explicada_pca(var_exp, n_comp)
                st.pyplot(fig_bar)
                # Gráfico de varianza acumulada (solo una vez, modularizado)
                from visualizaciones import plot_varianza_acumulada_pca
                st.write("### Varianza acumulada por componente")
                fig_line, codo_idx = plot_varianza_acumulada_pca(var_exp, n_comp)
                st.info(f"Recomendación automática: El método del codo sugiere usar **{codo_idx}** componentes principales. A partir de aquí, el incremento de varianza explicada es menor a 2%. Puedes ajustar según tu objetivo.")
                st.pyplot(fig_line)
                # === MATRIZ DE COMPONENTES PRINCIPALES (LOADINGS) ===
                with st.expander("🔬 Matriz de componentes principales (loadings) y explicación", expanded=False):
                    st.markdown("""
                    **¿Qué es esto?**
                    La matriz de componentes principales muestra cómo cada variable original contribuye a cada componente principal (los “loadings” o pesos).
                    - Un valor alto (positivo o negativo) indica que esa variable influye mucho en ese componente.
                    - El signo indica la dirección, pero no si “ayuda” o “perjudica” la clasificación.
                    - Los valores cercanos a cero indican poca influencia de esa variable en ese componente.
                    """)
                    # Mostrar la matriz de componentes principales (loadings)
                    loadings = pd.DataFrame(pca.components_, columns=feature_cols_pca, index=[f"PC{i+1}" for i in range(len(pca.components_))])
                    st.dataframe(loadings.style.format("{:.3f}"), use_container_width=True)
                    st.caption("Cada fila es un componente principal, cada columna es una variable original. Los valores indican la importancia (peso) de cada variable en ese componente.")
                    # Gráfico de barras de loadings para un componente seleccionado
                    st.markdown("**Visualización de los pesos (loadings) para un componente:**")
                    comp_idx = st.selectbox("Selecciona el componente para ver los pesos (loadings):", loadings.index, index=0, key="select_loading_comp")
                    fig_load, ax_load = plt.subplots(figsize=(8, 3))
                    loadings.loc[comp_idx].plot(kind='bar', ax=ax_load, color='teal', alpha=0.7)
                    ax_load.set_ylabel('Peso (loading)')
                    ax_load.set_title(f'Pesos de variables en {comp_idx}')
                    ax_load.axhline(0, color='gray', linewidth=1)
                    plt.tight_layout()
                    st.pyplot(fig_load)
                    st.caption("Las barras muestran la magnitud y dirección de la contribución de cada variable al componente seleccionado. Valores altos (positivos o negativos) indican mayor influencia.")

                    # Mostrar la combinación lineal explícita del componente seleccionado
                    st.markdown("**Combinación lineal del componente seleccionado:**")
                    coefs = loadings.loc[comp_idx]
                    terms = []
                    for var, coef in coefs.items():
                        if abs(coef) < 1e-4:
                            continue  # omitir coeficientes despreciables
                        sign = '+' if coef >= 0 else '-'
                        val = f"{abs(coef):.3f}"
                        terms.append(f"{sign} {val} × {var}")
                    if terms:
                        # El primer término puede tener signo +, lo quitamos para estética
                        if terms[0].startswith('+ '):
                            terms[0] = terms[0][2:]
                        formula = f"{comp_idx} = " + ' '.join(terms)
                        st.code(formula, language="latex")
                        st.caption("Esta expresión muestra cómo se calcula el valor del componente principal como combinación lineal de las variables originales. Los coeficientes indican el peso y la dirección de cada variable en el componente.")
                    else:
                        st.info("Este componente tiene coeficientes muy pequeños para todas las variables.")

                # === MATRIZ DE COVARIANZA Y CORRELACIÓN DE LOS COMPONENTES PRINCIPALES ===
                import seaborn as sns
                with st.expander("📐 Matriz de covarianza y correlación de los componentes principales", expanded=False):
                    st.markdown("""
                    **¿Qué es esto?**
                    - La **matriz de covarianza** muestra cómo varían conjuntamente los componentes principales. Valores altos (positivos o negativos) indican que dos componentes tienden a aumentar o disminuir juntos.
                    - La **matriz de correlación** muestra la relación lineal entre los componentes principales, normalizada entre -1 y 1.
                    - En PCA, los componentes principales son ortogonales, por lo que la matriz de correlación fuera de la diagonal debe ser cercana a cero.
                    """)
                    # Matriz de covarianza de los componentes principales
                    cov_pcs = np.cov(X_proj.T)
                    cov_df = pd.DataFrame(cov_pcs, index=[f"PC{i+1}" for i in range(X_proj.shape[1])], columns=[f"PC{i+1}" for i in range(X_proj.shape[1])])
                    st.write("#### Matriz de covarianza de los componentes principales:")
                    st.dataframe(cov_df.style.format("{:.3f}"), use_container_width=True)
                    fig_cov, ax_cov = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cov_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_cov, annot_kws={"size":8})
                    ax_cov.set_title("Heatmap matriz de covarianza (PCs)")
                    plt.tight_layout()
                    st.pyplot(fig_cov)
                    st.caption("La diagonal muestra la varianza de cada componente. Los valores fuera de la diagonal deberían ser cercanos a cero (ortogonalidad).")

                    # Matriz de correlación de los componentes principales
                    corr_pcs = np.corrcoef(X_proj.T)
                    corr_df = pd.DataFrame(corr_pcs, index=[f"PC{i+1}" for i in range(X_proj.shape[1])], columns=[f"PC{i+1}" for i in range(X_proj.shape[1])])
                    st.write("#### Matriz de correlación de los componentes principales:")
                    st.dataframe(corr_df.style.format("{:.3f}"), use_container_width=True)
                    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, annot_kws={"size":8}, vmin=-1, vmax=1)
                    ax_corr.set_title("Heatmap matriz de correlación (PCs)")
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    st.caption("La diagonal es 1 (autocorrelación perfecta). Los valores fuera de la diagonal deben ser cercanos a cero, indicando independencia lineal entre componentes.")
                    st.info("En PCA, la ortogonalidad de los componentes se refleja en matrices casi diagonales. Si ves valores altos fuera de la diagonal, revisa el preprocesamiento de los datos.")
                # ...existing code...
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
                config_plotly = {
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'pca_3d_{comp_x}_{comp_y}_{comp_z}',
                        'height': 800,
                        'width': 1200,
                        'scale': 2
                    },
                    'responsive': True
                }
                safe_plotly_chart(fig_pca, config=config_plotly, use_container_width=True)
                
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
                safe_plotly_chart(fig_pca, width='stretch')
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

# ================== TEORÍA Y GUÍA COMPLETA DE PCA ==================
with st.sidebar.expander("📖 Teoría y guía completa de PCA", expanded=False):
    st.markdown("""
### ¿Qué es PCA y para qué sirve?
El Análisis de Componentes Principales (PCA) es una técnica de reducción de dimensionalidad. Busca transformar un conjunto de variables posiblemente correlacionadas en un conjunto más pequeño de variables nuevas (componentes principales), que explican la mayor parte de la varianza de los datos.

**¿Por qué es importante?**
- Permite visualizar datos complejos en 2D/3D.
- Elimina redundancia y ruido.
- Facilita la clasificación y el clustering.

---
### ¿Qué significa la varianza en PCA?
La varianza mide cuánta información (o dispersión) de los datos conserva cada componente principal. Un componente con alta varianza explica más de la estructura original de los datos.

**Varianza explicada**: porcentaje de la información total que captura cada componente.

**Varianza acumulada**: suma de la varianza explicada por los primeros n componentes. Se usa para decidir cuántos componentes conservar.

---
### ¿Cómo interpretar los gráficos de PCA?
- **Gráfico de varianza explicada**: ayuda a decidir cuántos componentes usar (busca el “codo” de la curva).
- **Gráfico de dispersión (2D/3D)**: cada punto es una muestra proyectada en los componentes principales. Si los grupos (clases) se separan bien, es más fácil clasificarlos.
- **Solapamiento de grupos**: si los puntos de diferentes clases se mezclan, significa que esas clases tienen características similares y serán más difíciles de separar.

---
### ¿Qué es la matriz de componentes y los loadings?
La matriz de componentes muestra cómo cada variable original contribuye a cada componente principal (los “loadings” o pesos).
- Un valor alto (positivo o negativo) indica que esa variable influye mucho en ese componente.
- El signo indica la dirección, pero no si “ayuda” o “perjudica” la clasificación.

---
### ¿Cómo elegir el número de componentes?
- Tradicionalmente, se elige el número de componentes que explica al menos el 80% de la varianza acumulada.
- Mejor aún: usar la auto-selección por validación cruzada (ver sección “Guía PCA y selección automática”) para elegir el n que maximiza el rendimiento del clasificador.

---
### Preguntas frecuentes
**¿Puedo usar PCA con variables categóricas?**
No directamente. PCA requiere variables numéricas. Convierte las categóricas a numéricas primero.

**¿Qué pasa si todos los grupos se mezclan en el gráfico?**
Significa que las clases son muy similares en las variables elegidas. Prueba con otras variables o técnicas.

**¿El signo de los loadings indica si una variable ayuda o perjudica?**
No. Solo indica la dirección en el espacio de componentes. Lo importante es el valor absoluto (magnitud).

**¿Por qué a veces el mejor n de componentes no coincide con el 80% de varianza?**
Porque la varianza no siempre se traduce en mejor capacidad de clasificación. Por eso es mejor usar validación cruzada.


""")

# ================== GUÍA PCA Y SELECCIÓN AUTOMÁTICA ==================
with st.sidebar.expander("🧑‍🏫 Guía PCA y selección automática (CV)", expanded=False):
    st.markdown("""
### ¿Qué es la auto-selección de componentes PCA por validación cruzada (CV)?

Esta funcionalidad te ayuda a elegir **cuántos componentes principales (n)** usar en PCA, pero de forma **objetiva y automática**, basándose en el rendimiento real de un modelo de clasificación (no solo en la varianza explicada).

**¿Cómo funciona?**
1. Escala los datos.
2. Para cada posible n (número de componentes):
   - Aplica PCA con ese n.
   - Proyecta los datos.
   - Entrena y evalúa un clasificador (por defecto, LogisticRegression) usando validación cruzada (CV).
   - Guarda el promedio de la métrica elegida (accuracy o f1_macro).
3. Busca el n más pequeño cuya media de score esté dentro de la tolerancia (por ejemplo, 1%) del mejor resultado observado.
4. Te muestra una gráfica: eje X = n, eje Y = score. Marca el n recomendado y el n con mejor score.
5. Actualiza el slider de componentes con el n recomendado.

---
### ¿Qué es un clasificador? ¿Por qué LogisticRegression?
- Un **clasificador** es un modelo que predice a qué clase pertenece cada muestra (por ejemplo, género musical, tipo de vino, etc.).
- **LogisticRegression** es un modelo simple y estándar, ideal para comparar y medir la separabilidad de los datos tras PCA.
- No es el modelo final, solo una “regla de evaluación” para comparar cuántos componentes usar.

---
### ¿Qué significan las métricas?
- **accuracy**: Porcentaje de aciertos totales. Útil si las clases están balanceadas.
- **f1_macro**: Promedio del F1-score de cada clase. Mejor si tienes clases desbalanceadas o te importa el rendimiento en todas las clases por igual.

---
### ¿Qué es la tolerancia?
- Es el margen de “flexibilidad” para elegir el número de componentes.
- Ejemplo: si el mejor score se logra con 14 componentes, pero con 13 el score es solo 1% menor, la tolerancia de 0.01 (1%) permite elegir 13 (menos dimensiones, casi mismo rendimiento).

---
### ¿Por qué usar este método?
- Así eliges el número de componentes que realmente maximiza (o casi) el rendimiento de tu clasificador, no solo la varianza explicada.
- Evitas usar más dimensiones de las necesarias (menos sobreajuste, más interpretabilidad).
- Es un método objetivo, reproducible y adaptado a tu dataset y problema de clasificación.

---
### Consejos para el parcial
- Si te preguntan “¿cuántos componentes usar?”, responde: “Elijo el n que maximiza el rendimiento de mi clasificador según validación cruzada, usando f1_macro si hay desbalance de clases, o accuracy si no”.
- Si te piden justificar: “No me baso solo en la varianza explicada, sino en el rendimiento real del modelo sobre los datos proyectados”.
- Puedes mostrar la gráfica de score vs n y explicar cómo se eligió el n recomendado.

---
### ¿Quieres más ejemplos o analogías? ¡Pregúntame!
""")

with st.sidebar.expander("📖 Guía de uso"):
    st.markdown(GUIA_USO)

with st.sidebar.expander("🎯 Métricas explicadas"):
    st.markdown(METRICAS_EXPLICADAS)


with st.sidebar.expander("🧮 Bayes Ingenuo explicado"):
    st.markdown(BAYES_EXPLICADO)

with st.sidebar.expander("🔍 Interpretación de PCA"):
    st.markdown(PCA_SIDEBAR)

with st.sidebar.expander("⚙️ Configuración avanzada"):
    st.markdown(CONFIG_AVANZADA)

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
