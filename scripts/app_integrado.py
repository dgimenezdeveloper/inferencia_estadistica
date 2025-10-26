import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

 
 
 
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
import streamlit as st
st.set_page_config(layout="wide")
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

TEXTO_SVM = """
### 🧠 Máquinas de Vectores de Soporte (SVM)

Las SVM son algoritmos supervisados para clasificación y regresión. Buscan el hiperplano que maximiza el margen entre clases. Si los datos no son separables linealmente, usan kernels para proyectarlos a espacios de mayor dimensión.

**¿Cuándo usar SVM?**
- Cuando tienes datos con fronteras complejas o no lineales.
- Cuando necesitas robustez ante outliers (con C bajo).
- Cuando el número de variables es alto respecto a las muestras.

**Parámetros clave:**
- **Kernel:** 'linear', 'rbf', 'poly', 'sigmoid'.
- **C:** Controla la penalización por errores (regularización).
- **Gamma:** Afecta la flexibilidad del modelo (solo 'rbf', 'poly', 'sigmoid').
- **Degree:** Grado del polinomio (solo 'poly').

**Ventajas:**
- Potente para problemas complejos.
- Puede manejar datos no lineales.
- Robusto ante overfitting si se ajusta bien C y kernel.

**Desventajas:**
- Sensible a la escala de los datos (escalar siempre).
- Puede ser lento con muchos datos.
- Difícil de interpretar para kernels no lineales.

**Recomendaciones:**
- Escala siempre las variables antes de usar SVM.
- Prueba varios kernels y valores de C/gamma.
- Usa validación cruzada para elegir hiperparámetros.
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

# === Selector principal con opción de inicio ===
analisis = st.sidebar.selectbox(
    "Selecciona el tipo de análisis",
    [
        "Inicio",
        "Exploración de datos",
        "Discriminante (LDA/QDA)",
        "Bayes Ingenuo",
        "Regresión Logística",
        "Reducción de dimensiones (PCA)",
        "K-means (Clustering)",
        "SVM (Máquinas de Vectores de Soporte)",
        "Comparativa de Modelos"
    ]
)


# === Selección de archivo y columna de clase global (antes de todo) ===
carpeta_datos = os.path.join(os.path.dirname(__file__), '..', 'datos')
archivos_csv = [f for f in os.listdir(carpeta_datos) if f.endswith('.csv')] if os.path.exists(carpeta_datos) else []
st.sidebar.markdown("---")
st.sidebar.write("### 1️⃣ Selecciona el archivo de datos")
opcion_archivo = st.sidebar.selectbox("Seleccionar archivo CSV", archivos_csv)
archivo_subido = st.sidebar.file_uploader("O sube tu propio archivo CSV", type=["csv"])

# Cargar el dataset (modularizado)
df, _msg_carga = cargar_dataset(archivo_subido, opcion_archivo, carpeta_datos)
if _msg_carga:
    st.sidebar.success(_msg_carga)

# Selección global de columna de clase y nombres descriptivos
if df is not None:
    max_unique_target = 20
    num_cols_global, cat_cols_global = seleccionar_columnas(df, max_unique_target=max_unique_target)
    if cat_cols_global:
        st.sidebar.markdown("---")
        st.sidebar.write("### 2️⃣ Selecciona la columna de clase y asigna nombres descriptivos")
        target_col_global = st.sidebar.selectbox("Columna de clase (target):", cat_cols_global, key="target_global_sidebar")
        clase_unicos_global = sorted(df[target_col_global].unique())
        clase_labels_global = st.session_state.get("clase_labels_global", {})
        for v in clase_unicos_global:
            label = st.sidebar.text_input(f"Nombre descriptivo para '{v}'", value=clase_labels_global.get(v, str(v)), key=f"sidebar_label_{v}")
            clase_labels_global[v] = label if label.strip() else str(v)
        st.session_state["target_col_global"] = target_col_global
        st.session_state["clase_labels_global"] = clase_labels_global
        st.sidebar.write("#### Vista previa de clases:")
        conteo = df[target_col_global].value_counts().sort_index()
        nombres = [clase_labels_global.get(v, str(v)) for v in conteo.index]
        st.sidebar.dataframe(pd.DataFrame({"Clase": nombres, "Cantidad": conteo.values}))
        st.sidebar.caption("Estos nombres descriptivos se usarán en todas las secciones de la app.")
    else:
        st.sidebar.warning("No se detectaron columnas categóricas elegibles como clase. Elige un archivo adecuado.")

# El resto de la app usará st.session_state["target_col_global"] y st.session_state["clase_labels_global"]


if analisis == "Inicio":
    st.title("Inicio: Fundamentos para un Análisis Estadístico y de Patrones Correcto")
    st.markdown("""
    ## ¿Cómo abordar un problema de análisis estadístico y clasificación?
    
    El proceso de análisis no es solo aplicar algoritmos, sino **razonar** y **justificar** cada decisión. Aquí tienes una guía conceptual para elegir y aplicar correctamente los métodos:
    """)
    st.markdown("""
    ---
    ### 1. Entiende el problema y los datos
    - ¿Cuál es el objetivo? (clasificar, predecir, explorar)
    - ¿Qué representa cada variable? ¿Qué significa la variable de clase?
    - ¿Las variables son numéricas, categóricas, ordinales?
    - ¿Hay valores atípicos, nulos o errores?
    
    **Ejemplo:** Si tu objetivo es predecir el tipo de vino según características químicas, asegúrate de entender qué mide cada variable y si tiene sentido biológico/químico.
    """)
    st.markdown("""
    ---
    ### 2. Analiza la estructura y relaciones entre variables
    - Observa la **matriz de correlación**: ¿hay variables muy correlacionadas? Esto afecta a Bayes Ingenuo y puede motivar el uso de PCA.
    - Observa la **matriz de covarianza**: ¿las escalas y varianzas son similares entre clases? Esto es clave para LDA/QDA.
    - ¿Las clases están balanceadas? Si no, elige métricas adecuadas (f1_macro, balanced accuracy).
    
    **Ejemplo:** Si dos variables tienen correlación 0.98, Bayes Ingenuo no es recomendable salvo que uses PCA.
    """)
    st.markdown("""
    ---
    ### 3. Conoce los supuestos y fundamentos de cada algoritmo
    - **LDA (Análisis Discriminante Lineal):**
        - Supone que las clases tienen **covarianzas iguales** y que las variables siguen una distribución normal multivariante.
        - Es robusto si los datos cumplen estos supuestos y las clases están bien separadas linealmente.
        - Útil para interpretación y visualización.
    - **QDA (Análisis Discriminante Cuadrático):**
        - Permite **covarianzas diferentes** por clase.
        - Más flexible, pero requiere más datos para estimar bien las matrices.
        - Puede sobreajustar si hay pocas muestras por clase.
    - **Bayes Ingenuo:**
        - Supone **independencia condicional** entre variables dado la clase.
        - Muy eficiente y rápido, pero sensible a correlaciones fuertes.
        - Funciona bien con muchas variables si la independencia es razonable.
    - **PCA (Análisis de Componentes Principales):**
        - No es un clasificador, sino una técnica para **reducir la dimensionalidad**.
        - Útil si hay muchas variables o alta correlación/redundancia.
        - Puede mejorar la estabilidad de los modelos y reducir el sobreajuste.
    """)
    st.markdown("""
    ---
    ### 4. ¿Cómo decidir qué método usar?
    - **¿Las variables están muy correlacionadas?**
        - Sí: Considera PCA antes de clasificar, o elimina variables redundantes.
        - No: Puedes usar LDA, QDA, SVM o Bayes Ingenuo según los otros supuestos.
    - **¿Las clases tienen covarianzas similares?**
        - Sí: LDA es apropiado.
        - No: QDA o SVM pueden capturar mejor la diferencia.
    - **¿Las variables son independientes?**
        - Sí: Bayes Ingenuo es ideal.
        - No: Prefiere LDA/QDA/SVM o usa PCA antes de Bayes.
    - **¿Tienes muchas variables y pocas muestras?**
        - Sí: PCA ayuda a evitar sobreajuste. SVM puede funcionar bien con muchas variables, pero es sensible a la escala.
    - **¿Qué métrica te importa más?**
        - Si las clases están desbalanceadas, usa f1_macro o balanced accuracy.
    - **¿Dataset muy grande (>10k muestras)?**
        - SVM puede ser lento. Considera muestreo estratificado o algoritmos alternativos.

    **Ejemplo de razonamiento:**
    > "Tengo 10 variables, 3 de ellas muy correlacionadas. Las clases parecen tener varianzas distintas. Probaré QDA y SVM, pero antes aplicaré PCA para reducir la redundancia."
    """)
    st.markdown("""
    ---
    ### 5. Buenas prácticas y advertencias teóricas
    - **No apliques algoritmos sin revisar los supuestos.**
    - **No elimines variables solo por correlación:** verifica el impacto real en las métricas.
    - **Justifica cada decisión:** ¿por qué elegiste ese modelo, ese preprocesamiento?
    - **Compara siempre varios modelos:** no te quedes solo con el accuracy.
    - **No te fíes solo de la varianza explicada en PCA:** valida con métricas de clasificación.
    - **Si los resultados no tienen sentido, revisa los datos y los supuestos.**
    - **SVM es sensible a la escala:** Escala siempre las variables numéricas antes de entrenar SVM.
    - **SVM puede ser lento con muchos datos:** Usa muestreo estratificado o selecciona un subconjunto representativo si tienes más de 10,000 muestras.
    - **Para visualizar fronteras de decisión en SVM:** Elige las dos variables más relevantes (puedes usar la sugerencia automática en la app).
    - **SVM funciona bien con variables no linealmente separables usando kernels no lineales (RBF, poly).**
    
    **Recuerda:** El análisis correcto es el que puedes justificar teóricamente y que se adapta a la naturaleza de tus datos.
    """)
    st.success("¡Listo! Usa el menú de la izquierda para explorar cada método, revisa los supuestos y justifica tus decisiones.")

elif analisis == "SVM (Máquinas de Vectores de Soporte)" and df is not None:
    st.title("🧠 SVM: Máquinas de Vectores de Soporte")

    with st.expander("ℹ️ Guía de uso y significado de cada opción", expanded=True):
        st.markdown("""
        ### 🧭 **¿Cómo usar esta vista de SVM?**
        
        Aquí puedes entrenar y comparar modelos SVM de forma interactiva. Cada opción tiene un propósito:
        
        - **Archivo de datos**: Selecciona el dataset sobre el que quieres trabajar.
        - **Columna de clase y nombres**: Elige la variable a predecir y asigna nombres descriptivos a las clases.
        - **Estrategia de entrenamiento**:
            - *Todo el dataset*: Usa todas las filas (puede ser lento si hay muchas).
            - *Muestreo estratificado*: Selecciona una muestra representativa, manteniendo proporciones de clase (recomendado para datasets grandes).
            - *Muestra aleatoria rápida*: Selecciona una muestra aleatoria para pruebas rápidas.
        - **Selección de variables**: Elige qué columnas usar como predictores. Prueba diferentes combinaciones para ver su impacto.
        - **Preprocesamiento**: Escala los datos para que SVM funcione correctamente.
        - **Configuración del modelo**:
            - *Kernel*: Elige la función de separación (linear, rbf, poly, sigmoid).
            - *C*: Controla la regularización (C bajo = menos sobreajuste, C alto = más ajuste).
            - *Gamma*: Influencia de cada muestra (solo para algunos kernels).
            - *Degree*: Grado del polinomio (solo para kernel poly).
        - **Optimización automática (Grid Search)**: Busca la mejor combinación de parámetros automáticamente.
        - **Comparación de kernels**: Compara el rendimiento de los 4 kernels principales.
        - **Entrenamiento y métricas**: Entrena el modelo y revisa las métricas de rendimiento.
        - **Visualización de fronteras**: Si eliges solo 2 variables, verás gráficamente cómo el modelo separa las clases.
        
        **Recomendaciones:**
        - Para análisis rápido, usa muestreo estratificado de 5k-10k muestras.
        - Para análisis final, usa todo el dataset si es posible.
        - Siempre escala los datos antes de entrenar SVM.
        - Compara kernels y ajusta parámetros para encontrar el mejor modelo.
        """)

    with st.expander("📚 ¿Qué es SVM? Fundamentos teóricos", expanded=False):
        st.markdown("""
        ### Conceptos fundamentales de SVM
        
        **SVM (Support Vector Machine)** es un algoritmo de aprendizaje supervisado que busca encontrar el **hiperplano óptimo** que separa las clases maximizando el **margen** (distancia entre el hiperplano y los puntos más cercanos de cada clase).
        
        **Conceptos clave:**
        - **Vectores de soporte**: puntos más cercanos al hiperplano de decisión
        - **Margen**: distancia entre el hiperplano y los vectores de soporte
        - **Kernel**: función que transforma los datos a un espacio de mayor dimensión
        - **Hiperparámetro C**: controla el trade-off entre margen y errores de clasificación
        
        **¿Cuándo usar SVM?**
        - Datos con fronteras complejas no lineales
        - Cuando el número de características es mayor que el de muestras
        - Cuando necesitas robustez ante outliers
        - Para problemas de clasificación binaria o multiclase
        """)

    st.info("🎯 **Objetivo didáctico**: Entender cómo diferentes kernels y parámetros afectan la clasificación, visualizar fronteras de decisión y comparar rendimiento.")
    
    # Selección de variables y target
    target_col = st.session_state.get("target_col_global")
    clase_labels = st.session_state.get("clase_labels_global")
    if not target_col or target_col not in df.columns:
        st.warning("⚠️ Selecciona una columna de clase en el panel izquierdo.")
    else:
        # Preparación de datos
        y = df[target_col]
        feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        
        if len(feature_cols) < 2:
            st.error("❌ SVM requiere al menos 2 variables numéricas. Agrega más columnas numéricas a tu dataset.")
        else:
            st.success(f"✅ Dataset cargado: {len(df)} muestras, {len(feature_cols)} variables numéricas, {len(y.unique())} clases")
            
            # --- ESTRATEGIA INTELIGENTE DE MUESTREO PARA SVM ---
            df_svm = df.copy()
            muestreo_aplicado = False
            
            if len(df) > 10000:
                st.warning(f"⚠️ **Dataset grande detectado**: {len(df)} muestras")
                st.markdown("""
                **SVM y datasets grandes:**
                - SVM tiene complejidad O(n²) a O(n³), se vuelve muy lento con muchos datos
                - Con >50k muestras puede tardar horas o colgarse
                - **Alternativas inteligentes**: muestreo estratificado, selección representativa
                """)
                
                # Opciones de muestreo más inteligentes
                estrategia_muestreo = st.radio(
                    "**Estrategia de entrenamiento:**",
                    [
                        "🚀 Usar todo el dataset (puede ser lento)",
                        "🎯 Muestreo estratificado inteligente (recomendado)", 
                        "⚡ Muestra aleatoria rápida"
                    ],
                    index=1,
                    key="estrategia_svm"
                )
                
                if estrategia_muestreo.startswith("🎯"):
                    # Muestreo estratificado inteligente
                    st.info("**Muestreo estratificado**: Mantiene la proporción de cada clase, preserva la estructura del dataset")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        tamaño_muestra = st.slider(
                            "Tamaño de la muestra estratificada", 
                            min_value=2000, 
                            max_value=min(20000, len(df)), 
                            value=min(8000, len(df)), 
                            step=1000,
                            key="tamaño_estratificado"
                        )
                    with col2:
                        # Mostrar distribución actual de clases
                        class_dist = y.value_counts()
                        st.write("**Distribución de clases actual:**")
                        for clase, count in class_dist.items():
                            clase_nombre = clase_labels.get(clase, str(clase)) if clase_labels else str(clase)
                            porcentaje = (count / len(y)) * 100
                            st.write(f"- {clase_nombre}: {count} ({porcentaje:.1f}%)")
                    
                    # Aplicar muestreo estratificado
                    if tamaño_muestra >= len(df):
                        # Si el tamaño de muestra es igual o mayor al dataset, usar todo
                        df_svm = df.copy()
                        y = df_svm[target_col]
                        st.info(f"✅ Usando todo el dataset: {len(df_svm)} muestras (tamaño solicitado >= dataset)")
                    else:
                        # Aplicar muestreo estratificado normal
                        from sklearn.model_selection import train_test_split
                        test_fraction = tamaño_muestra / len(df)
                        
                        # Asegurar que test_size esté en rango válido (0.0, 1.0)
                        if test_fraction >= 1.0:
                            test_fraction = 0.99  # Usar 99% como máximo
                        elif test_fraction <= 0.0:
                            test_fraction = 0.01  # Usar 1% como mínimo
                        
                        _, df_sampled, _, y_sampled = train_test_split(
                            df, y, 
                            test_size=test_fraction, 
                            stratify=y, 
                            random_state=42
                        )
                        df_svm = df_sampled
                        y = y_sampled
                        st.success(f"✅ Muestreo estratificado aplicado: {len(df_svm)} muestras manteniendo proporciones de clase")
                    
                    muestreo_aplicado = True
                    
                elif estrategia_muestreo.startswith("⚡"):
                    # Muestra aleatoria simple
                    tamaño_rapido = st.slider(
                        "Tamaño de muestra aleatoria", 
                        min_value=1000, 
                        max_value=min(10000, len(df)), 
                        value=min(5000, len(df)), 
                        step=500,
                        key="tamaño_rapido"
                    )
                    
                    # Asegurar que no se pida más muestras de las disponibles
                    tamaño_efectivo = min(tamaño_rapido, len(df))
                    
                    if tamaño_efectivo >= len(df):
                        df_svm = df.copy()
                        y = df_svm[target_col]
                        st.info(f"✅ Usando todo el dataset: {len(df_svm)} muestras (tamaño solicitado >= dataset)")
                    else:
                        df_svm = df.sample(n=tamaño_efectivo, random_state=42)
                        y = df_svm[target_col]
                        st.warning(f"⚡ Muestra aleatoria aplicada: {len(df_svm)} muestras (puede alterar distribución de clases)")
                    
                    muestreo_aplicado = True
                
                else:
                    # Usar todo el dataset con advertencia
                    st.error(f"⚠️ **ADVERTENCIA**: Usar {len(df)} muestras puede ser MUY lento (estimado: >10 minutos)")
                    confirmar = st.checkbox("Confirmo que quiero usar todo el dataset (bajo mi responsabilidad)", key="confirmar_completo")
                    if not confirmar:
                        st.stop()
            
            elif len(df) > 5000:
                st.info(f"📊 Dataset mediano: {len(df)} muestras. SVM debería funcionar bien, pero puede tardar algunos minutos.")
                usar_muestreo = st.checkbox(f"🎯 Aplicar muestreo estratificado de 5000 muestras (más rápido)", value=False, key="muestreo_opcional")
                if usar_muestreo:
                    from sklearn.model_selection import train_test_split
                    test_fraction = 5000 / len(df)
                    
                    # Asegurar que test_size esté en rango válido
                    if test_fraction >= 1.0:
                        # Si 5000 >= len(df), usar todo el dataset
                        df_svm = df.copy()
                        y = df_svm[target_col]
                        st.info("✅ Usando todo el dataset (5000 >= tamaño actual)")
                    else:
                        _, df_sampled, _, y_sampled = train_test_split(
                            df, y, test_size=test_fraction, stratify=y, random_state=42
                        )
                        df_svm = df_sampled
                        y = y_sampled
                        st.success("✅ Muestreo estratificado aplicado: 5000 muestras")
                    
                    muestreo_aplicado = True
            
            # Mostrar información final del dataset a usar
            if muestreo_aplicado:
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Dataset original", f"{len(df):,} muestras")
                    st.metric("🎯 Dataset para SVM", f"{len(df_svm):,} muestras")
                with col2:
                    reduccion = (1 - len(df_svm)/len(df)) * 100
                    st.metric("📉 Reducción", f"{reduccion:.1f}%")
                    # Verificar si se mantuvieron las proporciones
                    if len(y.unique()) == len(df[target_col].unique()):
                        st.success("✅ Todas las clases preservadas")
                    else:
                        st.warning("⚠️ Algunas clases se perdieron en el muestreo")
                
                # Información educativa sobre el muestreo
                with st.expander("🎓 **¿Pierdo información con el muestreo? (Clic para aprender)**", expanded=False):
                    st.markdown("""
                    ### 🤔 ¿Realmente pierdo información importante?
                    
                    **La respuesta corta: Depende del tipo de muestreo y la naturaleza de tus datos.**
                    
                    #### ✅ **Muestreo estratificado (recomendado)**
                    - **Preserva la estructura**: Mantiene las proporciones de cada clase
                    - **Representativo**: Si tu dataset es homogéneo, una muestra estratificada de 5k-10k puede ser tan informativa como el dataset completo
                    - **Validez estadística**: Estudios muestran que muestras estratificadas >1000 por clase suelen ser representativas
                    
                    #### ⚠️ **Muestreo aleatorio simple**
                    - **Riesgo de sesgo**: Puede sobrerepresentar o subrepresentar algunas clases
                    - **Pérdida de patrones raros**: Puede perder muestras de clases minoritarias
                    
                    #### 📊 **¿Cuándo el muestreo es aceptable?**
                    - **Datos redundantes**: Si tienes muchas muestras similares
                    - **Clases balanceadas**: Con >1000 muestras por clase en la muestra
                    - **Objetivo exploratorio**: Para análisis inicial antes del modelo final
                    
                    #### ❌ **Cuándo NO hacer muestreo:**
                    - **Clases muy desbalanceadas**: Puedes perder clases minoritarias
                    - **Datos únicos**: Cada muestra aporta información valiosa
                    - **Análisis final**: Para el modelo de producción usa todos los datos
                    
                    #### 🧠 **Recomendación práctica:**
                    1. **Exploración inicial**: Usa muestreo estratificado para entender el problema
                    2. **Ajuste de hiperparámetros**: Usa toda la data o validación cruzada
                    3. **Modelo final**: Entrena con todos los datos disponibles
                    
                    **💡 Tip**: Si SVM es muy lento, considera algoritmos alternativos como Random Forest o Gradient Boosting que escalan mejor.
                    """)
            
            # Selección de variables para el análisis
            st.subheader("🔍 1. Selección de variables")
            cols1, cols2 = st.columns(2)
            
            with cols1:
                st.write("**Variables disponibles:**")
                # --- Sugerencia automática de variables ---
                if 'svm_sugeridas' not in st.session_state:
                    st.session_state['svm_sugeridas'] = feature_cols[:2]

                col_sug1, col_sug2 = st.columns([2,1])
                with col_sug1:
                    # Filtrar sugeridas para que estén en feature_cols
                    sugeridas_validas = [v for v in st.session_state['svm_sugeridas'] if v in feature_cols]
                    # Si no hay sugeridas válidas, usar primeras dos
                    if len(sugeridas_validas) < 2:
                        sugeridas_validas = feature_cols[:2]
                        st.session_state['svm_sugeridas'] = sugeridas_validas
                    selected_features = st.multiselect(
                        "Selecciona las variables para el modelo",
                        feature_cols,
                        default=sugeridas_validas,
                        key="svm_features"
                    )
                with col_sug2:
                    if st.button("🔎 Sugerir mejores variables", key="btn_sugerir_vars"):
                        try:
                            from sklearn.feature_selection import SelectKBest, f_classif
                            import numpy as np
                            # Prepara X numérico y y
                            X_sug = df_svm[feature_cols]
                            y_sug = y
                            # Imputar nulos si hay
                            if X_sug.isnull().sum().sum() > 0:
                                from preprocesamiento import manejar_nulos
                                X_sug = manejar_nulos(X_sug, metodo='media')
                            # SelectKBest con ANOVA F-score
                            selector = SelectKBest(score_func=f_classif, k=2)
                            selector.fit(X_sug, y_sug)
                            idxs = np.argsort(selector.scores_)[::-1][:2]
                            mejores_vars = [feature_cols[i] for i in idxs]
                            st.session_state['svm_sugeridas'] = mejores_vars
                            st.success(f"Variables sugeridas: {mejores_vars[0]} y {mejores_vars[1]}")
                        except Exception as e:
                            st.error(f"No se pudo sugerir variables automáticamente: {str(e)}")
            
            with cols2:
                st.write("**Información del dataset:**")
                st.metric("Total de muestras", len(df_svm))
                st.metric("Variables numéricas", len(feature_cols))
                st.metric("Clases únicas", len(y.unique()))
                
                # Balance de clases
                class_counts = y.value_counts()
                balance_ratio = class_counts.min() / class_counts.max()
                if balance_ratio < 0.3:
                    st.warning(f"⚠️ Clases desbalanceadas (ratio: {balance_ratio:.2f})")
                else:
                    st.success(f"✅ Clases balanceadas (ratio: {balance_ratio:.2f})")
            
            if len(selected_features) >= 2:
                X = df_svm[selected_features]

                # Crear columna temporal con nombres descriptivos de clase
                if clase_labels:
                    df_svm = df_svm.copy()
                    df_svm['Clase_desc'] = df_svm[target_col].map(clase_labels)
                    color_col = 'Clase_desc'
                    color_label = 'Clase'
                else:
                    color_col = target_col
                    color_label = target_col

                # === Gráficos adicionales: histogramas y dispersión ===
                st.subheader("📊 Visualización de variables seleccionadas")
                col_hist, col_scat = st.columns([2, 3])
                with col_hist:
                    st.markdown("**Histogramas de variables**")
                    import plotly.express as px
                    for col in selected_features:
                        fig_hist = px.histogram(df_svm, x=col, color=color_col, nbins=20,
                            title=f"Histograma de {col}",
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                            labels={color_col: color_label})
                        fig_hist.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_hist, use_container_width=True)
                with col_scat:
                    if len(selected_features) >= 2:
                        st.markdown("**Dispersión de las dos primeras variables**")
                        fig_scat = px.scatter(
                            df_svm,
                            x=selected_features[0],
                            y=selected_features[1],
                            color=color_col,
                            symbol=color_col,
                            title=f"Dispersión: {selected_features[0]} vs {selected_features[1]}",
                            color_discrete_sequence=px.colors.qualitative.Plotly,
                            labels={color_col: color_label}
                        )
                        fig_scat.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
                        fig_scat.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_scat, use_container_width=True)
                    else:
                        st.info("Selecciona al menos 2 variables para ver el gráfico de dispersión.")

                # Verificar y manejar nulos
                if X.isnull().sum().sum() > 0:
                    st.warning("⚠️ Se encontraron valores nulos. Imputando con la media...")
                    from preprocesamiento import manejar_nulos
                    X = manejar_nulos(X, metodo='media')

                # Escalado de datos
                st.subheader("⚙️ 2. Preprocesamiento")
                escalar_datos_svm = st.checkbox("Escalar datos (recomendado para SVM)", value=True, key="escalar_svm")

                if escalar_datos_svm:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_model = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                    st.info("✅ Datos escalados (media=0, std=1)")
                else:
                    X_model = X.copy()
                    st.warning("⚠️ Datos sin escalar. SVM es sensible a la escala.")
                
                # Configuración del modelo
                st.subheader("🛠️ 3. Configuración del modelo SVM")
                
                # Pestañas para configuración básica y avanzada
                tab_basic, tab_advanced, tab_comparison = st.tabs(["🎯 Configuración Básica", "⚙️ Configuración Avanzada", "📊 Comparación de Kernels"])
                
                with tab_basic:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key="kernel_basic")
                        st.caption("**RBF**: Fronteras curvas, versátil\n**Linear**: Fronteras rectas, rápido\n**Poly**: Fronteras polinómicas\n**Sigmoid**: Similar a redes neuronales")
                    
                    with col2:
                        C = st.slider("C (Regularización)", min_value=0.001, max_value=100.0, value=1.0, step=0.001, format="%.3f", key="C_basic")
                        st.caption("**C bajo**: Margen amplio, puede subajustar\n**C alto**: Margen estrecho, puede sobreajustar")
                    
                    with col3:
                        if kernel in ['rbf', 'poly', 'sigmoid']:
                            gamma = st.selectbox("Gamma", ["scale", "auto", "manual"], index=0, key="gamma_basic")
                            if gamma == "manual":
                                gamma_value = st.slider("Valor de Gamma", min_value=0.001, max_value=10.0, value=1.0, step=0.001, key="gamma_manual")
                                gamma = gamma_value
                        else:
                            gamma = 'scale'
                            st.info("Gamma no aplica para kernel linear")
                        
                        if kernel == 'poly':
                            degree = st.slider("Degree (Grado)", min_value=2, max_value=6, value=3, key="degree_basic")
                        else:
                            degree = 3
                
                with tab_advanced:
                    st.markdown("### 🔬 Optimización automática de hiperparámetros")
                    
                    enable_grid_search = st.checkbox("Activar Grid Search (búsqueda automática)", value=False, key="enable_grid_search")
                    
                    if enable_grid_search:
                        st.info("🤖 Grid Search probará diferentes combinaciones de parámetros y elegirá la mejor según validación cruzada.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            kernels_grid = st.multiselect("Kernels a probar", ["linear", "rbf", "poly", "sigmoid"], default=["linear", "rbf"], key="kernels_grid")
                            cv_folds = st.slider("Folds de validación cruzada", min_value=3, max_value=10, value=5, key="cv_folds")
                        
                        with col2:
                            C_values = st.multiselect("Valores de C", [0.001, 0.01, 0.1, 1, 10, 100], default=[0.1, 1, 10], key="C_values")
                            scoring_metric = st.selectbox("Métrica de optimización", ["accuracy", "f1_macro", "f1_weighted"], key="scoring_metric")
                
                with tab_comparison:
                    st.markdown("### 📈 Comparación automática de todos los kernels")
                    st.info("Esta opción entrena modelos con los 4 kernels principales y compara su rendimiento.")
                    
                    enable_comparison = st.checkbox("Activar comparación de kernels", value=False, key="enable_comparison")
                    if enable_comparison:
                        comparison_metric = st.selectbox("Métrica para comparación", ["accuracy", "f1_macro", "f1_weighted"], key="comparison_metric")
                
                # Botón de entrenamiento
                st.markdown("---")
                if st.button("🚀 Entrenar modelo SVM", key="train_svm", type="primary"):
                    with st.spinner("🔄 Entrenando modelo SVM..."):
                        import time
                        from sklearn.model_selection import GridSearchCV, cross_val_score
                        from sklearn.metrics import classification_report
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        try:
                            if enable_grid_search:
                                # Grid Search
                                st.subheader("🤖 Resultados de Grid Search")
                                
                                param_grid = {
                                    'kernel': kernels_grid,
                                    'C': C_values
                                }
                                
                                # Agregar parámetros específicos por kernel
                                if 'rbf' in kernels_grid or 'poly' in kernels_grid or 'sigmoid' in kernels_grid:
                                    param_grid['gamma'] = ['scale', 'auto']
                                if 'poly' in kernels_grid:
                                    param_grid['degree'] = [2, 3, 4]
                                
                                from sklearn.svm import SVC
                                svm_base = SVC(probability=True, random_state=42)
                                
                                grid_search = GridSearchCV(svm_base, param_grid, cv=cv_folds, scoring=scoring_metric, n_jobs=-1)
                                
                                start_time = time.time()
                                grid_search.fit(X_model, y)
                                training_time = time.time() - start_time
                                
                                st.success(f"✅ Grid Search completado en {training_time:.2f} segundos")
                                
                                # Mejores parámetros
                                st.write("**🏆 Mejores parámetros encontrados:**")
                                for param, value in grid_search.best_params_.items():
                                    st.write(f"- **{param}**: {value}")
                                
                                st.metric("Mejor score (CV)", f"{grid_search.best_score_:.4f}")
                                
                                # Usar el mejor modelo
                                best_model = grid_search.best_estimator_
                                model_params = grid_search.best_params_
                                
                            elif enable_comparison:
                                # Comparación de kernels
                                st.subheader("📊 Comparación de kernels")
                                
                                kernels_to_compare = ["linear", "rbf", "poly", "sigmoid"]
                                comparison_results = []
                                
                                for kern in kernels_to_compare:
                                    from modelos import entrenar_svm
                                    temp_model = entrenar_svm(X_model, y, kernel=kern, C=1.0, probability=True)
                                    scores = cross_val_score(temp_model, X_model, y, cv=5, scoring=comparison_metric)
                                    
                                    comparison_results.append({
                                        'Kernel': kern,
                                        'Score Promedio': scores.mean(),
                                        'Desviación Estándar': scores.std(),
                                        'Score Mínimo': scores.min(),
                                        'Score Máximo': scores.max()
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_results)
                                comparison_df = comparison_df.sort_values('Score Promedio', ascending=False)
                                
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Gráfico de comparación
                                fig_comparison = go.Figure()
                                fig_comparison.add_trace(go.Bar(
                                    x=comparison_df['Kernel'],
                                    y=comparison_df['Score Promedio'],
                                    error_y=dict(type='data', array=comparison_df['Desviación Estándar']),
                                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                                ))
                                fig_comparison.update_layout(
                                    title=f"Comparación de Kernels ({comparison_metric})",
                                    xaxis_title="Kernel",
                                    yaxis_title=f"{comparison_metric.capitalize()}",
                                    showlegend=False
                                )
                                st.plotly_chart(fig_comparison, use_container_width=True)
                                
                                # Usar el mejor kernel
                                best_kernel_row = comparison_df.iloc[0]
                                st.success(f"🏆 Mejor kernel: **{best_kernel_row['Kernel']}** (score: {best_kernel_row['Score Promedio']:.4f})")
                                
                                from modelos import entrenar_svm
                                best_model = entrenar_svm(X_model, y, kernel=best_kernel_row['Kernel'], C=1.0, probability=True)
                                model_params = {'kernel': best_kernel_row['Kernel'], 'C': 1.0}
                                
                            else:
                                # Entrenamiento básico
                                from modelos import entrenar_svm
                                start_time = time.time()
                                best_model = entrenar_svm(X_model, y, kernel=kernel, C=C, gamma=gamma, degree=degree, probability=True)
                                training_time = time.time() - start_time
                                model_params = {'kernel': kernel, 'C': C, 'gamma': gamma, 'degree': degree}
                                
                                st.success(f"✅ Modelo entrenado en {training_time:.3f} segundos")
                            
                            # Predicciones
                            from modelos import predecir_svm
                            y_pred, y_prob = predecir_svm(best_model, X_model)
                            
                            # Métricas principales
                            st.subheader("📊 4. Métricas de rendimiento")
                            
                            # Métricas en columnas
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                            from metricas import calcular_metricas_clasificacion, mostrar_metricas_clasificacion
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                acc = accuracy_score(y, y_pred)
                                st.metric("Accuracy", f"{acc:.4f}", f"{acc*100:.1f}%")
                            with col2:
                                f1 = f1_score(y, y_pred, average='macro')
                                st.metric("F1-Score (macro)", f"{f1:.4f}")
                            with col3:
                                st.metric("Vectores de soporte", len(best_model.support_))
                                st.caption(f"{len(best_model.support_)/len(X_model)*100:.1f}% del dataset")
                            with col4:
                                if not (enable_grid_search or enable_comparison):
                                    st.metric("Tiempo entrenamiento", f"{training_time:.3f}s")
                            
                            # Métricas detalladas
                            class_names_svm = [clase_labels.get(i, str(i)) for i in np.unique(y)] if clase_labels else [str(i) for i in np.unique(y)]
                            metricas = calcular_metricas_clasificacion(y, y_pred, y_prob, class_names_svm, st=st)
                            mostrar_metricas_clasificacion(st, metricas, "Métricas de SVM")
                            
                            # Matriz de confusión
                            st.subheader("🎯 5. Matriz de confusión")
                            from metricas import visualizar_matriz_confusion_mejorada
                            visualizar_matriz_confusion_mejorada(metricas['confusion_matrix'], class_names_svm, st, "Matriz de Confusión SVM")
                            
                            # Visualización de fronteras de decisión (solo para 2 features)
                            if len(selected_features) == 2:
                                st.subheader("🗺️ 6. Visualización de fronteras de decisión")
                                
                                # Crear mesh para visualización
                                h = 0.02  # step size in the mesh
                                x_min, x_max = X_model.iloc[:, 0].min() - 1, X_model.iloc[:, 0].max() + 1
                                y_min, y_max = X_model.iloc[:, 1].min() - 1, X_model.iloc[:, 1].max() + 1
                                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                                
                                # Predicción en el mesh
                                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                                Z = best_model.predict(mesh_points)
                                Z = Z.reshape(xx.shape)
                                
                                # Crear figura con plotly
                                fig_boundary = go.Figure()
                                
                                # Agregar contorno de fronteras
                                fig_boundary.add_trace(go.Contour(
                                    x=np.arange(x_min, x_max, h),
                                    y=np.arange(y_min, y_max, h),
                                    z=Z,
                                    showscale=False,
                                    colorscale='RdYlBu',
                                    opacity=0.3,
                                    hoverinfo='skip'
                                ))
                                
                                # Agregar puntos de datos
                                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                                for i, class_val in enumerate(np.unique(y)):
                                    mask = y == class_val
                                    class_name = clase_labels.get(class_val, str(class_val)) if clase_labels else str(class_val)
                                    
                                    fig_boundary.add_trace(go.Scatter(
                                        x=X_model.loc[mask, X_model.columns[0]],
                                        y=X_model.loc[mask, X_model.columns[1]],
                                        mode='markers',
                                        name=class_name,
                                        marker=dict(
                                            color=colors[i % len(colors)],
                                            size=8,
                                            line=dict(width=1, color='white')
                                        )
                                    ))
                                
                                # Agregar vectores de soporte
                                support_vectors = X_model.iloc[best_model.support_]
                                fig_boundary.add_trace(go.Scatter(
                                    x=support_vectors.iloc[:, 0],
                                    y=support_vectors.iloc[:, 1],
                                    mode='markers',
                                    name='Vectores de Soporte',
                                    marker=dict(
                                        color='black',
                                        size=12,
                                        symbol='circle-open',
                                        line=dict(width=3)
                                    )
                                ))
                                
                                fig_boundary.update_layout(
                                    title=f"Fronteras de Decisión SVM - Kernel: {model_params['kernel']}",
                                    xaxis_title=selected_features[0],
                                    yaxis_title=selected_features[1],
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_boundary, use_container_width=True)
                                
                                st.info(f"""
                                **Interpretación de la visualización:**
                                - Las **regiones coloreadas** muestran las fronteras de decisión del modelo
                                - Los **puntos** son las muestras de entrenamiento coloreadas por clase real
                                - Los **círculos negros** son los vectores de soporte (puntos críticos para la frontera)
                                - El kernel **{model_params['kernel']}** determina la forma de las fronteras
                                """)
                            
                            else:
                                st.info("🗺️ Para visualizar fronteras de decisión, selecciona exactamente 2 variables.")
                            

                            # Curvas ROC para binario y multiclase
                            st.subheader("📈 7. Curva ROC")
                            from sklearn.preprocessing import label_binarize
                            import plotly.graph_objects as go
                            import numpy as np
                            n_classes = len(np.unique(y))
                            y_unique = np.unique(y)
                            if y_prob is not None and n_classes >= 2:
                                y_bin = label_binarize(y, classes=y_unique)
                                if n_classes == 2:
                                    # Binario: usar la columna de la clase positiva
                                    from metricas import crear_curvas_roc_interactivas
                                    crear_curvas_roc_interactivas(y, y_prob, labels=y_unique, clase_labels=clase_labels)
                                else:
                                    # Multiclase: una curva por clase (one-vs-rest)
                                    from sklearn.metrics import roc_curve, auc
                                    fig_roc = go.Figure()
                                    for i, class_val in enumerate(y_unique):
                                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                                        roc_auc = auc(fpr, tpr)
                                        class_name = clase_labels.get(class_val, str(class_val)) if clase_labels else str(class_val)
                                        fig_roc.add_trace(go.Scatter(
                                            x=fpr, y=tpr, mode='lines',
                                            name=f"Clase {class_name} (AUC={roc_auc:.2f})",
                                            line=dict(width=2)
                                        ))
                                    fig_roc.add_trace(go.Scatter(
                                        x=[0, 1], y=[0, 1], mode='lines',
                                        name='Aleatorio',
                                        line=dict(dash='dash', color='gray'),
                                        showlegend=True
                                    ))
                                    fig_roc.update_layout(
                                        title="Curvas ROC multiclase (one-vs-rest)",
                                        xaxis_title="Tasa de Falsos Positivos (FPR)",
                                        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
                                        width=700, height=450
                                    )
                                    st.plotly_chart(fig_roc, use_container_width=True)
                                    st.info("Cada curva muestra la capacidad del modelo para distinguir una clase frente al resto. El AUC mide la calidad de la separación para cada clase.")
                            else:
                                st.info("No se pueden calcular curvas ROC: faltan probabilidades o clases.")
                            
                            # Análisis e interpretación automática
                            st.subheader("🧠 8. Interpretación y recomendaciones")
                            
                            # Análisis de rendimiento
                            if acc >= 0.9:
                                st.success("🎉 **Excelente rendimiento** (Accuracy ≥ 90%)")
                            elif acc >= 0.8:
                                st.info("✅ **Buen rendimiento** (Accuracy ≥ 80%)")
                            elif acc >= 0.7:
                                st.warning("⚠️ **Rendimiento moderado** (Accuracy ≥ 70%)")
                            else:
                                st.error("❌ **Rendimiento bajo** (Accuracy < 70%)")
                            
                            # Análisis de vectores de soporte
                            support_ratio = len(best_model.support_) / len(X_model)
                            if support_ratio > 0.5:
                                st.warning(f"⚠️ **Muchos vectores de soporte** ({support_ratio*100:.1f}% de los datos). El modelo puede estar sobreajustando. Considera reducir C o cambiar de kernel.")
                            elif support_ratio < 0.1:
                                st.info(f"✅ **Pocos vectores de soporte** ({support_ratio*100:.1f}% de los datos). El modelo es eficiente y probablemente generaliza bien.")
                            else:
                                st.success(f"✅ **Cantidad apropiada de vectores de soporte** ({support_ratio*100:.1f}% de los datos).")
                            
                            # Recomendaciones específicas por kernel
                            kernel_used = model_params['kernel']
                            if kernel_used == 'linear':
                                st.info("📝 **Kernel Linear**: Ideal para datos linealmente separables. Si el rendimiento es bajo, prueba kernels no lineales.")
                            elif kernel_used == 'rbf':
                                st.info("📝 **Kernel RBF**: Versátil para datos no lineales. Ajusta C y gamma para optimizar.")
                            elif kernel_used == 'poly':
                                st.info("📝 **Kernel Polinómico**: Bueno para relaciones polinómicas. Cuidado con grados altos (overfitting).")
                            elif kernel_used == 'sigmoid':
                                st.info("📝 **Kernel Sigmoid**: Similar a redes neuronales. Puede ser inestable con algunos datos.")
                            
                            # Sugerencias de mejora
                            st.markdown("### 💡 Sugerencias para mejorar el modelo:")
                            sugerencias = []
                            
                            if acc < 0.8:
                                sugerencias.append("🔧 **Ajustar hiperparámetros**: Prueba Grid Search para encontrar la mejor combinación")
                                sugerencias.append("🔄 **Cambiar kernel**: Experimenta con diferentes kernels")
                                if not escalar_datos_svm:
                                    sugerencias.append("📏 **Escalar datos**: SVM es muy sensible a la escala de las variables")
                            
                            if len(selected_features) > 10:
                                sugerencias.append("📉 **Reducir dimensionalidad**: Considera usar PCA o selección de características")
                            
                            if balance_ratio < 0.3:
                                sugerencias.append("⚖️ **Balancear clases**: Usa class_weight='balanced' o técnicas de muestreo")
                            
                            if support_ratio > 0.4:
                                sugerencias.append("🎛️ **Reducir C**: Un C más bajo puede reducir overfitting")
                            
                            if not sugerencias:
                                sugerencias.append("🎉 **¡Buen trabajo!** El modelo parece estar funcionando bien")
                            
                            for sug in sugerencias:
                                st.write(f"- {sug}")
                            
                            # Información técnica adicional
                            with st.expander("🔬 Información técnica detallada"):
                                st.write(f"**Parámetros del modelo:**")
                                for param, value in model_params.items():
                                    st.write(f"- {param}: {value}")
                                
                                st.write(f"**Estadísticas del modelo:**")
                                st.write(f"- Número de características: {X_model.shape[1]}")
                                st.write(f"- Número de muestras: {X_model.shape[0]}")
                                st.write(f"- Número de clases: {len(np.unique(y))}")
                                st.write(f"- Vectores de soporte por clase: {dict(zip(np.unique(y), np.bincount(best_model.support_[best_model.dual_coef_[0] != 0])))}")
                                
                                if hasattr(best_model, 'dual_coef_'):
                                    st.write(f"- Coeficientes duales: {best_model.dual_coef_.shape}")
                                if hasattr(best_model, 'intercept_'):
                                    st.write(f"- Intercepto: {best_model.intercept_}")
                        
                        except Exception as e:
                            st.error(f"❌ Error al entrenar el modelo: {str(e)}")
                            st.write("Posibles soluciones:")
                            st.write("- Verifica que tengas suficientes datos")
                            st.write("- Asegúrate de que las clases estén balanceadas")
                            st.write("- Prueba escalando los datos")
                            st.write("- Reduce el número de características")
            else:
                st.warning("⚠️ Selecciona al menos 2 variables para entrenar el modelo SVM.")

elif analisis == "Exploración de datos" and df is not None:
    st.title("Exploración de datos: Análisis exploratorio antes de modelar")
    st.markdown("""
    Antes de aplicar cualquier algoritmo, explora el dataset para entender su estructura, relaciones y posibles problemas. Aquí encontrarás explicaciones, advertencias y sugerencias automáticas para guiarte en la interpretación y el siguiente paso.
    """)
    # Estadísticos descriptivos

    st.header("1. Estadísticos descriptivos")
    st.info("Revisa la media, mediana, desviación estándar, valores mínimos y máximos de cada variable numérica. Esto ayuda a detectar escalas diferentes, outliers y posibles errores de carga.")
    st.markdown("""
    **¿Por qué es importante?**
    - Permite detectar variables con escalas muy diferentes (puede requerir escalado).
    - Ayuda a identificar posibles errores de carga (valores extremos, ceros, etc.).
    - Da una idea de la dispersión y simetría de los datos.
    """)

    st.dataframe(df.describe(include='all').T, use_container_width=True)
    # Descripción breve de los estadísticos
    st.markdown("""
    <div style='font-size:0.98em; color:#b0b8c9; margin-top:0.5em;'>
    <strong>¿Qué significa cada columna?</strong><br>
    <ul style='margin-bottom:0.2em;'>
    <li><b>count</b>: Cantidad de valores no nulos en la columna.</li>
    <li><b>unique</b>: (Solo categóricas) Número de valores únicos.</li>
    <li><b>top</b>: (Solo categóricas) Valor más frecuente (moda).</li>
    <li><b>freq</b>: (Solo categóricas) Frecuencia del valor más frecuente.</li>
    <li><b>mean</b>: Media aritmética (solo numéricas).</li>
    <li><b>std</b>: Desviación estándar (solo numéricas).</li>
    <li><b>min</b>: Valor mínimo.</li>
    <li><b>25%</b>: Primer cuartil (el 25% de los datos están por debajo).</li>
    <li><b>50%</b>: Mediana (valor central).</li>
    <li><b>75%</b>: Tercer cuartil (el 75% de los datos están por debajo).</li>
    <li><b>max</b>: Valor máximo.</li>
    </ul>
    <i>Estos estadísticos ayudan a detectar errores, outliers, escalas diferentes y a entender la distribución de los datos.</i>
    </div>
    """, unsafe_allow_html=True)


    st.header("2. Valores nulos y atípicos")
    st.markdown("""
    **¿Por qué es importante?**
    - Los nulos pueden impedir el entrenamiento de modelos o sesgar resultados.
    - Los outliers pueden distorsionar la media, la varianza y afectar la clasificación.
    """)
    nulos = df.isnull().sum()
    total = len(df)
    nulos_pct = (nulos / total * 100).round(2)
    st.write("#### Resumen de nulos por columna:")
    st.dataframe(pd.DataFrame({"Nulos": nulos, "%": nulos_pct}))
    # Advertencia automática
    cols_muchos_nulos = nulos_pct[nulos_pct > 20].index.tolist()
    if cols_muchos_nulos:
        st.warning(f"Las columnas {', '.join(cols_muchos_nulos)} tienen más de 20% de nulos. Considera imputar, eliminar o analizar su impacto.")
    else:
        st.success("No se detectaron columnas con más de 20% de nulos.")

    # Outliers básicos (z-score > 3)
    from scipy.stats import zscore
    num_cols_eda = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols_eda:
        zscores = df[num_cols_eda].apply(zscore).abs()
        outliers = (zscores > 3).sum()
        st.write("#### Outliers detectados (z-score > 3):")
        st.dataframe(pd.DataFrame({"Outliers": outliers, "%": (outliers/total*100).round(2)}))
        cols_muchos_out = (outliers/total*100 > 10)
        if cols_muchos_out.any():
            st.warning("Hay variables con más de 10% de outliers. Considera revisar, transformar o filtrar esos valores.")
        else:
            st.success("No se detectaron variables con más de 10% de outliers.")

    # Matriz de correlación

    st.header("3. Matriz de correlación")
    st.markdown("""
    **¿Por qué es importante?**
    - Correlaciones altas indican redundancia y pueden afectar a Bayes Ingenuo y motivar el uso de PCA.
    - LDA/QDA funcionan mejor con baja correlación entre variables.
    - **SVM** puede manejar variables correlacionadas, pero la interpretación de las fronteras de decisión es más clara si las variables no son redundantes.
    - Si quieres visualizar fronteras de decisión de SVM, elige dos variables poco correlacionadas y relevantes para la clase.
    """)
    st.markdown("""
    ---
    ### 4. Consejos prácticos para SVM en la exploración de datos
    - **Escala las variables numéricas** antes de entrenar SVM (media=0, std=1).
    - Si tienes muchas variables, usa la sugerencia automática para elegir las dos más relevantes para visualizar fronteras.
    - Si el dataset es muy grande, considera muestrear para pruebas rápidas.
    - SVM puede detectar patrones complejos, pero la visualización solo es posible con dos variables.
    - Si las clases están muy desbalanceadas, usa métricas como f1_macro o balanced accuracy.
    """)
    if len(num_cols_eda) >= 2:
        import seaborn as sns
        import matplotlib.pyplot as plt
        corr = df[num_cols_eda].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(min(0.7*len(num_cols_eda)+2, 10), min(0.7*len(num_cols_eda)+2, 10)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr, annot_kws={"size":9})
        ax_corr.set_title("Matriz de correlación (todas las variables numéricas)")
        plt.tight_layout()
        st.pyplot(fig_corr)
        # Advertencia automática
        high_corr = (corr.abs() > 0.85) & (corr.abs() < 0.9999)
        n_high_corr = high_corr.sum().sum() // 2
        if n_high_corr > 0:
            st.warning(f"Se detectaron {n_high_corr} pares de variables con correlación > 0.85. Considera eliminar variables redundantes o aplicar PCA antes de Bayes Ingenuo.")
        else:
            st.success("No se detectaron pares de variables con correlación mayor a 0.85.")
        st.caption("Correlaciones cercanas a 1 o -1 fuera de la diagonal sugieren variables redundantes. Bayes Ingenuo asume independencia, LDA/QDA prefieren baja correlación.")

    # Matriz de covarianza

    st.header("4. Matriz de covarianza")
    st.markdown("""
    **¿Por qué es importante?**
    - Covarianzas altas pueden indicar redundancia o escalas muy diferentes.
    - LDA asume covarianzas similares entre clases; PCA usa esta matriz para encontrar combinaciones óptimas.
    """)
    if len(num_cols_eda) >= 2:
        cov = df[num_cols_eda].cov()
        # Umbral para covarianza alta (puedes ajustar este valor)
        umbral_cov = 100
        # Crear máscara para valores altos fuera de la diagonal
        mask_high = (cov.abs() > umbral_cov) & (~np.eye(len(cov), dtype=bool))
        # Crear heatmap con overlay para valores altos
        fig_cov, ax_cov = plt.subplots(figsize=(min(0.7*len(num_cols_eda)+2, 10), min(0.7*len(num_cols_eda)+2, 10)))
        sns.heatmap(cov, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_cov, annot_kws={"size":9})
        # Resaltar celdas con covarianza alta
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                if mask_high.iloc[i, j]:
                    ax_cov.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
        ax_cov.set_title("Matriz de covarianza (todas las variables numéricas)")
        plt.tight_layout()
        st.pyplot(fig_cov)
        # Advertencia automática y pares problemáticos
        max_cov = cov.where(~cov.isna()).abs().values[np.triu_indices_from(cov, k=1)].max() if cov.shape[0] > 1 else 0
        pares_altos = []
        for i in range(cov.shape[0]):
            for j in range(i+1, cov.shape[1]):
                if mask_high.iloc[i, j]:
                    pares_altos.append((cov.index[i], cov.columns[j], cov.iloc[i, j]))
        if pares_altos:
            st.warning(f"Se detectan covarianzas muy altas (> {umbral_cov}) entre algunas variables. Considera escalar los datos o aplicar PCA.")
            st.markdown(f"<span style='color:#e74c3c'><b>Pares de variables con covarianza alta:</b></span>", unsafe_allow_html=True)
            for v1, v2, val in pares_altos:
                st.markdown(f"- <b>{v1}</b> y <b>{v2}</b>: cov = <b>{val:.2f}</b>", unsafe_allow_html=True)
        else:
            st.success("No se detectaron covarianzas excesivamente altas.")
        st.caption(f"Las celdas resaltadas en rojo indican pares de variables con covarianza (en valor absoluto) mayor a {umbral_cov}. Covarianzas altas pueden indicar redundancia o escalas muy diferentes. LDA asume covarianzas similares entre clases.")


    st.header("5. Distribución de clases (si aplica)")
    st.markdown("""
    **¿Por qué es importante?**
    - El balance de clases afecta la elección de la métrica y la robustez del modelo.
    - Nombres descriptivos facilitan la interpretación de resultados.
    """)
    # Usar nombres descriptivos de las clases si están definidos en session_state
    target_col = st.session_state.get("target_col_global")
    clase_labels = st.session_state.get("clase_labels_global")
    if target_col and target_col in df.columns:
        conteo = df[target_col].value_counts().sort_index()
        if clase_labels:
            nombres = [clase_labels.get(v, str(v)) for v in conteo.index]
        else:
            nombres = [str(v) for v in conteo.index]
        st.write(f"#### Distribución de la variable de clase: {target_col}")
        st.bar_chart(pd.Series(conteo.values, index=nombres))
        st.dataframe(pd.DataFrame({"Clase": nombres, "Cantidad": conteo.values}))
        # Advertencia automática solo si hay al menos dos clases
        if len(conteo) >= 2:
            min_c, max_c = conteo.min(), conteo.max()
            if max_c > 0 and min_c / max_c < 0.3:
                st.warning("Las clases están desbalanceadas. Considera usar métricas como f1_macro o balanced accuracy y técnicas de balanceo.")
            else:
                st.success("Las clases están razonablemente balanceadas.")
        st.caption("El balance de clases afecta la elección de la métrica y la robustez del modelo. Los nombres descriptivos se usarán en el resto de la app si es posible.")
    else:
        st.info("No se ha seleccionado una columna de clase o no hay nombres descriptivos definidos.")


    st.header("6. Visualizaciones básicas")
    st.markdown("""
    **¿Por qué es importante?**
    - Permite detectar asimetrías, outliers y escalas diferentes.
    - Ayuda a decidir si es necesario escalar, transformar o filtrar variables.
    """)
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols_to_plot = num_cols_eda[:6]  # Limitar para no saturar
    if cols_to_plot:
        fig, axs = plt.subplots(len(cols_to_plot), 1, figsize=(7, 2.5*len(cols_to_plot)))
        if len(cols_to_plot) == 1:
            axs = [axs]
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df[col].dropna(), kde=True, ax=axs[i], color='#4f8cff')
            axs[i].set_title(f"Distribución de {col}")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Distribuciones sesgadas o multimodales pueden requerir transformaciones o análisis especial.")

    # === Recomendación final automática ===
    st.header("7. Recomendación y próximos pasos")
    recomendaciones = []
    # Nulos
    if cols_muchos_nulos:
        recomendaciones.append("Imputa o elimina columnas con muchos nulos antes de modelar.")
    # Outliers
    if cols_muchos_out.any():
        recomendaciones.append("Considera tratar los outliers (transformar, filtrar o escalar).")
    # Correlación
    if n_high_corr > 0:
        recomendaciones.append("Aplica PCA o elimina variables redundantes si usarás Bayes Ingenuo.")
    # Covarianza
    if max_cov > 100:
        recomendaciones.append("Escala las variables si hay covarianzas muy altas.")
    # Clases desbalanceadas
    if 'min_c' in locals() and 'max_c' in locals() and max_c > 0 and min_c / max_c < 0.3:
        recomendaciones.append("Usa métricas robustas al desbalance (f1_macro, balanced accuracy) y considera técnicas de balanceo.")
    # Algoritmo sugerido
    if n_high_corr > 0:
        alg = "LDA/QDA o Bayes Ingenuo + PCA"
        just = "porque hay variables muy correlacionadas. Bayes Ingenuo puro no es recomendable salvo que uses PCA."
    elif 'min_c' in locals() and 'max_c' in locals() and max_c > 0 and min_c / max_c < 0.3:
        alg = "LDA/QDA (con métricas robustas)"
        just = "porque las clases están desbalanceadas."
    elif max_cov > 100:
        alg = "LDA/QDA (tras escalar)"
        just = "porque hay covarianzas muy altas."
    else:
        alg = "Cualquier algoritmo (LDA, QDA, Bayes Ingenuo)"
        just = "porque no se detectan problemas graves en la exploración."
    st.info(f"**Sugerencia de algoritmo inicial:** {alg}\n\n*Justificación:* {just}")
    if recomendaciones:
        st.warning("**Recomendaciones para mejorar la clasificación:**\n- " + "\n- ".join(recomendaciones))
    else:
        st.success("No se detectan problemas graves. Puedes avanzar al modelado.")
    st.markdown("""
    ---
    **¿Cómo continuar?**
    1. Aplica las recomendaciones anteriores si corresponde.
    2. Elige el algoritmo sugerido y justifica tu decisión.
    3. Si las métricas no son buenas, revisa la exploración y prueba otras técnicas (PCA, balanceo, selección de variables).
    4. Justifica cada paso con base en la evidencia explorada.
    """)

# === Comparativa de Modelos ===
elif analisis == "Comparativa de Modelos" and df is not None:
    st.title("🏆 Comparativa de Modelos: LDA, QDA, Bayes, SVM (con y sin PCA)")
    st.markdown("""
    Esta vista entrena y evalúa automáticamente los principales algoritmos de clasificación sobre tu dataset, con y sin reducción de dimensionalidad (PCA).
    Se muestran métricas comparativas, justificación automática y conclusiones para ayudarte a elegir el mejor modelo.
    """)

    # Selección de target y métricas
    target_col = st.session_state.get("target_col_global")
    if not target_col or target_col not in df.columns:
        target_col = st.selectbox("Selecciona la variable objetivo (clase)", [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])], key="target_comp")
        st.session_state["target_col_global"] = target_col
    else:
        st.info(f"Variable objetivo: {target_col}")

    metricas_disp = ["accuracy", "f1_macro", "balanced_accuracy"]
    metrica = st.selectbox("Métrica principal para comparar", metricas_disp, index=0, key="metrica_comp")
    pca_var = st.slider("% de varianza explicada por PCA", min_value=80, max_value=100, value=95, step=1, key="pca_var_comp")

    # Botón para comparar
    if st.button("🚀 Comparar modelos automáticamente", key="btn_comparar_modelos"):
        with st.spinner("Entrenando y evaluando modelos..."):
            # Preparar datos
            y = df[target_col]
            feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
            X = df[feature_cols]
            # Imputar nulos si hay
            if X.isnull().sum().sum() > 0:
                st.warning("Se imputan nulos con la media para la comparación.")
                X = X.fillna(X.mean())
            # Escalar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # PCA
            pca = PCA(n_components=min(len(feature_cols), X.shape[0]-1))
            X_pca = pca.fit_transform(X_scaled)
            var_cumsum = np.cumsum(pca.explained_variance_ratio_)*100
            n_comp = np.argmax(var_cumsum >= pca_var) + 1
            X_pca_final = X_pca[:, :n_comp]

            # Modelos a comparar
            modelos = {
                "LDA": LDA(),
                "QDA": QDA(),
                "Bayes Ingenuo": GaussianNB(),
                "SVM Linear": SVC(kernel="linear", probability=True, random_state=42),
                "SVM RBF": SVC(kernel="rbf", probability=True, random_state=42)
            }

            resultados = []
            for nombre, modelo in modelos.items():
                # Sin PCA
                try:
                    scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring=metrica)
                    resultados.append({
                        "Modelo": nombre,
                        "PCA": "No",
                        "Métrica": scores.mean(),
                        "Std": scores.std()
                    })
                except Exception as e:
                    resultados.append({
                        "Modelo": nombre,
                        "PCA": "No",
                        "Métrica": np.nan,
                        "Std": np.nan
                    })
                # Con PCA
                try:
                    scores_pca = cross_val_score(modelo, X_pca_final, y, cv=5, scoring=metrica)
                    resultados.append({
                        "Modelo": nombre,
                        "PCA": "Sí",
                        "Métrica": scores_pca.mean(),
                        "Std": scores_pca.std()
                    })
                except Exception as e:
                    resultados.append({
                        "Modelo": nombre,
                        "PCA": "Sí",
                        "Métrica": np.nan,
                        "Std": np.nan
                    })

            df_res = pd.DataFrame(resultados)
            st.subheader("📊 Resultados comparativos")
            st.dataframe(df_res.pivot(index="Modelo", columns="PCA", values="Métrica"), use_container_width=True)

            # Gráfico resumen
            fig = px.bar(df_res, x="Modelo", y="Métrica", color="PCA", barmode="group", error_y="Std",
                         title=f"Comparación de modelos ({metrica})", labels={"Métrica": metrica.capitalize()})
            st.plotly_chart(fig, use_container_width=True)

            # Justificación automática (esqueleto)
            st.subheader("🧠 Justificación y observaciones")
            mejor = df_res.sort_values("Métrica", ascending=False).iloc[0]
            st.write(f"El mejor modelo según la métrica seleccionada es **{mejor['Modelo']}** {'con PCA' if mejor['PCA']=='Sí' else 'sin PCA'}, con un valor de {mejor['Métrica']:.3f}.")
            st.write("- Si PCA mejora el rendimiento, probablemente hay redundancia o ruido en las variables.")
            st.write("- Si SVM destaca, puede indicar fronteras no lineales. Si LDA/QDA, las clases pueden ser linealmente separables.")
            st.write("- Si Bayes Ingenuo es competitivo, las variables pueden ser casi independientes.")
            st.write("- Observa la desviación estándar: valores altos indican inestabilidad o sensibilidad a la partición.")

            # Conclusión automática (esqueleto)
            st.subheader("✅ Conclusión y recomendación")
            st.write(f"Se recomienda usar **{mejor['Modelo']}** {'con PCA' if mejor['PCA']=='Sí' else 'sin PCA'} para este dataset, según la métrica seleccionada. Considera revisar los supuestos teóricos y la interpretabilidad antes de decidir el modelo final.")

            st.info("Puedes explorar los detalles de cada modelo en sus respectivas vistas del menú para ver matrices de confusión, curvas ROC y más.")


# === VISTA REGRESIÓN LOGÍSTICA ===
if analisis == "Regresión Logística" and df is not None:
    st.title("Regresión Logística (Clasificación Supervisada)")
    st.markdown("""
    La regresión logística es un modelo supervisado para clasificación binaria o multiclase. Permite predecir la probabilidad de pertenencia a cada clase y es interpretable.
    """)
    # Selección de variables
    columnas = df.columns.tolist()
    num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
    target_col = st.session_state.get("target_col_global")
    clase_labels_global = st.session_state.get("clase_labels_global", {})
    if not target_col or target_col not in df.columns:
        st.error("No hay columna de clase válida seleccionada. Elige una columna de clase en el panel lateral.")
    else:
        st.write("#### Valores únicos de la clase:")
        conteo_clase = df[target_col].value_counts().sort_index()
        st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global.get(v, str(v)) for v in conteo_clase.index], "Cantidad": conteo_clase.values}), width='stretch')
        feature_cols = st.multiselect(
            "Selecciona las columnas de atributos (features):",
            [c for c in num_cols if c != target_col],
            default=[c for c in num_cols if c != target_col],
            key="features_logreg"
        )

        if feature_cols:
            X = df[feature_cols]
            y = df[target_col]
            # Manejo de nulos
            if X.isnull().values.any():
                st.warning("Se encontraron valores faltantes en los atributos. Imputando con la media de cada columna...")
                X = X.fillna(X.mean())
            # === División train/test y validación cruzada ===
            st.markdown("---")
            st.subheader("¿Cómo evaluar el modelo de forma realista?")
            st.markdown("""
            Antes de analizar los resultados, elige **cómo quieres evaluar el modelo**:
            - **División entrenamiento/test:** Separa una parte de los datos (por ejemplo, 20%) para testear el modelo en datos no vistos.
            - **Validación cruzada (k-fold):** El dataset se divide en k partes y el modelo se entrena y evalúa k veces, usando cada parte como test una vez.
            """)
            eval_mode = st.radio(
                "Selecciona el método de evaluación:",
                ["División entrenamiento/test", "Validación cruzada (k-fold)"],
                index=0,
                help="Elige si prefieres reservar un conjunto de test fijo o evaluar con k particiones distintas."
            )
            if eval_mode == "División entrenamiento/test":
                test_size = st.slider(
                    "¿Qué porcentaje de los datos reservar para test?",
                    min_value=10, max_value=50, value=20, step=5,
                    help="El resto de los datos se usa para entrenar el modelo."
                )
                st.caption("El modelo se entrena con el (100 - test)% de los datos y se evalúa con el test% restante. Así puedes ver si el modelo generaliza bien a datos nuevos.")
                use_cv = False
                k_folds = 5
            else:
                use_cv = True
                k_folds = st.slider(
                    "¿Cuántos folds (k) usar en la validación cruzada?",
                    min_value=3, max_value=10, value=5,
                    help="El dataset se divide en k partes. El modelo se entrena y evalúa k veces, cada vez usando una parte distinta como test."
                )
                st.caption("La validación cruzada es más robusta: todos los datos se usan para entrenar y testear, y se obtiene un promedio de métricas.")
            # === Opciones de regularización ===
            st.markdown("---")
            st.subheader("Regularización: controlar la complejidad del modelo")
            st.markdown("""
            La regularización penaliza coeficientes muy grandes para evitar sobreajuste. Hay tres opciones:
            - **Sin regularización (None):** Los coeficientes pueden tomar cualquier valor. Puede causar sobreajuste si hay muchas variables o ruido.
            - **L1 (Lasso):** Penaliza la suma de valores absolutos de los coeficientes. Tiende a eliminar variables irrelevantes (coeficientes = 0). Útil para selección de variables.
            - **L2 (Ridge):** Penaliza la suma de cuadrados de los coeficientes. Mantiene todas las variables, pero reduce su influencia. Más estable numéricamente.
            
            **¿Cuándo usar cada tipo?**
            - **L1:** Cuando sospechás que muchas variables son irrelevantes y querés un modelo interpretable y simple.
            - **L2:** Cuando todas las variables pueden ser relevantes, pero querés evitar que tengan demasiada influencia individual.
            - **Sin regularización:** Solo si tenés pocos datos o pocas variables, y confiás en que no hay sobreajuste.
            """)
            
            penalty_option = st.selectbox(
                "Selecciona el tipo de regularización:",
                ["L2 (Ridge)", "L1 (Lasso)", "Sin regularización (None)"],
                index=0,
                help="L2 es la opción estándar y más estable. L1 es útil para selección de variables."
            )
            
            if penalty_option == "L2 (Ridge)":
                penalty = 'l2'
                solver = 'lbfgs'
            elif penalty_option == "L1 (Lasso)":
                penalty = 'l1'
                solver = 'saga'  # saga soporta l1
            else:  # Sin regularización
                penalty = None
                solver = 'lbfgs'
            
            # Control de C (inverso de lambda de regularización)
            if penalty != 'none':
                C_value = st.slider(
                    "Fuerza de regularización (C):",
                    min_value=0.001, max_value=100.0, value=1.0, step=0.1,
                    help="C es el inverso de la fuerza de regularización. Valores pequeños = más regularización (más penalización). Valores grandes = menos regularización (más libertad)."
                )
                st.caption(f"C={C_value:.3f} → {'Regularización fuerte' if C_value < 0.1 else 'Regularización moderada' if C_value < 10 else 'Regularización débil'}")
            else:
                C_value = 1.0  # No importa si penalty=None
            
            # Escalado
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            import numpy as np
            if use_cv:
                st.info(f"Validación cruzada con {k_folds} folds: el modelo se entrena y evalúa {k_folds} veces con diferentes particiones.")
                from sklearn.model_selection import cross_val_score
                model = LogisticRegression(max_iter=500, multi_class='auto', solver=solver, penalty=penalty, C=C_value)
                # Si penalty es None, no lo pasamos como argumento (usa default)
                if penalty is None:
                    model = LogisticRegression(max_iter=500, multi_class='auto', solver=solver, C=C_value)
                else:
                    model = LogisticRegression(max_iter=500, multi_class='auto', solver=solver, penalty=penalty, C=C_value)
                if penalty is None:
                    model = LogisticRegression(max_iter=500, multi_class='auto', solver=solver, C=C_value)
                else:
                    model = LogisticRegression(max_iter=500, multi_class='auto', solver=solver, penalty=penalty, C=C_value)
                accs = cross_val_score(model, X_scaled, y, cv=k_folds, scoring='accuracy')
                precs = cross_val_score(model, X_scaled, y, cv=k_folds, scoring='precision_weighted')
                recs = cross_val_score(model, X_scaled, y, cv=k_folds, scoring='recall_weighted')
                f1s = cross_val_score(model, X_scaled, y, cv=k_folds, scoring='f1_weighted')
                st.write("### Métricas promedio (validación cruzada)")
                st.dataframe({
                    'Accuracy': [f"{accs.mean():.3f} ± {accs.std():.3f}"],
                    'Precision': [f"{precs.mean():.3f} ± {precs.std():.3f}"],
                    'Recall': [f"{recs.mean():.3f} ± {recs.std():.3f}"],
                    'F1': [f"{f1s.mean():.3f} ± {f1s.std():.3f}"]
                })
                st.caption("Las métricas muestran el promedio y la desviación estándar entre los folds. Valores similares indican estabilidad. La validación cruzada es útil cuando el dataset es pequeño o se quiere una evaluación más robusta.")
                # Entrenar modelo final en todo el dataset para predicción interactiva y coeficientes
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
            else:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42, stratify=y)
                model = LogisticRegression(max_iter=500, multi_class='auto', solver=solver, penalty=penalty, C=C_value)
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                # Métricas
                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)
                prec_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
                prec_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
                rec_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
                rec_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                f1_train = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
                f1_test = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                st.write("### Métricas de entrenamiento y test")
                st.dataframe({
                    'Conjunto': ['Entrenamiento', 'Test'],
                    'Accuracy': [f"{acc_train:.3f}", f"{acc_test:.3f}"],
                    'Precision': [f"{prec_train:.3f}", f"{prec_test:.3f}"],
                    'Recall': [f"{rec_train:.3f}", f"{rec_test:.3f}"],
                    'F1': [f"{f1_train:.3f}", f"{f1_test:.3f}"]
                })
                st.caption("Compara las métricas en entrenamiento y test. Si hay mucha diferencia, puede haber sobreajuste. La división test es útil para datasets grandes o para simular el rendimiento en datos realmente nuevos.")
                # Matriz de confusión de test
                st.write("#### Matriz de confusión (test):")
                cm = confusion_matrix(y_test, y_pred_test)
                st.dataframe(pd.DataFrame(cm, index=[clase_labels_global.get(c, str(c)) for c in model.classes_], columns=[clase_labels_global.get(c, str(c)) for c in model.classes_]))
                
                # === Comparación de regularizaciones ===
                with st.expander("🔍 Comparar el efecto de diferentes regularizaciones en el rendimiento"):
                    st.markdown("""
                    Veamos cómo cambian las métricas con diferentes tipos de regularización:
                    """)
                    
                    comparison_results = []
                    reg_configs = [
                        ("Sin regularización", None, "lbfgs", 1.0),
                        ("L2 (Ridge)", "l2", "lbfgs", C_value),
                        ("L1 (Lasso)", "l1", "saga", C_value)
                    ]
                    
                    for reg_name, reg_penalty, reg_solver, reg_C in reg_configs:
                        if reg_penalty is None:
                            model_comp = LogisticRegression(
                                max_iter=500, 
                                multi_class='auto', 
                                solver=reg_solver, 
                                C=reg_C,
                                random_state=42
                            )
                        else:
                            model_comp = LogisticRegression(
                                max_iter=500, 
                                multi_class='auto', 
                                solver=reg_solver, 
                                penalty=reg_penalty, 
                                C=reg_C,
                                random_state=42
                            )
                        model_comp.fit(X_train, y_train)
                        y_pred_comp_train = model_comp.predict(X_train)
                        y_pred_comp_test = model_comp.predict(X_test)
                        
                        acc_comp_train = accuracy_score(y_train, y_pred_comp_train)
                        acc_comp_test = accuracy_score(y_test, y_pred_comp_test)
                        f1_comp_train = f1_score(y_train, y_pred_comp_train, average='weighted', zero_division=0)
                        f1_comp_test = f1_score(y_test, y_pred_comp_test, average='weighted', zero_division=0)
                        
                        # Calcular número de variables activas (coeficientes no nulos)
                        if model_comp.coef_.shape[0] == 1:
                            n_active = np.sum(np.abs(model_comp.coef_[0]) > 1e-5)
                        else:
                            n_active = np.sum(np.any(np.abs(model_comp.coef_) > 1e-5, axis=0))
                        
                        comparison_results.append({
                            'Regularización': reg_name,
                            'Acc Train': f"{acc_comp_train:.3f}",
                            'Acc Test': f"{acc_comp_test:.3f}",
                            'F1 Train': f"{f1_comp_train:.3f}",
                            'F1 Test': f"{f1_comp_test:.3f}",
                            'Variables activas': f"{n_active}/{len(feature_cols)}",
                            'Sobreajuste': f"{abs(acc_comp_train - acc_comp_test):.3f}"
                        })
                    
                    df_comparison = pd.DataFrame(comparison_results)
                    st.dataframe(df_comparison, hide_index=True, use_container_width=True)
                    
                    st.markdown("""
                    <div style='background:#23293a;padding:1em;border-radius:8px;'>
                    <b>¿Cómo interpretar esta tabla?</b><br>
                    - <b>Acc/F1 Train vs Test:</b> Si train es mucho mayor que test, hay sobreajuste.<br>
                    - <b>Variables activas:</b> L1 puede eliminar variables. L2 y sin regularización usan todas.<br>
                    - <b>Sobreajuste:</b> Diferencia entre accuracy de train y test. Menor = mejor generalización.<br>
                    - <b>Recomendación:</b> Busca el equilibrio entre buen rendimiento en test y baja diferencia train-test.<br>
                    </div>
                    """, unsafe_allow_html=True)
            # Si no hay validación cruzada, mostrar matriz de confusión de entrenamiento también
            if not use_cv:
                st.write("#### Matriz de confusión (entrenamiento):")
                cm_train = confusion_matrix(y_train, y_pred_train)
                st.dataframe(pd.DataFrame(cm_train, index=[clase_labels_global.get(c, str(c)) for c in model.classes_], columns=[clase_labels_global.get(c, str(c)) for c in model.classes_]))

            # === Curva ROC y AUC ===
            st.markdown("---")
            st.subheader("Curva ROC y AUC: capacidad discriminativa del modelo")
            st.markdown("""
            La curva ROC (Receiver Operating Characteristic) muestra la relación entre la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR) para distintos umbrales de decisión. El área bajo la curva (AUC) mide la capacidad del modelo para distinguir entre clases:
            - **AUC = 1.0:** Separación perfecta.
            - **AUC = 0.5:** Sin capacidad discriminativa (aleatorio).
            - **Cuanto mayor el AUC, mejor el modelo para distinguir clases.**
            """)
            import plotly.graph_objects as go
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            y_labels = model.classes_
            n_classes = len(y_labels)
            # Usar probabilidades y verdaderos según modo de evaluación
            if use_cv:
                # En CV, usar todo el dataset (no hay test explícito)
                y_true = y
                y_score = model.predict_proba(X_scaled)
            else:
                y_true = y_test
                y_score = model.predict_proba(X_test)
            if n_classes == 2:
                # Binario: usar la probabilidad de la clase positiva
                fpr, tpr, _ = roc_curve(y_true, y_score[:, 1], pos_label=y_labels[1])
                auc_val = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_val:.3f})', line=dict(width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio', line=dict(dash='dash', color='gray')))
                fig_roc.update_layout(title="Curva ROC (binaria)", xaxis_title="Tasa de Falsos Positivos (FPR)", yaxis_title="Tasa de Verdaderos Positivos (TPR)", width=600, height=400)
                st.plotly_chart(fig_roc, use_container_width=True)
                st.info(f"AUC = {auc_val:.3f}. Un valor cercano a 1 indica excelente capacidad de discriminación.")
                st.markdown("""
                <div style='background:#23293a;padding:1em;border-radius:8px;'>
                <b>¿Qué ves aquí?</b><br>
                - <b>Curva ROC binaria:</b> Muestra la capacidad del modelo para distinguir entre las dos clases posibles.<br>
                - <b>AUC:</b> Área bajo la curva. Si es cercano a 1, el modelo distingue muy bien; si es 0.5, es como adivinar.<br>
                - <b>Interpretación:</b> Cuanto más arriba y a la izquierda esté la curva, mejor.<br>
                </div>
                """, unsafe_allow_html=True)
            elif n_classes > 2:
                # Multiclase: one-vs-rest
                y_bin = label_binarize(y_true, classes=y_labels)
                fig_roc = go.Figure()
                aucs = []
                for i, c in enumerate(y_labels):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                    auc_val = auc(fpr, tpr)
                    aucs.append(auc_val)
                    class_name = clase_labels_global.get(c, str(c))
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{class_name} (AUC={auc_val:.2f})', line=dict(width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio', line=dict(dash='dash', color='gray')))
                fig_roc.update_layout(title="Curvas ROC (one-vs-rest, multiclase)", xaxis_title="Tasa de Falsos Positivos (FPR)", yaxis_title="Tasa de Verdaderos Positivos (TPR)", width=700, height=450)
                st.plotly_chart(fig_roc, use_container_width=True)
                st.info(f"AUC promedio: {np.mean(aucs):.3f}. Cada curva muestra la capacidad del modelo para distinguir una clase frente al resto.")
                st.markdown(f"""
                <div style='background:#23293a;padding:1em;border-radius:8px;'>
                <b>¿Qué ves aquí?</b><br>
                - <b>Curvas ROC multiclase (one-vs-rest):</b> Para cada clase, se calcula una curva considerando esa clase como positiva y las demás como negativas.<br>
                - <b>AUC de cada clase:</b> Indica qué tan bien el modelo distingue esa clase frente a las otras.<br>
                - <b>AUC promedio:</b> {np.mean(aucs):.3f}.<br>
                - <b>Interpretación:</b> Curvas más arriba y a la izquierda indican mejor discriminación. Si varias curvas están cerca de la diagonal, el modelo no distingue bien esas clases.<br>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No se puede calcular la curva ROC: se requiere al menos dos clases.")
            
            # === Visualización de frontera de decisión (solo si hay 2 variables) ===
            if len(feature_cols) == 2:
                st.markdown("---")
                st.subheader("🗺️ Visualización de la frontera de decisión")
                st.markdown("""
                Cuando usas solo 2 variables, puedes ver gráficamente cómo el modelo separa las clases en el espacio de atributos. 
                Los puntos son las observaciones reales y las regiones de color muestran la predicción del modelo en cada zona del espacio.
                """)
                # Crear mesh para visualización
                h = 0.02
                x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
                y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                # Convertir Z a numérico si es categórico (asegurar 1D)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(y)
                Z_num = le.transform(Z.ravel()).reshape(xx.shape)
                # Graficar
                fig_boundary = go.Figure()
                # Contorno de fondo
                fig_boundary.add_trace(go.Contour(
                    x=np.arange(x_min, x_max, h),
                    y=np.arange(y_min, y_max, h),
                    z=Z_num,
                    showscale=False,
                    colorscale='RdYlBu',
                    opacity=0.3,
                    hoverinfo='skip'
                ))
                # Agregar puntos de datos
                if use_cv:
                    X_plot = X_scaled
                    y_plot = y
                else:
                    X_plot = X_test
                    y_plot = y_test
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                for i, class_val in enumerate(np.unique(y_plot)):
                    mask = y_plot == class_val
                    class_name = clase_labels_global.get(class_val, str(class_val))
                    fig_boundary.add_trace(go.Scatter(
                        x=X_plot[mask, 0],
                        y=X_plot[mask, 1],
                        mode='markers',
                        name=class_name,
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=8,
                            line=dict(width=1, color='white')
                        )
                    ))
                fig_boundary.update_layout(
                    title=f"Frontera de decisión: {feature_cols[0]} vs {feature_cols[1]}",
                    xaxis_title=f"{feature_cols[0]} (escalado)",
                    yaxis_title=f"{feature_cols[1]} (escalado)",
                    showlegend=True
                )
                st.plotly_chart(fig_boundary, use_container_width=True)
                st.markdown("""
                <div style='background:#23293a;padding:1em;border-radius:8px;'>
                <b>¿Qué ves aquí?</b><br>
                - <b>Regiones de color:</b> Muestran la predicción del modelo en cada zona del espacio.<br>
                - <b>Puntos:</b> Son las observaciones reales del conjunto de datos (coloreadas por su clase verdadera).<br>
                - <b>Interpretación:</b> Si los puntos de cada clase están bien agrupados en su región correspondiente, el modelo separa bien. Si hay muchos puntos en la región equivocada, hay errores de clasificación.<br>
                </div>
                """, unsafe_allow_html=True)
            elif len(feature_cols) > 2:
                st.info("💡 Para visualizar la frontera de decisión, selecciona exactamente 2 variables en la sección de atributos.")
            
            # === Interpretación de coeficientes ===
            st.markdown("---")
            st.subheader("Interpretación de coeficientes")
            st.markdown("""
            En regresión logística, cada coeficiente indica cuánto influye la variable correspondiente en la probabilidad de pertenecer a la clase positiva (o a cada clase, si es multiclase):
            - **Signo positivo:** Aumentar la variable aumenta la probabilidad de la clase positiva.
            - **Signo negativo:** Aumentar la variable disminuye la probabilidad de la clase positiva.
            - **Magnitud:** Cuanto mayor el valor absoluto, mayor el impacto.
            """)
            import numpy as np
            coef = model.coef_
            if coef.shape[0] == 1:
                # Binaria
                df_coef = pd.DataFrame({
                    'Variable': feature_cols,
                    'Coeficiente': coef[0]
                })
                df_coef['Impacto'] = np.where(df_coef['Coeficiente'] > 0, '↑ Aumenta prob. clase 1', '↓ Disminuye prob. clase 1')
                st.dataframe(df_coef.set_index('Variable').style.format({'Coeficiente': '{:.3f}'}))
            else:
                # Multiclase
                df_coef = pd.DataFrame(coef.T, columns=[f"Clase {clase_labels_global.get(c, str(c))}" for c in model.classes_], index=feature_cols)
                st.dataframe(df_coef.style.format('{:.3f}'))
                st.caption("Cada columna muestra el efecto de la variable sobre la probabilidad de esa clase (vs. la clase base).")
            st.info("Un coeficiente positivo significa que al aumentar esa variable, aumenta la probabilidad de la clase indicada. Un coeficiente negativo, lo contrario. La magnitud indica la fuerza del efecto.")
            st.caption("Puedes ordenar la tabla para ver qué variables tienen mayor impacto.")

            # === Ranking de importancia de variables ===
            st.markdown("---")
            st.subheader("Ranking de importancia de variables")
            st.markdown("""
            El ranking de importancia te permite identificar rápidamente qué variables tienen mayor influencia en la predicción del modelo. Se calcula usando el valor absoluto de los coeficientes:
            - **Mayor valor absoluto = mayor importancia**
            - Con regularización L1, algunas variables pueden tener importancia cero (eliminadas)
            - Con L2, todas las variables suelen tener algún peso, pero los menos importantes se reducen
            """)
            import plotly.express as px
            if coef.shape[0] == 1:
                # Binaria
                importancias = np.abs(coef[0])
            else:
                # Multiclase: promedio del valor absoluto entre clases
                importancias = np.mean(np.abs(coef), axis=0)
            df_importancia = pd.DataFrame({
                'Variable': feature_cols,
                'Importancia (|coef|)': importancias
            }).sort_values('Importancia (|coef|)', ascending=False)
            fig_imp = px.bar(
                df_importancia,
                x='Importancia (|coef|)',
                y='Variable',
                orientation='h',
                color='Importancia (|coef|)',
                color_continuous_scale='Blues',
                title='Importancia relativa de cada variable',
                labels={'Importancia (|coef|)': 'Importancia (valor absoluto del coeficiente)'}
            )
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info("Las variables más arriba en el gráfico son las que más influyen en la predicción. Si usas L1, las variables con barra cero han sido eliminadas por el modelo.")
            st.caption("La regularización puede cambiar el ranking: L1 tiende a dejar solo las variables más relevantes, L2 distribuye el peso entre todas.")
            
            # === Efecto de la regularización en los coeficientes ===
            st.markdown("---")
            st.subheader("Efecto de la regularización en los coeficientes")
            st.markdown("""
            La regularización modifica los coeficientes del modelo. Veamos cómo varían al cambiar la fuerza de regularización:
            """)
            
            # Visualización del efecto de la regularización
            if penalty != 'none':
                st.markdown(f"""
                **Tipo de regularización actual: {penalty_option}**
                
                - **L1 (Lasso):** Fuerza algunos coeficientes a ser exactamente 0, eliminando variables.
                - **L2 (Ridge):** Reduce todos los coeficientes, pero ninguno llega a 0.
                """)
                
                # Calcular coeficientes para diferentes valores de C
                C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                coef_paths = {feat: [] for feat in feature_cols}
                
                for C_test in C_values:
                    model_test = LogisticRegression(
                        max_iter=500, 
                        multi_class='auto', 
                        solver=solver, 
                        penalty=penalty, 
                        C=C_test,
                        random_state=42
                    )
                    if use_cv:
                        model_test.fit(X_scaled, y)
                    else:
                        model_test.fit(X_train, y_train)
                    
                    # Guardar coeficientes (promedio si es multiclase)
                    if model_test.coef_.shape[0] == 1:
                        coefs = model_test.coef_[0]
                    else:
                        # Multiclase: promediar coeficientes entre clases
                        coefs = np.mean(np.abs(model_test.coef_), axis=0)
                    
                    for i, feat in enumerate(feature_cols):
                        coef_paths[feat].append(coefs[i])
                
                # Crear gráfico interactivo de evolución de coeficientes
                fig_coef_reg = go.Figure()
                for feat in feature_cols:
                    fig_coef_reg.add_trace(go.Scatter(
                        x=C_values,
                        y=coef_paths[feat],
                        mode='lines+markers',
                        name=feat,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                
                fig_coef_reg.update_xaxes(type='log', title='C (Fuerza de regularización)')
                fig_coef_reg.update_yaxes(title='Valor del coeficiente')
                fig_coef_reg.update_layout(
                    title=f'Evolución de coeficientes con {penalty_option}',
                    hovermode='x unified',
                    showlegend=True
                )
                
                # Agregar línea vertical para el C actual
                fig_coef_reg.add_vline(
                    x=C_value, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"C actual = {C_value}",
                    annotation_position="top"
                )
                
                st.plotly_chart(fig_coef_reg, use_container_width=True)
                
                st.markdown(f"""
                <div style='background:#23293a;padding:1em;border-radius:8px;'>
                <b>¿Qué muestra este gráfico?</b><br>
                - <b>Eje X (C):</b> Fuerza de regularización (escala logarítmica). Valores pequeños = más penalización.<br>
                - <b>Eje Y:</b> Valor del coeficiente de cada variable.<br>
                - <b>Línea roja:</b> Tu valor actual de C ({C_value:.3f}).<br>
                - <b>Interpretación {penalty_option}:</b><br>
                  {'• Coeficientes tienden a 0 a medida que C disminuye (más regularización).<br>• Algunas variables pueden eliminarse completamente (coef = 0) con L1.' if penalty == 'l1' else '• Todos los coeficientes se reducen uniformemente a medida que C disminuye.<br>• Con L2, ningún coeficiente llega exactamente a 0, pero se vuelven muy pequeños.'}<br>
                </div>
                """, unsafe_allow_html=True)
                
                # Comparación de número de variables activas (para L1)
                if penalty == 'l1':
                    st.write("#### Variables activas (coeficiente ≠ 0) según C:")
                    active_vars = []
                    for C_test in C_values:
                        model_test = LogisticRegression(
                            max_iter=500, 
                            multi_class='auto', 
                            solver=solver, 
                            penalty=penalty, 
                            C=C_test,
                            random_state=42
                        )
                        if use_cv:
                            model_test.fit(X_scaled, y)
                        else:
                            model_test.fit(X_train, y_train)
                        
                        # Contar coeficientes no nulos
                        if model_test.coef_.shape[0] == 1:
                            n_active = np.sum(np.abs(model_test.coef_[0]) > 1e-5)
                        else:
                            # Multiclase: una variable está activa si algún coeficiente es no nulo
                            n_active = np.sum(np.any(np.abs(model_test.coef_) > 1e-5, axis=0))
                        active_vars.append(n_active)
                    
                    df_active = pd.DataFrame({
                        'C': C_values,
                        'Variables activas': active_vars,
                        'Total variables': len(feature_cols)
                    })
                    st.dataframe(df_active, hide_index=True)
                    st.caption("Con L1 (Lasso), al reducir C (más regularización), más variables son eliminadas (coeficiente = 0). Esto es útil para selección automática de variables.")
            else:
                st.info("Sin regularización, los coeficientes no están penalizados. Esto puede funcionar bien con datasets pequeños, pero puede causar sobreajuste si hay muchas variables o ruido.")
            
            st.caption("Esta es una vista inicial. Puedes complejizarla luego agregando validación cruzada, regularización, interpretación de coeficientes, etc.")

            # === Predicción interactiva ===
            st.markdown("---")
            st.subheader("Predicción interactiva y probabilidades por clase")
            st.markdown("""
            Ingresa valores para cada variable y obtén la predicción del modelo y las probabilidades asociadas a cada clase.
            """)
            col_inputs = st.columns(len(feature_cols))
            input_vals = []
            for i, col in enumerate(feature_cols):
                minv = float(df[col].min())
                maxv = float(df[col].max())
                meanv = float(df[col].mean())
                val = col_inputs[i].number_input(f"{col}", min_value=minv, max_value=maxv, value=meanv, key=f"input_{col}_logreg")
                input_vals.append(val)
            if st.button("Predecir", key="btn_predecir_logreg"):
                # Escalar igual que entrenamiento
                input_array = np.array(input_vals).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                pred = model.predict(input_scaled)[0]
                probas = model.predict_proba(input_scaled)[0]
                st.success(f"Predicción: {clase_labels_global.get(pred, str(pred))}")
                st.write("#### Probabilidades por clase:")
                class_names = [clase_labels_global.get(c, str(c)) for c in model.classes_]
                df_proba = pd.DataFrame({
                    'Clase': class_names,
                    'Probabilidad': [f"{p:.3f}" for p in probas]
                })
                st.dataframe(df_proba)
                import plotly.graph_objects as go
                fig_proba = go.Figure(go.Bar(
                    x=class_names,
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
                st.caption("La clase con mayor probabilidad es la predicción del modelo. Si varias clases tienen probabilidades similares, el modelo está menos seguro.")
        else:
            st.info("Selecciona al menos una variable numérica para entrenar el modelo.")

# === VISTA K-MEANS CLUSTERING ===
elif analisis == "K-means (Clustering)" and df is not None:
    st.title("K-means: Agrupamiento no supervisado")
    st.markdown("""
    K-means es un algoritmo de aprendizaje no supervisado que agrupa los datos en k clusters según su similitud. No requiere etiquetas de clase.
    """)
    columnas = df.columns.tolist()
    num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = st.multiselect(
        "Selecciona las columnas numéricas para clustering:",
        num_cols,
        default=num_cols,
        key="features_kmeans"
    )
    if feature_cols and len(feature_cols) >= 2:
        X = df[feature_cols]
        # Manejo de nulos
        if X.isnull().values.any():
            st.warning("Se encontraron valores faltantes. Imputando con la media...")
            X = X.fillna(X.mean())
        # Escalado
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Selección de k
        k = st.slider("Selecciona el número de clusters (k)", min_value=2, max_value=10, value=3, step=1)
        # Entrenamiento
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        st.write(f"#### Asignación de clusters (primeras 10 filas):")
        st.dataframe(pd.DataFrame({"Cluster": clusters[:10]}, index=df.index[:10]))
        # Visualización 2D si hay al menos 2 features
        import plotly.express as px
        if len(feature_cols) >= 2:
            fig = px.scatter(
                x=X[feature_cols[0]], y=X[feature_cols[1]], color=clusters.astype(str),
                labels={"x": feature_cols[0], "y": feature_cols[1], "color": "Cluster"},
                title="Visualización de clusters (2 primeras variables)"
            )
            st.plotly_chart(fig, use_container_width=True)
        # Método del codo
        st.markdown("---")
        st.subheader("Método del codo para elegir k óptimo")
        inertia = []
        ks = range(1, 11)
        for ki in ks:
            km = KMeans(n_clusters=ki, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertia.append(km.inertia_)
        import matplotlib.pyplot as plt
        fig2, ax2 = plt.subplots()
        ax2.plot(ks, inertia, '-o')
        ax2.set_xlabel('k (número de clusters)')
        ax2.set_ylabel('Inercia (WCSS)')
        ax2.set_title('Método del codo')
        st.pyplot(fig2)
        st.caption("Elige el k donde la curva deja de bajar abruptamente (el 'codo').")
        st.info("Esta es una vista inicial. Puedes agregar silhouette, visualización 3D, interpretación, etc. más adelante.")
    else:
        st.info("Selecciona al menos dos variables numéricas para aplicar k-means.")

elif df is not None:

    # Usar nombres descriptivos de las clases definidos en el sidebar
    target_col = st.session_state.get("target_col_global")
    clase_labels_global = st.session_state.get("clase_labels_global", {})
    if target_col and target_col in df.columns:
        conteo_clase = df[target_col].value_counts().sort_index()
        st.write("#### Valores únicos de la clase:")
        st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global.get(v, str(v)) for v in conteo_clase.index], "Cantidad": conteo_clase.values}), width='stretch')
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
        # Usar nombres descriptivos y columna de clase definidos globalmente
        columnas = df.columns.tolist()
        num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
        target_col = st.session_state.get("target_col_global")
        clase_labels_global = st.session_state.get("clase_labels_global", {})
        if not target_col or target_col not in df.columns:
            st.error("No hay columna de clase válida seleccionada. Elige una columna de clase en el panel lateral.")
        else:
            st.write("#### Valores únicos de la clase:")
            conteo_clase = df[target_col].value_counts().sort_index()
            st.dataframe(pd.DataFrame({"Valor de clase": [clase_labels_global.get(v, str(v)) for v in conteo_clase.index], "Cantidad": conteo_clase.values}), width='stretch')
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

                    # Explicación didáctica sobre confianza y umbral de decisión
                    st.info("""
**¿Qué significa la confianza de la predicción?**  
La confianza es la probabilidad que el modelo asigna a la clase predicha. Si la probabilidad es alta (por ejemplo, 0.95), el modelo está muy seguro. Si es intermedia (por ejemplo, 0.75-0.85), la confianza es moderada. Si varias clases tienen probabilidades similares, la predicción es incierta.

**¿Para qué sirve el umbral de decisión?**  
El umbral es el valor mínimo de probabilidad que debe superar una clase para que el modelo la asigne como predicción. Si ninguna clase supera el umbral, la predicción se considera incierta.  
Esto te permite ser más exigente: solo aceptar predicciones cuando el modelo está realmente seguro.

**Ejemplo práctico:**
- Si el umbral es 0.80 y la clase 2 tiene probabilidad 0.87, la predicción es la clase 2 con confianza moderada.
- Si ninguna clase supera 0.80, el modelo no hace una predicción segura y te avisa que la observación es ambigua.

**¿Cuándo ajustar el umbral?**
- Si prefieres evitar errores y solo aceptar predicciones muy seguras, sube el umbral.
- Si prefieres que el modelo siempre dé una respuesta, usa el umbral por defecto (0.5 o menor).
""")
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
                    mostrar_metricas_clasificacion(st, metricas, "Métricas de Bayes Ingenuo")
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
                            # Explicación didáctica sobre curva ROC y AUC
                            st.info("""
    **¿Qué es la curva ROC?**

    - Es un gráfico que muestra la capacidad del modelo para distinguir entre clases.
    - El eje X es la tasa de falsos positivos (FPR) y el eje Y la tasa de verdaderos positivos (TPR).
    - Cada punto representa un posible umbral de decisión para clasificar.

    **¿Para qué sirve?**
    - Permite ver cómo cambia el rendimiento del modelo al variar el umbral de probabilidad.
    - Si la curva está cerca del vértice superior izquierdo, el modelo es muy bueno.
    - Si la curva está cerca de la diagonal, el modelo no distingue bien (es casi aleatorio).

    **¿Qué es el AUC?**
    - Es el área bajo la curva ROC (AUC = Area Under Curve).
    - Va de 0 a 1. Un AUC de 1 es perfecto, 0.5 es como tirar una moneda.
    - Cuanto más alto el AUC, mejor el modelo para separar las clases.

    **¿Cómo lo uso?**
    - Compara modelos: el que tenga mayor AUC es mejor distinguiendo entre clases.
    - Si ves AUC cercanos a 1, tu modelo es muy bueno. Si ves valores bajos, hay que mejorar el modelo o los datos.

    **Ejemplo visual:**
    - Una curva que sube rápido hacia arriba y luego a la derecha indica un modelo excelente.
    - Una curva cerca de la diagonal (línea punteada) indica un modelo poco útil.
    """)
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
        # Usar nombres descriptivos y columna de clase definidos globalmente
        columnas = df.columns.tolist()
        num_cols = [c for c in columnas if pd.api.types.is_numeric_dtype(df[c])]
        target_col = st.session_state.get("target_col_global")
        clase_labels_global = st.session_state.get("clase_labels_global", {})
        if not target_col or target_col not in df.columns:
            st.error("No hay columna de clase válida seleccionada. Elige una columna de clase en el panel lateral.")
        else:
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
            mostrar_ejemplos_por_clase(df, target_col, clase_labels_global, st, n=3)

            # === MATRIZ DE CORRELACIÓN Y COVARIANZA DE FEATURES ===
            if feature_cols and len(feature_cols) >= 2:
                import matplotlib.pyplot as plt
                import seaborn as sns
                st.markdown("---")
                st.write("### 🔗 Matriz de correlación entre variables seleccionadas")
                st.caption("LDA y QDA asumen ciertas propiedades sobre la relación entre variables. Si ves correlaciones fuertes (valores cercanos a 1 o -1 fuera de la diagonal), puede ser útil aplicar PCA o eliminar variables redundantes para mejorar la discriminación.")
                corr_lda = df[feature_cols].corr()
                fig_corr, ax_corr = plt.subplots(figsize=(min(0.7*len(feature_cols)+2, 10), min(0.7*len(feature_cols)+2, 10)))
                sns.heatmap(corr_lda, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr, annot_kws={"size":9})
                ax_corr.set_title("Matriz de correlación (features seleccionadas)")
                plt.tight_layout()
                st.pyplot(fig_corr)
                st.info("""
**¿Qué muestra esta matriz?**
La matriz de correlación indica el grado de relación lineal entre cada par de variables. Valores cercanos a 1 o -1 indican alta correlación positiva o negativa, respectivamente. LDA y QDA funcionan mejor si las variables no están fuertemente correlacionadas.
""")
                st.markdown("---")
                st.write("### 📏 Matriz de covarianza entre variables seleccionadas")
                st.caption("La matriz de covarianza muestra cómo varían conjuntamente las variables y la escala de sus varianzas. Es útil para entender la dispersión y redundancia de la información antes de aplicar PCA.")
                cov_lda = df[feature_cols].cov()
                fig_cov, ax_cov = plt.subplots(figsize=(min(0.7*len(feature_cols)+2, 10), min(0.7*len(feature_cols)+2, 10)))
                sns.heatmap(cov_lda, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_cov, annot_kws={"size":9})
                ax_cov.set_title("Matriz de covarianza (features seleccionadas)")
                plt.tight_layout()
                st.pyplot(fig_cov)
                st.info("""
**¿Qué muestra esta matriz?**
La matriz de covarianza refleja la varianza de cada variable (diagonal) y la covarianza entre pares de variables (fuera de la diagonal). Covarianzas altas (positivas o negativas) pueden indicar redundancia. PCA utiliza esta matriz para encontrar combinaciones de variables que expliquen mejor la variabilidad de los datos.
""")
                st.markdown("---")
            elif feature_cols:
                st.info("Selecciona al menos dos variables numéricas para ver las matrices de correlación y covarianza.")

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
                mostrar_metricas_clasificacion(st, metricas, f"Métricas de {algoritmo}")
                
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

with st.sidebar.expander("🧠 SVM: Máquinas de Vectores de Soporte", expanded=False):
    st.markdown(TEXTO_SVM)

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
