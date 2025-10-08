import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize

# Las funciones que dependen de Streamlit reciben 'st' como argumento
def calcular_metricas_clasificacion(y_true, y_pred, y_prob=None, class_names=None, st=None):
    """
    Calcula todas las métricas de evaluación para clasificación
    """
    metricas = {}
    metricas['accuracy'] = accuracy_score(y_true, y_pred)
    metricas['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metricas['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metricas['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y_true, y_pred)
    support_per_class = np.array([(y_true == l).sum() for l in labels])
    metricas['precision_per_class'] = precision_per_class
    metricas['recall_per_class'] = recall_per_class
    metricas['f1_per_class'] = f1_per_class
    metricas['support_per_class'] = support_per_class
    cm = confusion_matrix(y_true, y_pred)
    metricas['confusion_matrix'] = cm
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                metricas['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                y_true_bin = label_binarize(y_true, classes=classes)
                metricas['roc_auc'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
        except:
            metricas['roc_auc'] = None
    # Nombres de clases
    labels_map = st.session_state.get("clase_labels_global", {}) if st else {}
    if class_names is None:
        class_names = [labels_map.get(i, str(i)) for i in np.unique(y_true)]
    else:
        class_names = [labels_map.get(i, str(i)) for i in class_names]
    metricas['class_names'] = class_names
    return metricas

def mostrar_metricas_clasificacion(metricas, st, titulo="Métricas de Evaluación"):
    """
    Muestra las métricas de clasificación en Streamlit de forma organizada
    """
    st.write(f"### {titulo}")
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
    for c in ['Precision','Recall','F1-Score']:
        tabla_metricas[c] = tabla_metricas[c].apply(lambda x: float(x) if isinstance(x, (float, np.floating, int, np.integer)) else np.nan)
    def pastel_metric(val):
        if pd.isnull(val) or val == 0:
            return 'background-color: #ffffff; color: #111; font-weight: bold;'
        if val >= 0.8:
            color = '#81c784'
        elif val >= 0.6:
            color = '#fff176'
        elif val > 0.0:
            color = '#e57373'
        else:
            color = '#ffffff'
        return f'background-color: {color}; color: #111; font-weight: bold;'
    styled = tabla_metricas.style.applymap(pastel_metric, subset=['Precision','Recall','F1-Score'])\
        .format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-Score': '{:.3f}', 'Support': '{:.0f}'})
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("""
<div style='margin: 0.5em 0 1em 0; font-size: 0.95em;'>
<strong>Glosario de colores:</strong><br>
<span style='background-color:#81c784; color:#111; padding:2px 8px; border-radius:4px;'>🟩 Verde</span> = valor alto (≥ 0.8, buen desempeño)<br>
<span style='background-color:#fff176; color:#111; padding:2px 8px; border-radius:4px;'>🟨 Amarillo</span> = valor medio (≥ 0.6, aceptable)<br>
<span style='background-color:#e57373; color:#111; padding:2px 8px; border-radius:4px;'>🟥 Rojo</span> = valor bajo (> 0.0, necesita mejora)<br>
<span style='background-color:#fff; color:#111; padding:2px 8px; border-radius:4px;'>⬜ Blanco</span> = cero o nulo
</div>
""", unsafe_allow_html=True)
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

def visualizar_matriz_confusion_mejorada(cm, class_names, st, titulo="Matriz de Confusión"):
    """
    Crea una visualización mejorada de la matriz de confusión
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_percent = cm.astype('float') / row_sums * 100
    cm_percent = np.round(cm_percent, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Matriz de Confusión (Valores Absolutos)')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Real')
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_title('Matriz de Confusión (Porcentajes)')
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Real')
    percent_sums = cm_percent.sum(axis=1)
    for i, total in enumerate(percent_sums):
        ax2.text(len(class_names), i + 0.5, f"{total:.1f}", va='center', ha='left', color='black', fontsize=9, fontweight='bold')
    ax2.set_xlim(-0.5, len(class_names) + 0.5)
    ax2.set_xticks(list(ax2.get_xticks()) + [len(class_names)])
    ax2.set_xticklabels(list(class_names) + ['Σ'], rotation=45 if len(class_names) > 3 else 0)
    plt.tight_layout()
    st.pyplot(fig)
    diagonal_sum = np.trace(cm)
    total_sum = np.sum(cm)
    accuracy = diagonal_sum / total_sum
    st.write("#### Interpretación de la matriz de confusión:")
    st.write(f"- **Accuracy**: {accuracy:.3f} ({diagonal_sum}/{total_sum} predicciones correctas)")
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_confusion = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
    if cm_no_diag[max_confusion] > 0:
        st.write(f"- **Mayor confusión**: {class_names[max_confusion[0]]} confundida con {class_names[max_confusion[1]]} ({cm_no_diag[max_confusion]} casos)")

def crear_curvas_roc_interactivas(y_true, y_prob, class_names, go, px):
    """
    Crea curvas ROC interactivas con Plotly
    """
    fig = go.Figure()
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Clasificador aleatorio',
            line=dict(color='navy', dash='dash', width=2),
            hovertemplate='Línea de referencia<extra></extra>'
        ))
    else:
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

def calcular_metricas_regresion(y_true, y_pred, st, titulo="Métricas de Regresión"):
    """
    Calcula y muestra métricas para problemas de regresión
    """
    metricas = {}
    metricas['mse'] = mean_squared_error(y_true, y_pred)
    metricas['rmse'] = np.sqrt(metricas['mse'])
    metricas['mae'] = mean_absolute_error(y_true, y_pred)
    metricas['r2'] = r2_score(y_true, y_pred)
    st.write(f"### {titulo}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R^2 Score", f"{metricas['r2']:.3f}")
    with col2:
        st.metric("RMSE", f"{metricas['rmse']:.3f}")
    with col3:
        st.metric("MAE", f"{metricas['mae']:.3f}")
    with col4:
        st.metric("MSE", f"{metricas['mse']:.3f}")
    st.markdown(f"""
- 1.0 = Perfecto ajuste  
- 0.0 = Modelo no mejor que predecir la media  
- <0.0 = Modelo peor que predecir la media  
**R^2 Score ({metricas['r2']:.3f})**: Proporción de varianza explicada por el modelo.  
{'✅ Excelente' if metricas['r2'] > 0.9 else '✅ Bueno' if metricas['r2'] > 0.7 else '⚠️ Regular' if metricas['r2'] > 0.5 else '❌ Pobre'}  
**RMSE ({metricas['rmse']:.3f})**: Error cuadrático medio. Mismas unidades que la variable objetivo.  
Penaliza más los errores grandes.  
**MAE ({metricas['mae']:.3f})**: Error absoluto medio. Mismas unidades que la variable objetivo.  
Menos sensible a valores atípicos que RMSE.  
**MSE ({metricas['mse']:.3f})**: Error cuadrático medio. Unidades al cuadrado.  
Base matemática para RMSE.
""")
    with st.expander("💡 Interpretación de métricas de regresión"):
        st.markdown(f"""
**R^2 Score ({metricas['r2']:.3f})**: Proporción de varianza explicada por el modelo.
{'✅ Excelente' if metricas['r2'] > 0.9 else '✅ Bueno' if metricas['r2'] > 0.7 else '⚠️ Regular' if metricas['r2'] > 0.5 else '❌ Pobre'}
**RMSE ({metricas['rmse']:.3f})**: Error cuadrático medio. Mismas unidades que la variable objetivo.
Penaliza más los errores grandes.
**MAE ({metricas['mae']:.3f})**: Error absoluto medio. Mismas unidades que la variable objetivo.
Menos sensible a valores atípicos que RMSE.
**MSE ({metricas['mse']:.3f})**: Error cuadrático medio. Unidades al cuadrado.
Base matemática para RMSE.
""")
    fig_reg, ax_reg = plt.subplots(figsize=(8, 6))
    ax_reg.scatter(y_true, y_pred, alpha=0.6)
    ax_reg.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax_reg.set_xlabel('Valores Reales')
    ax_reg.set_ylabel('Valores Predichos')
    ax_reg.set_title('Valores Reales vs Predichos')
    ax_reg.grid(True, alpha=0.3)
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax_reg.plot(y_true, p(y_true), "b--", alpha=0.8, linewidth=1, label=f'Ajuste lineal')
    ax_reg.legend()
    st.pyplot(fig_reg)
    residuos = y_true - y_pred
    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(y_pred, residuos, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Valores Predichos')
    ax1.set_ylabel('Residuos')
    ax1.set_title('Residuos vs Valores Predichos')
    ax1.grid(True, alpha=0.3)
    ax2.hist(residuos, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuos')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Residuos')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_res)
    return metricas