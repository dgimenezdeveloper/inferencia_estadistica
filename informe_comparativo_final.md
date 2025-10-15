# Informe Comparativo Final: Análisis de Algoritmos de Clasificación para Rotación de Personal

## 1. Resumen Ejecutivo

Este informe presenta una comparación exhaustiva de los principales algoritmos de clasificación aplicados al problema de predicción de rotación de personal: LDA, QDA, Bayes Ingenuo, SVM y PCA. El análisis incluye métricas de rendimiento, interpretabilidad y recomendaciones para la implementación en entornos empresariales.

## 2. Metodología y Dataset

### 2.1. Características del problema
- **Objetivo:** Predecir si un empleado abandonará la empresa (clasificación binaria)
- **Variable objetivo:** `left` (0: Permaneció, 1: Abandonó)
- **Variables predictoras:** 7 variables numéricas (satisfaction_level, last_evaluation, number_project, average_monthly_hours, time_spend_company, Work_accident, promotion_last_5years)

### 2.2. Características del dataset
- **Total de observaciones:** 14,999
- **Distribución de clases:** 
  - Permaneció: 11,428 (76.2%)
  - Abandonó: 3,571 (23.8%)
- **Desbalance:** Ratio 0.31 (moderadamente desbalanceado)

### 2.3. Preprocesamiento aplicado
- **PCA:** Reducción de dimensionalidad (57.4% de varianza explicada con 5 componentes)
- **Escalado:** StandardScaler para algoritmos sensibles a escala
- **Muestreo estratificado:** Para SVM (8,001 muestras preservando distribución)

## 3. Resultados Comparativos

### 3.1. Tabla de métricas principales

| Algoritmo     | Con PCA | Sin PCA | Observaciones |
|---------------|---------|---------|---------------|
| **Bayes Ingenuo** | 0.7747 | 0.7922 | Mejor sin PCA |
| **LDA**           | 0.7384 | 0.7392 | Sin diferencia significativa |
| **QDA**           | 0.9062 | 0.858  | Mejor sin PCA |
| **SVM Linear**    | 0.7434 | 0.7013 | Mejor con PCA |
| **SVM RBF**       | 0.9641 | 0.8546 | Mejor con PCA |

### 3.2. Análisis de discrepancias en SVM

**Observación importante:** Se detectaron diferencias significativas entre las métricas de SVM en la vista individual vs. la comparativa:

- **Vista individual SVM:** Accuracy = 0.808, ROC-AUC = 0.823
- **Vista comparativa SVM RBF:** Accuracy = 0.9641, ROC-AUC = 0.8546

**Posibles causas de la discrepancia:**
1. **Diferentes conjuntos de datos:** La comparativa puede usar el dataset completo vs. el muestreado en la vista individual
2. **Diferentes hiperparámetros:** La optimización puede haber encontrado parámetros distintos
3. **Métodos de validación diferentes:** Cross-validation vs. hold-out
4. **Efecto del PCA:** En la comparativa se evalúa con y sin PCA

## 4. Ranking de Algoritmos por Métrica

### 4.1. Por Accuracy (sin PCA)
1. **QDA:** 0.906 ⭐️ **GANADOR**
2. **SVM RBF:** 0.855
3. **Bayes Ingenuo:** 0.794
4. **LDA:** 0.762
5. **SVM Linear:** 0.743

### 4.2. Por Accuracy (con PCA)
1. **SVM RBF:** 0.964 ⭐️ **GANADOR CON PCA**
2. **QDA:** 0.906
3. **Bayes Ingenuo:** 0.775
4. **SVM Linear:** 0.743
5. **LDA:** 0.738

## 5. Análisis Detallado por Algoritmo

### 5.1. QDA (Análisis Discriminante Cuadrático)
**Fortalezas:**
- Mejor accuracy sin PCA (0.906)
- Excelente para patrones no lineales
- Maneja bien diferentes matrices de covarianza por clase

**Debilidades:**
- Puede ser sensible a outliers
- Requiere más datos para estimar matrices de covarianza

**Recomendación:** ⭐️ **ALGORITMO PRINCIPAL**

### 5.2. SVM RBF (con PCA)
**Fortalezas:**
- Mejor accuracy con PCA (0.964)
- Maneja relaciones no lineales complejas
- Robusto ante outliers

**Debilidades:**
- Computacionalmente costoso
- Menos interpretable
- Sensible a hiperparámetros

**Recomendación:** ⭐️ **ALGORITMO ALTERNATIVO CON PCA**

### 5.3. Bayes Ingenuo
**Fortalezas:**
- Simplicidad e interpretabilidad
- Rápido entrenamiento
- Funciona bien con datasets pequeños

**Debilidades:**
- Asume independencia entre variables
- Rendimiento moderado

**Recomendación:** 📊 **LÍNEA BASE**

### 5.4. LDA (Análisis Discriminante Lineal)
**Fortalezas:**
- Interpretabilidad alta
- Asume frontera lineal
- Computacionalmente eficiente

**Debilidades:**
- Limitado a relaciones lineales
- Rendimiento inferior

**Recomendación:** 📈 **ANÁLISIS EXPLORATORIO**

### 5.5. SVM Linear
**Fortalezas:**
- Más interpretable que SVM RBF
- Eficiente computacionalmente

**Debilidades:**
- Rendimiento inferior
- Limitado a patrones lineales

**Recomendación:** ⚙️ **USO ESPECÍFICO**

## 6. Impacto del PCA

### 6.1. Justificación y observaciones del análisis comparativo:

**El mejor modelo según la métrica seleccionada es SVM RBF sin PCA, con un valor de 0.564.**

- **Si PCA mejora el rendimiento:** Probablemente hay redundancia o ruido en las variables
- **Si SVM destaca:** Puede indicar fronteras no lineales; si LDA/QDA, las clases pueden ser linealmente separables
- **Si Bayes Ingenuo es competitivo:** Las variables pueden ser casi independientes
- **Observa la desviación estándar:** Valores altos indican inestabilidad o sensibilidad a la partición

### 6.2. Efecto del PCA por algoritmo:
- **Mejoran con PCA:** SVM Linear (+0.04), SVM RBF (+0.11)
- **Empeoran con PCA:** Bayes Ingenuo (-0.01), QDA (-0.05)
- **Sin cambio significativo:** LDA (±0.00)

## 7. Conclusión y Recomendación Final

### 7.1. Recomendación Principal
✅ **Se recomienda usar SVM RBF sin PCA para este dataset, según la métrica seleccionada.**

**Justificación:**
- Accuracy superior en la comparativa
- Capacidad para manejar relaciones no lineales complejas
- Robustez ante outliers

### 7.2. Consideraciones para la implementación:
- **Revisar los supuestos teóricos y la interpretabilidad antes de decidir el modelo final**
- **Validar resultados con datos no vistos**
- **Considerar ensemble de QDA + SVM RBF para máximo rendimiento**
- **Monitorear performance en producción**

### 7.3. Estrategia recomendada:
1. **Modelo principal:** SVM RBF (sin PCA)
2. **Modelo de respaldo:** QDA
3. **Análisis exploratorio:** LDA + PCA
4. **Línea base:** Bayes Ingenuo

## 8. Limitaciones y Próximos Pasos

### 8.1. Limitaciones identificadas:
- Discrepancias en métricas entre vistas individuales y comparativa
- Desbalance de clases no completamente abordado
- Falta de análisis temporal si los datos lo permiten

### 8.2. Próximos pasos recomendados:
1. **Investigar discrepancias en métricas de SVM**
2. **Aplicar técnicas de balanceo (SMOTE, class_weight)**
3. **Validación cruzada temporal**
4. **Análisis de importancia de variables**
5. **Implementación en entorno de producción con monitoreo**

---

**Nota:** Para decisiones críticas de negocio, se recomienda validar estos resultados con un conjunto de datos independiente y considerar la interpretabilidad del modelo junto con su rendimiento predictivo.