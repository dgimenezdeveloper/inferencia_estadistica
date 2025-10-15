---
# Informe de Resultados: Análisis Discriminante Lineal (LDA) sobre Rotación de Personal

## 1. Introducción

La rotación de personal representa un desafío significativo para las organizaciones, generando costos asociados a la pérdida de talento, la búsqueda y capacitación de nuevos empleados, y la disminución de la productividad. Anticipar qué empleados tienen mayor riesgo de abandonar la empresa permite tomar medidas proactivas para retener talento y mantener la estabilidad organizacional.

En este informe se presentan los resultados del análisis realizado sobre un dataset de empleados, aplicando el algoritmo de Análisis Discriminante Lineal (LDA) para predecir la variable objetivo `left` (abandono/no abandono).

## 2. Análisis Exploratorio del Dataset

### 2.1. Descripción de la base

El dataset contiene información relevante sobre empleados, incluyendo variables numéricas y categóricas. La variable objetivo es `left`, que indica si el empleado abandonó (1) o permaneció (0) en la empresa.

**Variables principales seleccionadas:**

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_monthly_hours`
- `time_spend_company`
- `Work_accident`
- `promotion_last_5years`

### 2.2. Análisis de valores faltantes (NA)

Se realizó una revisión de valores faltantes. No se detectaron NA significativos en las variables seleccionadas, por lo que no fue necesario imputar ni eliminar registros.

### 2.3. Balanceo de clases

| Clase        | Cantidad |
|--------------|----------|
| Permaneció   | 11.428   |
| Abandonó     | 3.571    |

El dataset presenta cierto desbalance, con una mayor proporción de empleados que permanecieron. Esto puede influir en las métricas de desempeño y debe considerarse al interpretar los resultados.

### 2.4. Variables categóricas

Existen variables categóricas en la base. Para este análisis inicial con LDA, se priorizaron las variables numéricas. Las categóricas pueden ser transformadas a variables cuantitativas (one-hot encoding) en futuros análisis para enriquecer el modelo.

## 3. Selección de Variables para LDA

Se seleccionaron variables numéricas relevantes para la predicción, considerando su relación con la variable objetivo y la ausencia de multicolinealidad fuerte (ver matriz de correlación).

#### Matriz de correlación entre variables seleccionadas
Se observó que las variables elegidas no presentan correlaciones excesivamente altas entre sí, lo que favorece el desempeño de LDA.

## 4. Resultados del Modelo LDA

### 4.1. Métricas globales

- **Accuracy:** 0.762
- **Precision (Macro):** 0.648
- **Recall (Macro):** 0.588
- **F1-Score (Macro):** 0.596
- **ROC-AUC:** 0.801

### 4.2. Métricas por clase

| Clase       | Precision | Recall | F1-Score | Soporte |
|-------------|-----------|--------|----------|---------|
| Permaneció  | 0.798     | 0.920  | 0.855    | 11.428  |
| Abandonó    | 0.499     | 0.256  | 0.338    | 3.571   |

**Interpretación:**
- El modelo predice con mayor precisión y recall la clase "Permaneció".
- La clase "Abandonó" presenta menor recall y F1-score, lo que indica dificultad para identificar correctamente a quienes abandonan.

### 4.3. Validación cruzada (CV=5)

- **Accuracy (CV=5):** 0.738 ±0.061
- **Precision (CV=5):** 0.632 ±0.069
- **Recall (CV=5):** 0.574 ±0.039
- **F1-Score (CV=5):** 0.582 ±0.044

### 4.4. Matriz de confusión

|              | Predicho: Permaneció | Predicho: Abandonó |
|--------------|---------------------|--------------------|
| Real: Permaneció | 10509                | 919                |
| Real: Abandonó   | 2657                 | 914                |

**Mayor confusión:** El modelo tiende a confundir empleados que abandonaron con la clase "Permaneció" (2657 casos).

### 4.5. Curva ROC y AUC

El área bajo la curva ROC (AUC = 0.801) indica una buena capacidad discriminativa del modelo para distinguir entre ambas clases.

### 4.6. Visualización de separación de clases

La proyección de los componentes discriminantes muestra cierta superposición entre las clases, aunque se observa una tendencia a la separación.

## 5. Conclusiones y Recomendaciones

- El modelo LDA logra un desempeño aceptable, especialmente para la clase mayoritaria (Permaneció), pero tiene dificultades para identificar correctamente a quienes abandonan la empresa.
- El desbalance de clases afecta el recall y F1-score de la clase minoritaria.
- Se recomienda explorar técnicas de balanceo (submuestreo, sobremuestreo) y la inclusión de variables categóricas transformadas para mejorar la capacidad predictiva.
- El AUC > 0.8 sugiere que el modelo tiene potencial, pero requiere ajustes para ser útil en la práctica.

## 6. Próximos pasos

1. Probar otros algoritmos (QDA, Bayes, SVM, etc.) y comparar resultados.
2. Incluir variables categóricas mediante codificación adecuada.
3. Aplicar técnicas de balanceo de clases.
4. Analizar la importancia de variables y realizar ingeniería de atributos.
5. Validar los modelos con datos no vistos y ajustar hiperparámetros.

---
**Este informe será complementado con los resultados de los demás algoritmos para una comparación robusta y recomendaciones finales.**