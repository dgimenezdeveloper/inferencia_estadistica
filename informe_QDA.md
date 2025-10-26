
# Informe de Resultados: Análisis Discriminante Cuadrático (QDA) sobre Rotación de Personal

## 1. Introducción

Este informe presenta los resultados del análisis realizado con el algoritmo QDA (Análisis Discriminante Cuadrático) para predecir la rotación de personal en la misma base utilizada en el informe de LDA. El objetivo es comparar el desempeño de QDA frente a LDA y otros modelos, y extraer conclusiones sobre su utilidad práctica.

## 2. Consideraciones sobre el dataset

El análisis exploratorio, la selección de variables y el tratamiento de datos se mantuvieron igual que en el informe de LDA (ver `informe_LDA.md`). Se utilizaron las mismas variables numéricas y la variable objetivo `left`.

## 3. Resultados del Modelo QDA

### 3.1. Matriz de covarianza por clase
QDA estima una matriz de covarianza diferente para cada clase, lo que le permite modelar relaciones más complejas entre variables.

#### Covarianza clase 0 (Permaneció)
Valores destacados en la diagonal (varianzas) y fuera de la diagonal (covarianzas) muestran la dispersión y relación entre variables para quienes permanecieron.

#### Covarianza clase 1 (Abandonó)
Se observa mayor dispersión en algunas variables, lo que puede ayudar a QDA a distinguir mejor los patrones de abandono.

### 3.2. Métricas globales

- **Accuracy:** 0.906
- **Precision (Macro):** 0.862
- **Recall (Macro):** 0.895
- **F1-Score (Macro):** 0.877
- **ROC-AUC:** 0.945

### 3.3. Métricas por clase

| Clase       | Precision | Recall | F1-Score | Soporte |
|-------------|-----------|--------|----------|---------|
| Permaneció  | 0.959     | 0.917  | 0.937    | 11.428  |
| Abandonó    | 0.766     | 0.874  | 0.816    | 3.571   |

**Interpretación:**
- QDA mejora notablemente la predicción de la clase "Abandonó" respecto a LDA, logrando un recall y F1-score mucho más altos.
- El desempeño en la clase mayoritaria sigue siendo excelente.

### 3.4. Validación cruzada (CV=5)

- **Accuracy (CV=5):** 0.905 ±0.015
- **Precision (CV=5):** 0.862 ±0.023
- **Recall (CV=5):** 0.895 ±0.010
- **F1-Score (CV=5):** 0.876 ±0.018

### 3.5. Matriz de confusión

|              | Predicho: Permaneció | Predicho: Abandonó |
|--------------|---------------------|--------------------|
| Real: Permaneció | 10474                | 954                |
| Real: Abandonó   | 451                  | 3120               |

**Mayor confusión:** El modelo tiende a confundir algunos casos de "Permaneció" con "Abandonó" (954 casos), pero identifica correctamente la mayoría de los abandonos.

### 3.6. Curva ROC y AUC

El área bajo la curva ROC (AUC = 0.945) indica una excelente capacidad discriminativa del modelo para distinguir entre ambas clases.

### 3.7. Visualización de separación de clases

La proyección de los componentes discriminantes muestra una mejor separación entre las clases respecto a LDA, tanto en 2D como en 3D.

## 4. Conclusiones y Comparación con LDA

- QDA supera ampliamente a LDA en todas las métricas, especialmente en la identificación de empleados que abandonan la empresa.
- El recall y F1-score de la clase "Abandonó" son significativamente mayores, lo que lo hace más útil para la toma de decisiones.
- El modelo mantiene un excelente desempeño en la clase mayoritaria.
- La validación cruzada confirma la robustez del modelo.
- El AUC cercano a 0.95 refuerza la capacidad predictiva de QDA.

## 5. Recomendaciones y próximos pasos

- QDA es el modelo recomendado para este problema, según los resultados obtenidos.
- Se sugiere analizar la estabilidad del modelo ante la inclusión de variables categóricas y técnicas de balanceo.
- Continuar con la comparación frente a otros algoritmos (Bayes, SVM, etc.) para una decisión final.

---
**Este informe será integrado en el reporte comparativo final junto con los demás algoritmos.**