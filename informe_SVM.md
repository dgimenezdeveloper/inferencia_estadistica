# Informe de Resultados: Máquinas de Vectores de Soporte (SVM) sobre Rotación de Personal

## 1. Introducción

Este informe presenta los resultados del análisis realizado con el algoritmo SVM (Support Vector Machine) para predecir la rotación de personal en el mismo dataset utilizado en los informes anteriores. SVM es un algoritmo de aprendizaje supervisado que busca encontrar el hiperplano óptimo que separa las clases maximizando el margen entre los puntos más cercanos de cada clase.

## 2. Conceptos fundamentales de SVM

### 2.1. Definición
SVM (Support Vector Machine) es un algoritmo de aprendizaje supervisado que busca encontrar el hiperplano óptimo que separa las clases maximizando el margen (distancia entre el hiperplano y los puntos más cercanos de cada clase).

### 2.2. Conceptos clave:
- **Vectores de soporte:** puntos más cercanos al hiperplano de decisión
- **Margen:** distancia entre el hiperplano y los vectores de soporte
- **Kernel:** función que transforma los datos a un espacio de mayor dimensión
- **Hiperparámetro C:** controla el trade-off entre margen y errores de clasificación

### 2.3. ¿Cuándo usar SVM?
- Datos con fronteras complejas no lineales
- Cuando el número de características es mayor que el de muestras
- Cuando necesitas robustez ante outliers
- Para problemas de clasificación binaria o multiclase

## 3. Análisis del dataset y preprocesamiento

### 3.1. Características del dataset:
- **Total de muestras:** 14,999
- **Variables numéricas:** 7
- **Clases:** 2 (Permaneció: 11,428 / Abandonó: 3,571)
- **Desbalance detectado:** Ratio 0.31

### 3.2. Estrategia de muestreo:
- **Muestreo estratificado inteligente (recomendado):** Aplicado
- **Reducción del dataset:** 46.7% (8,001 muestras)
- **Distribución preservada:** Todas las clases representadas
- **Clases balanceadas:** Ratio 0.31 mantenido

### 3.3. Visualización de variables:
Los histogramas muestran distribuciones características:
- **satisfaction_level:** Distribución bimodal con concentración en valores bajos y altos
- **promotion_last_5years:** Altamente desbalanceada (mayoría = 0)
- **Dispersión satisfaction_level vs promotion_last_5years:** Clara separación de clases

### 3.4. Preprocesamiento aplicado:
- **Escalado de datos:** StandardScaler (media=0, std=1) - Recomendado para SVM
- **Datos escalados aplicados:** ✓

## 4. Configuración y optimización del modelo SVM

### 4.1. Optimización automática de hiperparámetros:
- **Grid Search activado:** Búsqueda automática
- **Tiempo de ejecución:** 332.92 segundos
- **Mejores parámetros encontrados:**
  - **C:** 0.1
  - **degree:** 2
  - **gamma:** scale
  - **kernel:** rbf
- **Mejor score (CV):** 0.8051

### 4.2. Configuración final del modelo:
- **Kernel:** RBF (Radial Basis Function)
- **Parámetro de regularización C:** 0.1
- **Gamma:** scale
- **Grado polinómico:** 2

## 5. Resultados del modelo SVM

### 5.1. Métricas globales:
- **Accuracy:** 0.808
- **Precision (Macro):** 0.810
- **Recall (Macro):** 0.618
- **F1-Score (Macro):** 0.636
- **ROC-AUC:** 0.823

### 5.2. Métricas por clase:

| Clase       | Precision | Recall | F1-Score | Soporte |
|-------------|-----------|--------|----------|---------|
| Permaneció  | 0.808     | 0.982  | 0.886    | 11,428  |
| Abandonó    | 0.812     | 0.253  | 0.386    | 3,571   |

**Interpretación:**
- El modelo tiene excelente precisión para ambas clases (>0.80)
- El recall para "Permaneció" es muy alto (0.982), pero muy bajo para "Abandonó" (0.253)
- Esto indica que el modelo es conservador: evita falsos positivos pero sacrifica sensibilidad

### 5.3. Matriz de confusión:

#### Valores absolutos:
|              | Predicho: Permaneció | Predicho: Abandonó |
|--------------|---------------------|--------------------|
| Real: Permaneció | 11,219              | 209                |
| Real: Abandonó   | 2,666               | 905                |

#### Porcentajes:
|              | Predicho: Permaneció | Predicho: Abandonó |
|--------------|---------------------|--------------------|
| Real: Permaneció | 98.2%               | 1.8%               |
| Real: Abandonó   | 74.7%               | 25.3%              |

**Interpretación:**
- **Accuracy:** 0.808 (12,124/14,999 predicciones correctas)
- **Mayor confusión:** Abandonó confundida con Permaneció (2,666 casos - 74.7%)
- El modelo tiene dificultades significativas para identificar casos de abandono

### 5.4. Vectores de soporte:
- **Total de vectores de soporte:** 6,504
- **Porcentaje del dataset:** 43.4%
- Este alto porcentaje sugiere que la separación entre clases no es trivial

### 5.5. Fronteras de decisión:
La visualización de las fronteras de decisión con kernel RBF muestra:
- Separación no lineal compleja entre las clases
- Fronteras curvas que se adaptan a la distribución de los datos
- Los vectores de soporte (puntos marcados) definen estas fronteras

## 6. Conclusiones y comparación con otros algoritmos

### 6.1. Fortalezas de SVM:
- **Alta precisión:** 0.810 (macro), competitiva con los mejores modelos
- **Excelente especificidad:** 98.2% de acierto en clase "Permaneció"
- **Robustez:** El kernel RBF maneja bien las relaciones no lineales
- **ROC-AUC sólido:** 0.823 indica buena capacidad discriminativa

### 6.2. Debilidades identificadas:
- **Baja sensibilidad:** Solo 25.3% de recall para "Abandonó"
- **Desbalance de clases:** El modelo está sesgado hacia la clase mayoritaria
- **Costo computacional:** 332.92 segundos para optimización
- **Interpretabilidad:** Menor que modelos lineales como LDA

### 6.3. Comparación preliminar con otros algoritmos:

| Algoritmo | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
|-----------|----------|-------------------|----------------|------------------|---------|
| LDA       | 0.762    | 0.648            | 0.588          | 0.596            | 0.801   |
| QDA       | 0.906    | 0.862            | 0.895          | 0.877            | 0.945   |
| Bayes     | 0.794    | 0.726            | 0.766          | 0.740            | 0.842   |
| **SVM**   | **0.808**| **0.810**        | **0.618**      | **0.636**        | **0.823**|

**Posición de SVM:**
- 2º en Accuracy (después de QDA)
- 1º en Precision (Macro)
- 3º en Recall (Macro)
- 3º en F1-Score (Macro)
- 3º en ROC-AUC

## 7. Recomendaciones y próximos pasos

### 7.1. Recomendaciones específicas:
- **Técnicas de balanceo:** Aplicar SMOTE, undersampling o class_weight para mejorar el recall de "Abandonó"
- **Ajuste de umbral:** Modificar el umbral de decisión para optimizar recall vs precision
- **Kernels alternativos:** Probar kernel lineal o polinómico para comparar
- **Validación temporal:** Si hay datos temporales, validar en períodos diferentes

### 7.2. Casos de uso recomendados:
- **Cuando la precisión es crítica:** SVM minimiza falsos positivos
- **Screening inicial:** Identificar empleados con alta probabilidad de permanencia
- **Conjunto con otros modelos:** Usar SVM en ensemble con QDA

### 7.3. Limitaciones a considerar:
- No es el mejor para identificar casos de abandono (baja sensibilidad)
- Requiere más tiempo de entrenamiento que otros algoritmos
- Menos interpretable que LDA o Bayes Ingenuo

---
**Este informe será integrado en el reporte comparativo final junto con los demás algoritmos.**