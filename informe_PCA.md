# Informe de Resultados: Análisis de Componentes Principales (PCA) sobre Rotación de Personal

## 1. Introducción

Este informe presenta los resultados del Análisis de Componentes Principales (PCA) aplicado al dataset de rotación de personal. El objetivo de PCA es reducir la dimensionalidad del conjunto de datos, identificar patrones y facilitar la visualización, así como mejorar el desempeño de los modelos de clasificación al eliminar redundancias y multicolinealidad.

## 2. Justificación y objetivo del uso de PCA

PCA se utiliza para:
- Explorar la estructura interna de los datos y detectar relaciones entre variables.
- Reducir el número de variables manteniendo la mayor parte de la información (varianza explicada).
- Facilitar la visualización de la separación entre clases en espacios de menor dimensión.
- Mejorar la eficiencia y desempeño de modelos de clasificación al eliminar variables redundantes.

## 3. Resultados del análisis PCA

### 3.1. Varianza explicada por los componentes principales

**Varianza total explicada:** 57.4%

**Distribución por componente:**
- **PC1:** 26.1% de varianza explicada
- **PC2:** 16.1% de varianza explicada  
- **PC3:** 15.1% de varianza explicada
- **PC4:** 13.7% de varianza explicada
- **PC5:** 12.1% de varianza explicada

**Recomendación automática:** El método del codo sugiere usar 5 componentes principales. A partir de aquí, el incremento de varianza explicada es menor a 2%.

### 3.2. Análisis automático de separación de clases

- **Número de clases:** 2
- **Clases más separadas:** Abandonó - Permaneneció
- **Clases más cercanas:** Abandonó - Permaneneció
- **Distancia promedio entre centroides:** 1.01
- **Separación:** Moderada. Existe solapamiento considerable entre clases.

### 3.3. Visualización de los componentes principales

#### Proyección 3D interactiva (PC1 vs PC2 vs PC3)
La visualización 3D muestra una separación moderada entre las clases, con cierto solapamiento en el centro del espacio de componentes principales. Los puntos azules (Permaneneció) y rojos (Abandonó) se distribuyen con patrones diferenciables pero no completamente separables.

#### Proyección 2D (PC1 vs PC2)
La vista bidimensional confirma el solapamiento considerable entre las clases, aunque se observan tendencias de agrupación por clase.

### 3.4. Interpretación de los componentes principales

#### PC1 (26.1% de varianza):
**Combinación lineal:** PC1 = -0.088 * satisfaction_level + 0.507 * last_evaluation + 0.579 * number_project + ...

**Variables más influyentes:**
- `last_evaluation` (0.507)
- `number_project` (0.579) 
- `average_monthly_hours` (0.549)
- `time_spend_company` (0.549)

#### PC2 (16.1% de varianza):
**Combinación lineal:** PC2 = 0.748 * satisfaction_level + 0.332 * last_evaluation - 0.103 * number_project + ...

**Variables más influyentes:**
- `satisfaction_level` (0.748)
- `last_evaluation` (0.332)

#### PC3 (15.1% de varianza):
**Variables más influyentes:**
- `average_monthly_hours` (0.5)
- `time_spend_company` (0.5)
- `Work_accident` (0.5)
- `promotion_last_5years` (0.5)

### 3.5. Matrices de correlación y covarianza

#### Matriz de correlación de componentes principales
Los componentes principales son ortogonales (correlación = 0), confirmando que cada uno captura información única y no redundante.

#### Matriz de covarianza de componentes principales
Las varianzas de cada componente reflejan su importancia relativa:
- PC1: 1.830
- PC2: 1.127
- PC3: 1.060
- PC4: 0.960
- PC5: 0.845

### 3.6. Matriz de componentes principales (loadings)

La matriz muestra cómo cada variable original contribuye a cada componente principal, permitiendo interpretar el significado de cada dimensión reducida.

## 4. Conclusiones y recomendaciones

### 4.1. Hallazgos principales

- **Varianza capturada:** Con 5 componentes principales se explica el 57.4% de la varianza total, una proporción moderada que sugiere que los datos tienen estructura pero también considerable variabilidad no explicada.

- **Separación de clases:** La separación entre "Abandonó" y "Permaneció" es moderada, con solapamiento considerable. Esto indica que la distinción entre clases no es trivial y requiere técnicas sofisticadas de clasificación.

- **Componentes más informativos:** 
  - PC1 captura principalmente información relacionada con la evaluación, número de proyectos, horas mensuales y tiempo en la empresa.
  - PC2 está dominado por el nivel de satisfacción.
  - Los componentes restantes distribuyen la información de manera más equilibrada.

### 4.2. Recomendaciones

- **Uso como preprocesamiento:** PCA puede ser beneficioso para reducir la dimensionalidad antes de aplicar algoritmos de clasificación, especialmente aquellos sensibles a la multicolinealidad.

- **Número de componentes:** Se recomienda usar entre 3-5 componentes principales, balanceando la retención de información con la simplicidad del modelo.

- **Interpretación:** Los primeros dos componentes sugieren que la decisión de abandonar está influenciada principalmente por factores de desempeño (PC1) y satisfacción (PC2).

- **Limitaciones:** El solapamiento considerable entre clases indica que PCA por sí solo no es suficiente para una clasificación precisa, pero puede ser útil como paso de preprocesamiento.

### 4.3. Próximos pasos

1. Evaluar el impacto de usar PCA como preprocesamiento en los modelos de clasificación (LDA, QDA, Bayes Ingenuo).
2. Comparar el desempeño de los modelos con y sin PCA.
3. Considerar técnicas de reducción de dimensionalidad supervisadas que podrían ofrecer mejor separación entre clases.

---
**Este informe complementa los análisis de los modelos de clasificación y será integrado en el reporte comparativo final.**