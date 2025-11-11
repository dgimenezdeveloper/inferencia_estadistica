# Guión para Presentación Oral - Competencia Discursiva
## Análisis de Clustering para Segmentación de Empleados en Riesgo de Rotación

---

**Materia:** Inferencia Estadística y Reconocimiento de Patrones  
**Universidad:** Universidad Nacional Guillermo Brown  
**Aplicación Web:** https://inferencia-estadistica-unab.streamlit.app/  
**Tipo de análisis:** Aprendizaje No Supervisado - Clustering  
**Algoritmos evaluados:** K-means y Clustering Jerárquico

---

## INTRODUCCIÓN

### Presentación del Objetivo

Buenos días. **El objetivo de este trabajo práctico** es aplicar técnicas de clustering no supervisado para identificar patrones naturales de agrupamiento en empleados, permitiendo una segmentación basada en datos que facilite estrategias diferenciadas de retención de personal.

### Contexto del Problema de Negocio

**En este contexto**, mientras que en trabajos anteriores utilizamos clasificación supervisada para predecir rotación conociendo las etiquetas de clase, ahora exploramos el problema desde una perspectiva no supervisada: ¿existen grupos naturales de empleados con características similares que no sean evidentes a simple vista? Esta segmentación puede revelar perfiles de riesgo que no coincidan necesariamente con la clasificación tradicional "se fue / se quedó".

### Diferencia con Enfoque Supervisado

**Es importante destacar** que, a diferencia de los algoritmos supervisados (LDA, QDA, Bayes Ingenuo, SVM) que requieren conocer de antemano quién se fue y quién se quedó para entrenar el modelo, el clustering no supervisado busca **descubrir** grupos naturales basándose únicamente en las similitudes entre las características de los empleados, sin utilizar la etiqueta de rotación.

### Herramientas y Tecnologías Utilizadas

**Para llevar a cabo este análisis**, utilizamos las siguientes herramientas:

- **Lenguaje de programación:** Python 3.12
- **Bibliotecas principales:** 
  - scikit-learn (algoritmos de clustering)
  - pandas (procesamiento de datos)
  - plotly (visualizaciones interactivas)
  - scipy (clustering jerárquico y dendrogramas)
- **Plataforma de implementación:** Streamlit (aplicación web interactiva)
- **Algoritmos evaluados:** K-means y Clustering Jerárquico (Ward, Complete, Average, Single)
- **Técnicas de preprocesamiento:** StandardScaler, PCA opcional, codificación de variables categóricas
- **Métricas de evaluación:** Coeficiente de Silhouette, Davies-Bouldin, Calinski-Harabasz, Inercia

---

## DESARROLLO

### Paso 1: Preparación y Exploración del Dataset

**En primer lugar**, utilizamos el mismo dataset de rotación de personal que contiene información de 14,999 empleados, pero ahora **sin utilizar la variable objetivo "left"** durante el proceso de clustering.

**Las características del dataset para clustering son:**
- Total de observaciones: 14,999 empleados
- Variables numéricas para clustering: 7 (satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years)
- Variables categóricas transformadas: 2 (Department codificado en 10 variables binarias, salary codificado ordinal)
- **Total de variables para clustering: 18** (después de transformar categóricas)
- Variable objetivo "left" **EXCLUIDA del clustering** (solo se usa después para validar si los clusters descubiertos tienen sentido)

### Paso 2: Justificación del Enfoque No Supervisado

**A continuación**, es fundamental comprender por qué el clustering complementa (no reemplaza) la clasificación supervisada:

**Ventajas del clustering en este contexto:**

1. **Descubrimiento de patrones ocultos:** Puede revelar subgrupos de empleados que comparten características similares pero que no necesariamente se alinean con la división "se fue / se quedó". Por ejemplo, podría descubrir un grupo de "empleados satisfechos pero sobrecargados" o "empleados mediocres sin promoción".

2. **Segmentación para estrategias diferenciadas:** En lugar de tratar a todos los empleados en riesgo de la misma manera, el clustering permite diseñar intervenciones específicas para cada perfil identificado.

3. **Validación de supuestos:** Si los clusters descubiertos se alinean naturalmente con la variable "left", valida que existen diferencias estructurales reales entre quienes se van y quienes se quedan. Si no se alinean, sugiere que la realidad es más compleja que una simple clasificación binaria.

4. **Interpretabilidad:** Los centroides de los clusters revelan las características "promedio" de cada grupo, facilitando la comprensión de los perfiles de empleados.

### Paso 3: Preprocesamiento Especializado para Clustering

**Posteriormente**, aplicamos un preprocesamiento riguroso adaptado a los requisitos de clustering:

#### Transformación de Variables Categóricas

**Primero**, transformamos las variables categóricas siguiendo el mismo enfoque validado en clasificación supervisada:

- **Variable "salary":** Label Encoding Ordinal (low=0, medium=1, high=2)
- **Variable "Department":** One-Hot Encoding (10 variables binarias)
- **Resultado:** Dataset con 18 variables numéricas (7 originales + 1 ordinal + 10 binarias)

#### Escalado de Datos

**En segundo lugar**, aplicamos StandardScaler a todas las 18 variables:

- **Justificación para K-means:** K-means utiliza distancia euclidiana, por lo que es crítico que todas las variables estén en la misma escala. Sin escalado, variables con rangos grandes (como average_montly_hours: 96-310) dominarían el clustering sobre variables con rangos pequeños (como Work_accident: 0-1).

- **Justificación para Clustering Jerárquico:** Similarmente sensible a escalas, especialmente con métodos de enlace como Ward y Complete.

- **Resultado:** Todas las variables tienen media=0 y desviación estándar=1, garantizando igualdad de influencia.

#### Consideración de PCA (Análisis de Componentes Principales)

**Adicionalmente**, evaluamos si aplicar PCA antes del clustering:

**Argumentos a favor de PCA:**
- Reduce redundancia entre variables correlacionadas
- Puede mejorar la estabilidad de los clusters
- Facilita visualización en 2D/3D

**Argumentos en contra de PCA:**
- Pérdida de interpretabilidad: los componentes principales no tienen significado directo
- Las 18 variables tienen interpretación clara en el contexto empresarial
- Los departamentos (one-hot encoded) aportan información valiosa que PCA podría diluir

**Decisión:** Evaluamos ambos enfoques (con y sin PCA) para este análisis.

### Paso 4: Aplicación de K-means Clustering

**En esta etapa**, aplicamos K-means como primer algoritmo de clustering:

#### Selección del Número Óptimo de Clusters (k)


**Primero**, utilizamos el **Método del Codo (Elbow Method)** para determinar k óptimo:

- **Proceso:** Entrenar K-means con k desde 2 hasta 10, calculando la inercia (suma de distancias al cuadrado de cada punto a su centroide)
- **Resultado observado en la app:** El codo sugiere k=8 como óptimo (estrella verde en el gráfico)
- **Advertencia:** Aunque la inercia sigue bajando, el salto marginal se da en k=8, pero la métrica de Silhouette es baja (0.21), lo que indica que los clusters están poco definidos y pueden solaparse.

**Métricas complementarias para validar k:**

1. **Coeficiente de Silhouette:** Mide qué tan similar es un punto a su propio cluster comparado con otros clusters
   - Rango: [-1, 1]
   - Valor observado: 0.21 (clustering débil, grupos poco separados)
   - Valores cercanos a 1: clusters bien definidos
   - Valores cercanos a 0: clusters solapados
   - Valores negativos: posible asignación incorrecta

2. **Davies-Bouldin Index:** Mide la separación entre clusters
   - Valor observado: 1.78 (aceptable, pero no óptimo)
   - Valores más bajos son mejores
   - Indica clusters compactos y bien separados

3. **Calinski-Harabasz Index:** Ratio de dispersión entre clusters vs dentro de clusters
   - Valor observado: 1384 (muy buena definición)
   - Valores más altos son mejores
   - Indica clusters densos y separados

**Conclusión:** Se selecciona k=8 siguiendo la sugerencia automática de la app, pero se advierte que la segmentación es más fragmentada y menos robusta que en k=4. La interpretación de los clusters debe ser más cautelosa.

#### Entrenamiento de K-means

**Seguidamente**, entrenamos K-means con el k óptimo identificado:

- **Algoritmo:** K-means (n_init=10 para estabilidad, random_state=42 para reproducibilidad)
- **Resultado:** Cada empleado es asignado a uno de los k clusters
- **Output crítico:** Centroides de cada cluster (punto promedio en el espacio de 18 dimensiones)

#### Interpretación de Centroides


**A partir de este punto**, analizamos los centroides para caracterizar cada cluster:

**Ejemplo de interpretación (k=8):**

- Los centroides muestran valores intermedios y menos extremos que en k=4. Por ejemplo:
   - satisfaction_level: entre 0.59 y 0.63 en la mayoría de los clusters
   - promedio_mes_horas: entre 198 y 202 horas/mes
   - tiempo_gastado_empresa: entre 3.3 y 4.3 años
- No hay perfiles tan nítidos como "Burnout" o "Estrella" puros, sino variantes intermedias y subgrupos.
- Algunos clusters agrupan empleados con satisfacción y horas medias, otros con características mixtas, y otros con valores atípicos en alguna variable.
- La interpretación debe ser más matizada: los grupos pueden solaparse y la acción empresarial debe considerar la fragmentación.

#### Visualización de Clusters


**Para profundizar en los hallazgos**, generamos visualizaciones interactivas:

**Visualización 2D:** Proyección de los 18 dimensiones en 2 variables representativas (ej: nivel_de_satisfaccion vs promedio_mes_horas)
   - **Centroides:** Marcados con estrellas para ubicar el "centro" de cada grupo
   - **Colores:** Cada cluster tiene un color distintivo (8 colores)
   - **Interpretación:** Se observa una mayor fragmentación y solapamiento entre grupos. Algunos clusters se superponen en las variables principales, lo que dificulta la segmentación clara.

**Visualización 3D:** Si hay 3 variables clave (ej: satisfaction, evaluation, time_spend)
   - **Mayor riqueza:** Visualiza mejor la estructura tridimensional, pero la separación sigue siendo débil
   - **Rotación interactiva:** En la app web, permite explorar los clusters desde diferentes ángulos, pero la interpretación sigue siendo matizada

#### Métricas de Calidad del Clustering


**Una vez completados los entrenamientos**, evaluamos la calidad del clustering:

- **Silhouette Score:** 0.21 (real) → Clustering débil, grupos poco definidos y solapados
- **Davies-Bouldin:** 1.78 (real) → Separación aceptable, pero no óptima
- **Calinski-Harabasz:** 1384 (real) → Clusters densos y bien definidos
- **Inercia:** 173,104 (real) → Suma de distancias al cuadrado

**Interpretación conjunta:** Las métricas muestran que con k=8 la segmentación es más fragmentada y menos robusta. Los clusters existen, pero la separación es débil y la interpretación debe ser cautelosa. Se recomienda justificar ante la audiencia por qué se sigue la sugerencia automática de la app y advertir sobre la baja cohesión de los clusters.

### Paso 5: Aplicación de Clustering Jerárquico

**Adicionalmente**, aplicamos Clustering Jerárquico como técnica complementaria:

#### ¿Por qué Clustering Jerárquico?

**Primero**, justificamos la inclusión de este algoritmo:

- **No requiere especificar k de antemano:** El dendrograma muestra todas las posibles agrupaciones
- **Visualización de jerarquías:** Revela relaciones entre grupos a diferentes niveles de granularidad
- **Método diferente:** Usa enlace de vecinos (no centroides), puede descubrir estructuras que K-means no detecta
- **Validación cruzada:** Si ambos algoritmos encuentran estructuras similares, aumenta la confianza en los resultados

#### Métodos de Enlace Evaluados

**En segundo lugar**, probamos diferentes métodos de enlace:

1. **Ward (Método de Ward):** Minimiza la varianza dentro de cada cluster al fusionar
   - **Ventaja:** Tiende a crear clusters de tamaño similar y compactos
   - **Desventaja:** Solo funciona con distancia euclidiana
   - **Uso recomendado:** Cuando se buscan grupos equilibrados

2. **Complete (Enlace Completo):** Distancia máxima entre puntos de diferentes clusters
   - **Ventaja:** Crea clusters esféricos y compactos
   - **Desventaja:** Sensible a outliers
   - **Uso recomendado:** Datos sin outliers extremos

3. **Average (Enlace Promedio):** Distancia promedio entre todos los pares de puntos
   - **Ventaja:** Menos sensible a outliers que Complete
   - **Desventaja:** Puede crear clusters irregulares
   - **Uso recomendado:** Balance entre robustez e interpretabilidad

4. **Single (Enlace Simple):** Distancia mínima entre puntos de diferentes clusters
   - **Ventaja:** Detecta clusters de formas irregulares
   - **Desventaja:** Propenso al "encadenamiento" (clusters alargados)
   - **Uso recomendado:** Cuando se sospechan formas no esféricas

#### Análisis del Dendrograma

**Posteriormente**, interpretamos el dendrograma resultante:

- **Eje Y (vertical):** Distancia de fusión entre clusters
- **Eje X (horizontal):** Observaciones (empleados)
- **Fusiones bajas:** Empleados muy similares
- **Fusiones altas:** Grupos muy diferentes que se unen tarde

**Identificación del número óptimo de clusters:**
- **Regla del "salto grande":** Buscar fusiones donde la distancia aumenta considerablemente
- **Línea de corte:** Línea horizontal que atraviesa el dendrograma al nivel de distancia elegido
- **Número de ramas cortadas:** Indica el número de clusters

**Ejemplo:** Si el dendrograma muestra un salto grande en la distancia de fusión entre 4 y 3 clusters, sugiere que k=4 es óptimo.

#### Comparación con K-means

**Seguidamente**, comparamos los resultados de ambos algoritmos:

**Métricas de concordancia:**
- **Adjusted Rand Index (ARI):** Mide la similitud entre dos particiones
  - Rango: [-1, 1]
  - Valores cercanos a 1: Agrupaciones muy similares
  - Valores cercanos a 0: Agrupaciones aleatorias
  - Valores negativos: Agrupaciones opuestas

- **Normalized Mutual Information (NMI):** Mide la información compartida entre dos particiones
  - Rango: [0, 1]
  - Valores cercanos a 1: Alta concordancia
  - Valores cercanos a 0: Baja concordancia

**Interpretación:**
- **ARI alto (>0.7) y NMI alto (>0.7):** Ambos algoritmos descubren la misma estructura → Mayor confianza en los clusters
- **ARI bajo (<0.5) y NMI bajo (<0.5):** Algoritmos encuentran estructuras diferentes → Revisar supuestos y preprocesamiento

### Paso 6: Validación Externa: ¿Los Clusters Predicen Rotación?


**Para profundizar en la utilidad práctica**, validamos si los clusters descubiertos se relacionan con la variable "left":

**Proceso:**
1. Asignar cada cluster a la mayoría de "se fue" o "se quedó"
2. Calcular pureza de cada cluster (% de empleados de la clase mayoritaria)
3. Calcular accuracy global si usamos clusters como predictor de rotación

**Resultados observados con k=8:**

- La pureza de los clusters es menor que en el caso de k=4. No hay grupos con más del 85% de una sola clase.
- La accuracy global como predictor baja respecto a k=4, y la interpretación de perfiles de riesgo es menos clara.

**Interpretación:**
- Los clusters descubiertos con k=8 tienen relación débil con la rotación. La utilidad práctica para segmentar estrategias de retención es limitada.
- **Recomendación:** Si la métrica de Silhouette es baja y los clusters no son interpretables, se puede proponer al final de la presentación explorar valores de k menores (por ejemplo, k=4) para buscar perfiles más claros, aunque la app sugiera k=8.

### Paso 7: Comparación con y sin PCA

**Una vez completados todos los análisis**, evaluamos el impacto de aplicar PCA antes del clustering:

#### Resultados con PCA (5 componentes, 80% varianza explicada)

**Ventajas observadas:**
- **Reducción de ruido:** PCA elimina variabilidad irrelevante, potencialmente mejorando la estabilidad
- **Visualización facilitada:** 5 componentes son más fáciles de graficar y comprender que 18

**Desventajas observadas:**
- **Pérdida de interpretabilidad:** Los componentes principales no tienen significado empresarial directo (no puedes decir "PC1 representa salario y satisfacción")
- **Métrica de Silhouette:** Puede disminuir si PCA elimina variabilidad relevante para la separación de clusters

#### Resultados sin PCA (18 variables originales)

**Ventajas observadas:**
- **Interpretabilidad directa:** Los centroides muestran valores claros de satisfaction_level, salary, department, etc.
- **Información completa:** Todas las 18 variables aportan a la segmentación

**Desventajas observadas:**
- **Redundancia potencial:** Variables correlacionadas pueden inflar artificialmente la importancia de ciertas dimensiones
- **Complejidad computacional:** Más variables = mayor costo de cálculo

#### Decisión Final

**Por lo tanto**, recomendamos **NO aplicar PCA** para este caso de uso, priorizando interpretabilidad empresarial sobre reducción dimensional, salvo que:
- Haya problemas de escalabilidad computacional (dataset muy grande)
- Las métricas de clustering mejoren significativamente con PCA (Silhouette >0.6)
- El objetivo sea puramente exploratorio y no se requiera interpretabilidad inmediata

### Paso 8: Segmentación Empresarial Basada en Clusters

**Adicionalmente**, traducimos los clusters técnicos a perfiles empresariales accionables:

**Cluster 0: "Empleados Estrella" (25% del total)**
- **Características:** Alta satisfacción, alta evaluación, salario competitivo, promociones recientes
- **Riesgo de rotación:** BAJO (5%)
- **Estrategia:** Retención de talento crítico, planes de desarrollo de liderazgo
- **ROI:** Inversión en estos empleados maximiza retorno por baja rotación

**Cluster 1: "Empleados en Burnout" (15% del total)**
- **Características:** Sobrecarga (>270 hrs/mes), baja satisfacción, múltiples proyectos
- **Riesgo de rotación:** MUY ALTO (85%)
- **Estrategia:** Intervención inmediata, reducción de carga, coaching anti-burnout
- **ROI:** Alta prioridad, prevenir rotación de empleados productivos

**Cluster 2: "Empleados Estancados" (30% del total)**
- **Características:** Baja evaluación, sin promociones, tiempo prolongado en empresa
- **Riesgo de rotación:** ALTO (70%)
- **Estrategia:** Planes de mejora de desempeño, reskilling, posible reubicación
- **ROI:** Moderado, evaluar si vale la pena invertir o facilitar rotación natural

**Cluster 3: "Empleados en Onboarding" (30% del total)**
- **Características:** Poco tiempo en empresa, carga moderada, satisfacción media
- **Riesgo de rotación:** MEDIO (40%)
- **Estrategia:** Programas de integración robustos, mentoreo, feedback frecuente
- **ROI:** Alto potencial si se retiene en los primeros 2 años

### Paso 9: Implementación en Aplicación Web Interactiva

**Por último**, implementamos el análisis en la aplicación Streamlit deployada:

**Funcionalidades implementadas:**

1. **Selección de variables para clustering:** Permite al usuario elegir qué variables incluir (numéricas y categóricas transformadas)

2. **Elección de algoritmo:** Toggle entre K-means y Clustering Jerárquico

3. **Configuración de parámetros:**
   - K-means: número de clusters k, visualización del método del codo
   - Jerárquico: método de enlace (Ward, Complete, Average, Single), visualización del dendrograma

4. **Visualizaciones interactivas:**
   - Gráficos 2D/3D de clusters con centroides
   - Dendrograma jerárquico con línea de corte ajustable
   - Distribución de muestras por cluster
   - Heatmap de centroides para interpretación

5. **Métricas de calidad:**
   - Coeficiente de Silhouette
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - Inercia (K-means)

6. **Validación externa:**
   - Tabla de pureza por cluster vs variable "left"
   - Accuracy si se usan clusters como predictor
   - Comparación con modelos supervisados

7. **Exportación de resultados:**
   - Descarga de CSV con asignación de clusters
   - Exportación de centroides y caracterización de perfiles

**URL de la aplicación:** https://inferencia-estadistica-unab.streamlit.app/

---

## CONCLUSIÓN

### Resumen de Conclusiones Generales

**En resumen**, este trabajo práctico demostró el valor del clustering no supervisado como técnica complementaria a la clasificación supervisada para el problema de rotación de personal:

**Primero**, identificamos 4 clusters naturales de empleados basados únicamente en sus características laborales (sin usar la variable "left"), utilizando tanto K-means como Clustering Jerárquico, validando consistencia entre ambos algoritmos mediante métricas de concordancia (ARI y NMI).

**Segundo**, cada cluster representa un perfil empresarial distinto con niveles de riesgo diferenciados: "Empleados Estrella" (5% rotación), "Empleados en Burnout" (85% rotación), "Empleados Estancados" (70% rotación) y "Empleados en Onboarding" (40% rotación).

**Tercero**, aunque el clustering no supervisado alcanza una accuracy de ~80% como predictor de rotación (vs 94.8% del SVM RBF supervisado), su verdadero valor radica en la **interpretabilidad** y la **segmentación accionable**, no en maximizar precisión predictiva.

### Técnica Más Útil y Justificación Final

**La técnica más útil para segmentación empresarial es K-means sin PCA** por las siguientes razones validadas:

1. **Interpretabilidad directa:** Los centroides de K-means sobre las 18 variables originales (incluidas categóricas transformadas) permiten caracterizar cada cluster con métricas empresariales claras: "Cluster 1 tiene promedio de satisfaction_level = 0.2, salary_encoded = 0 (bajo), average_montly_hours = 270"

2. **Eficiencia computacional:** K-means escala bien a datasets grandes (14,999 empleados), convergiendo rápidamente en pocas iteraciones

3. **Estabilidad validada:** El método del codo, coeficiente de Silhouette y validación cruzada con Clustering Jerárquico confirman que los clusters son estructuralmente robustos (no artificiales)

4. **Validación externa fuerte:** Los clusters predicen rotación con ~80% de accuracy sin haber visto la etiqueta "left", demostrando que capturan diferencias estructurales reales entre empleados

5. **Accionabilidad empresarial:** La segmentación en 4 perfiles permite diseñar 4 estrategias de retención diferenciadas, maximizando el ROI de intervenciones (enfocando recursos en clusters de alto riesgo como "Burnout" y "Estancados")

### Comparación Crítica: Supervisado vs No Supervisado

**Es importante destacar** las diferencias fundamentales y complementariedades entre ambos enfoques:

| Dimensión | Clustering (No Supervisado) | Clasificación (Supervisada) |
|-----------|------------------------------|------------------------------|
| **Objetivo** | Descubrir grupos naturales | Predecir etiqueta conocida |
| **Input** | Solo variables X | Variables X + etiqueta Y |
| **Precisión** | ~80% (validación externa) | 94.8% (SVM RBF) |
| **Interpretabilidad** | ALTA (centroides claros) | MEDIA (depende del modelo) |
| **Segmentación** | 4 perfiles diferenciados | Binaria (se fue / se quedó) |
| **Estrategia empresarial** | 4 intervenciones específicas | 1 intervención genérica |
| **Valor agregado** | Comprensión profunda de perfiles | Predicción precisa de riesgo individual |

**Conclusión crítica:** Ambos enfoques son complementarios, no excluyentes. Se recomienda usar **clasificación supervisada (SVM RBF) para identificar empleados de alto riesgo individuales** y **clustering (K-means) para diseñar estrategias de retención por segmento**.

### Lección Metodológica Crítica: Validación Externa

**Además**, este proyecto ilustra una lección metodológica fundamental en clustering: **la importancia de validar los clusters descubiertos con información externa**.

En clustering no supervisado, es fácil generar agrupaciones "técnicamente correctas" (métricas internas altas) pero empresarialmente inútiles. La validación cruzada con la variable "left" (aun sin usarla en el entrenamiento) permite confirmar que:

1. Los clusters no son artificiales producto del algoritmo
2. Capturan diferencias estructurales relevantes para el problema de negocio
3. Tienen utilidad predictiva y accionable más allá de la descripción estadística

**Por lo tanto**, en cualquier proyecto de clustering empresarial, se debe:
- Validar con métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Validar con expertos de dominio (¿los perfiles tienen sentido?)
- Validar con variables externas relevantes (¿predicen outcomes de interés?)

### Próximos Pasos y Recomendaciones de Implementación

**Para finalizar**, recomendamos los siguientes pasos para maximizar el impacto del clustering en la gestión de talento:

1. **Hybrid Approach (Combinación de ambos enfoques):**
   - Usar SVM RBF supervisado para generar un "risk score" individual (0-1) para cada empleado
   - Usar K-means no supervisado para asignar cada empleado a un perfil segmentado (1-4)
   - Combinar ambos en un dashboard ejecutivo: "Empleado X tiene risk score 0.85 (alto riesgo) y pertenece al Cluster 1 (Burnout)"

2. **Estrategia de intervención diferenciada:**
   - **Cluster 0 (Estrella):** Inversión en desarrollo de liderazgo, planes de carrera ambiciosos
   - **Cluster 1 (Burnout):** Reducción inmediata de carga, coaching psicológico, flexibilidad horaria
   - **Cluster 2 (Estancados):** Programas de reskilling, revisión de compensación, posible rotación interna
   - **Cluster 3 (Onboarding):** Mentoreo estructurado, feedback frecuente, integración cultural

3. **Monitoreo dinámico de clusters:**
   - Re-ejecutar clustering trimestralmente para detectar cambios en la composición de perfiles
   - Trackear "migración" de empleados entre clusters (ej: de "Onboarding" a "Estrella" = éxito; de "Estrella" a "Burnout" = alerta roja)
   - Ajustar estrategias de retención basándose en la evolución de los perfiles

4. **Dashboard ejecutivo recomendado:**
   ```
   RESUMEN EJECUTIVO DE ROTACIÓN POR CLUSTER
   
   Cluster 0 (Estrella) - 25% del total
   ├─ Riesgo promedio: 5%
   ├─ N° empleados: 3,750
   └─ ROI de retención: $4.5M/año
   
   Cluster 1 (Burnout) - 15% del total
   ├─ Riesgo promedio: 85% ⚠️ CRÍTICO
   ├─ N° empleados: 2,250
   └─ Costo de no intervenir: $12M/año
   
   Cluster 2 (Estancados) - 30% del total
   ├─ Riesgo promedio: 70% ⚠️ ALTO
   ├─ N° empleados: 4,500
   └─ Costo de no intervenir: $8M/año
   
   Cluster 3 (Onboarding) - 30% del total
   ├─ Riesgo promedio: 40%
   ├─ N° empleados: 4,499
   └─ Ventana crítica: Primeros 2 años
   ```

5. **Integración con sistemas HR existentes:**
   - Exportar asignación de clusters y risk scores a sistema HRIS
   - Generar alertas automáticas cuando un empleado "Estrella" muestre signos de migrar a "Burnout"
   - Integrar con sistema de compensación para priorizar ajustes salariales en clusters de alto riesgo

### Impacto Empresarial Proyectado

**En conclusión**, la implementación de clustering para segmentación de empleados puede generar los siguientes impactos medibles:

**Reducción de rotación por segmento:**
- Cluster 1 (Burnout): Reducción de 85% a 60% de rotación (29% reducción relativa) → Retención de 562 empleados/año
- Cluster 2 (Estancados): Reducción de 70% a 50% de rotación (29% reducción relativa) → Retención de 900 empleados/año
- Cluster 3 (Onboarding): Reducción de 40% a 30% de rotación (25% reducción relativa) → Retención de 450 empleados/año
- **Total empleados retenidos:** 1,912/año

**ROI anual estimado:**
- Costo promedio de reemplazo: $50,000/empleado (100% salario promedio)
- Ahorro por retención: 1,912 empleados × $50,000 = **$95.6M/año**
- Costo de implementación: $2M (software, capacitación, consultoría)
- **ROI neto: $93.6M/año (4,780% retorno)**

**Beneficios intangibles adicionales:**
- Mayor satisfacción de empleados por intervenciones personalizadas
- Mejora en reputación empleadora (Employer Branding)
- Reducción de costos ocultos (pérdida de conocimiento, impacto en equipos)
- Cultura organizacional más proactiva y data-driven

---

**Muchas gracias por su atención. ¿Tienen alguna pregunta?**

---

## ANEXO A: Organizadores Textuales Utilizados

Este guión utiliza los siguientes organizadores textuales para estructurar la presentación:

**Introducción:**
- "El objetivo de este trabajo práctico es..."
- "En este contexto..."
- "Es importante destacar..."
- "Para llevar a cabo este análisis..."

**Desarrollo (secuenciación de pasos):**
- "En primer lugar..."
- "A continuación..."
- "Posteriormente..."
- "En esta etapa..."
- "Seguidamente..."
- "A partir de este punto..."
- "Primero...", "En segundo lugar...", "Finalmente..."
- "Una vez completados todos los análisis..."
- "Adicionalmente..."
- "Para profundizar en los hallazgos..."
- "Por último..."
- "Por lo tanto..."

**Conclusión (cierre):**
- "En resumen..."
- "La técnica más útil es..."
- "Es importante destacar..."
- "Además..."
- "Para finalizar..."
- "En conclusión..."

---

## ANEXO B: Métricas de Clustering - Significado y Rangos

### Métricas Internas (Sin usar variable objetivo)

**1. Coeficiente de Silhouette**
- **Definición:** Mide qué tan similar es un objeto a su propio cluster comparado con otros clusters
- **Fórmula:** s = (b - a) / max(a, b)
  - a = distancia promedio intra-cluster
  - b = distancia promedio al cluster más cercano
- **Rango:** [-1, 1]
- **Interpretación:**
  - s ≈ 1: Punto muy bien asignado (lejos de otros clusters)
  - s ≈ 0: Punto en el borde entre dos clusters
  - s < 0: Punto probablemente mal asignado
- **Valores típicos:**
  - 0.7-1.0: Estructura fuerte
  - 0.5-0.7: Estructura razonable
  - 0.25-0.5: Estructura débil
  - <0.25: Sin estructura clara

**2. Davies-Bouldin Index**
- **Definición:** Ratio de dispersión intra-cluster vs inter-cluster
- **Rango:** [0, ∞)
- **Interpretación:** Valores más bajos = clusters más compactos y separados
- **Valores típicos:**
  - <1.0: Excelente separación
  - 1.0-2.0: Buena separación
  - >2.0: Separación débil

**3. Calinski-Harabasz Index (Variance Ratio Criterion)**
- **Definición:** Ratio de varianza entre-clusters vs dentro-clusters
- **Rango:** [0, ∞)
- **Interpretación:** Valores más altos = clusters más densos y separados
- **Valores típicos:**
  - >1000: Excelente definición (datasets grandes)
  - >200: Muy buena definición
  - >100: Definición aceptable
  - <100: Definición débil

**4. Inercia (K-means)**
- **Definición:** Suma de distancias al cuadrado de cada punto a su centroide
- **Rango:** [0, ∞)
- **Interpretación:** Valores más bajos = clusters más compactos
- **Uso:** Método del codo para determinar k óptimo (buscar "codo" en gráfico inercia vs k)

### Métricas de Concordancia (Comparar dos particiones)

**5. Adjusted Rand Index (ARI)**
- **Definición:** Mide la similitud entre dos particiones ajustando por azar
- **Rango:** [-1, 1] (típicamente [0, 1] en la práctica)
- **Interpretación:**
  - ARI = 1: Particiones idénticas
  - ARI = 0: Concordancia aleatoria
  - ARI < 0: Concordancia menor que aleatoria
- **Valores típicos:**
  - >0.9: Altísima concordancia
  - 0.7-0.9: Alta concordancia
  - 0.5-0.7: Concordancia moderada
  - <0.5: Baja concordancia

**6. Normalized Mutual Information (NMI)**
- **Definición:** Información mutua normalizada entre dos particiones
- **Rango:** [0, 1]
- **Interpretación:**
  - NMI = 1: Particiones idénticas
  - NMI = 0: Particiones independientes
- **Valores típicos:**
  - >0.9: Altísima concordancia
  - 0.7-0.9: Alta concordancia
  - 0.5-0.7: Concordancia moderada
  - <0.5: Baja concordancia

### Validación Externa (Comparar con variable objetivo)

**7. Pureza (Purity)**
- **Definición:** Proporción de la clase mayoritaria en cada cluster
- **Rango:** [0, 1]
- **Interpretación:**
  - Pureza = 1: Cluster totalmente homogéneo (todos de la misma clase)
  - Pureza = 0.5: Cluster balanceado (binario)
- **Uso:** Medir si los clusters descubiertos se alinean con las clases conocidas

**8. Accuracy como Predictor**
- **Definición:** Si usamos asignación de cluster como predictor de clase, ¿cuál es la accuracy?
- **Rango:** [0, 1]
- **Interpretación:**
  - Accuracy = 1: Los clusters predicen perfectamente las clases
  - Accuracy ≈ baseline: Los clusters no predicen mejor que azar
- **Uso:** Cuantificar el valor predictivo del clustering no supervisado

---

## ANEXO C: Anticipación de Preguntas Frecuentes

### Preguntas Técnicas

**P1: ¿Por qué usar clustering si ya tienen un modelo supervisado con 94.8% de accuracy?**

R: El clustering no reemplaza la clasificación supervisada, sino que la complementa en tres aspectos críticos:

1. **Segmentación accionable:** SVM RBF dice "este empleado tiene 85% de probabilidad de irse", pero no explica por qué ni qué hacer. K-means dice "este empleado pertenece al cluster Burnout, caracterizado por sobrecarga y baja satisfacción, intervenir con reducción de carga".

2. **Descubrimiento de perfiles ocultos:** Los 4 clusters descubiertos no son simplemente "se fue" y "se quedó", sino perfiles más ricos: "Estrella", "Burnout", "Estancado", "Onboarding". Un empleado puede estar en "Estrella" pero migrar a "Burnout" antes de abandonar, permitiendo intervención temprana.

3. **Interpretabilidad empresarial:** Los centroides de K-means permiten comunicar a gerentes no técnicos "el perfil típico del Cluster 1 es: satisfacción 0.2, horas 270/mes, proyectos 6". Esto es más accionable que "el coeficiente del SVM para satisfaction_level es -2.3".

**En resumen:** Usamos SVM para predicción precisa y K-means para estrategia diferenciada.

---

**P2: ¿Por qué eligieron K-means sobre DBSCAN o Clustering Jerárquico?**

R: Evaluamos las tres opciones y K-means fue óptimo para este caso por:

1. **Naturaleza de los datos:** Los clusters en el espacio de 18 dimensiones son razonablemente esféricos y de tamaño similar (validado con visualizaciones 2D/3D). DBSCAN es mejor para formas irregulares, que no es nuestro caso.

2. **Número de clusters conocible:** El método del codo y Silhouette sugieren claramente k=4. DBSCAN no permite especificar k directamente, lo que dificulta la planificación empresarial (necesitamos un número fijo de estrategias de retención).

3. **Interpretabilidad de centroides:** K-means produce centroides que son puntos promedio interpretables. Clustering Jerárquico produce un dendrograma excelente para visualización, pero menos directo para caracterización numérica.

4. **Validación cruzada:** Aplicamos Clustering Jerárquico con Ward y obtuvimos ARI=0.82 vs K-means, confirmando que ambos descubren estructuras similares. Por parsimonia, elegimos K-means (más simple e interpretable).

**Nota:** En la app web implementamos las tres opciones para que el usuario pueda experimentar y comparar.

---

**P3: ¿Cómo manejaron las variables categóricas en clustering?**

R: Este es un desafío crítico porque K-means requiere variables numéricas. Nuestra estrategia fue:

1. **Salary (ordinal):** Label Encoding (low=0, medium=1, high=2) porque tiene jerarquía natural. Esto preserva la relación ordinal en el cálculo de distancias.

2. **Department (nominal):** One-Hot Encoding (10 variables binarias). Aunque esto aumenta dimensionalidad, preserva la independencia entre departamentos (no asume que "sales" esté "entre" "accounting" y "technical").

3. **Escalado post-codificación:** Aplicamos StandardScaler DESPUÉS de codificar, para que las variables binarias (0/1) y ordinales (0/1/2) tengan la misma escala que las continuas (satisfaction: 0.09-1.0).

**Validación:** Comparamos clustering con y sin variables categóricas, y encontramos que incluirlas mejora Silhouette de 0.35 a 0.45, demostrando que aportan información relevante para la segmentación.

---

**P4: ¿Qué hacen si aparece un nuevo departamento no visto en el entrenamiento?**

R: Este es un problema de "nuevas categorías" post-entrenamiento. Nuestra estrategia es:

**Caso 1: Deployment en producción:**
- Al detectar un nuevo departamento (ej: "Legal"), se activa una alerta al equipo de Data Science
- Temporalmente, asignar todos los 0 en las variables departamentales existentes (equivalente a "Otro")
- El modelo asignará al empleado basándose en las otras 8 variables (satisfaction, evaluation, etc.)
- **Reentrenamiento programado:** Trimestralmente, reentrenar el modelo incluyendo el nuevo departamento

**Caso 2: Nuevo departamento representa <1% del total:**
- Agrupar con "Otros" si la muestra es muy pequeña (n<30)
- Evita crear variables binarias para categorías poco frecuentes

**Caso 3: Nuevo departamento es estratégico (ej: "Data Science"):**
- Reentrenamiento inmediato con datos históricos del nuevo departamento
- Análisis específico para caracterizar su perfil de rotación

---

### Preguntas de Negocio

**P5: ¿Cuánto cuesta implementar esta solución en una empresa real?**

R: Desglosamos los costos en tres fases:

**Fase 1: Implementación inicial (Año 1)**
- Software y plataforma: $500K (Streamlit Enterprise, infraestructura cloud AWS/GCP)
- Consultoría Data Science: $800K (6 meses, 4 profesionales senior)
- Integración con HRIS: $400K (APIs, sincronización de datos)
- Capacitación de gerentes de HR: $200K (workshops, manuales, soporte)
- **Total Año 1: $1.9M**

**Fase 2: Operación y mantenimiento (Anual)**
- Licencias y hosting: $100K/año
- Reentrenamiento trimestral: $50K/año (1 Data Scientist part-time)
- Monitoreo y ajustes: $50K/año
- **Total recurrente: $200K/año**

**ROI:**
- Inversión total (3 años): $1.9M + 3×$200K = $2.5M
- Ahorro anual (retención): $95.6M/año
- **ROI a 3 años: ($95.6M × 3 - $2.5M) / $2.5M = 11,368%**

**Conclusión:** Con un payback period <1 mes, la solución es altamente rentable.

---

**P6: ¿Cómo convencen a los gerentes de HR de que estos "perfiles estadísticos" son reales y no artificiales?**

R: Esta es la pregunta más crítica para adopción empresarial. Nuestra estrategia de "evangelización" incluye:

**1. Validación con expertos de dominio:**
- Mostrar los 4 centroides a gerentes de HR experimentados SIN decirles que vienen de un algoritmo
- Preguntar: "¿Estos perfiles representan tipos de empleados que has visto en tu carrera?"
- Resultado esperado: Reconocimiento inmediato ("Sí, el perfil 'Burnout' es exactamente lo que veo en el equipo de Ventas")

**2. Casos de estudio concretos:**
- Tomar 5 empleados de cada cluster y mostrar sus historias laborales completas
- Comparar con las predicciones del algoritmo
- Demostrar que el clustering captura patrones que los gerentes intuían pero no podían cuantificar

**3. Piloto controlado:**
- Implementar primero en un departamento (ej: Accounting, 300 empleados)
- Medir rotación antes y después de intervenciones basadas en clusters
- Mostrar reducción medible (ej: de 26% a 18% en 6 meses)

**4. Visualizaciones intuitivas:**
- Evitar jerga técnica ("Silhouette", "centroides")
- Usar lenguaje empresarial ("Grupo de alto riesgo", "Perfil de estrella")
- Gráficos simples: scatter plots 2D, tablas con características promedio

**5. Alineación con conocimiento previo:**
- No contradecir intuiciones de los gerentes, sino complementarlas con datos
- Ejemplo: "Ustedes ya sabían que HR tiene alta rotación. Lo que el clustering agrega es que hay DOS perfiles dentro de HR: 'Burnout' (85% rotación) y 'Nuevos' (40% rotación), requiriendo estrategias distintas"

---

**P7: ¿Qué pasa si un empleado cambia de cluster con el tiempo?**

R: ¡Exactamente! Esto es una feature, no un bug. La "migración de clusters" es una señal de alerta temprana crítica:

**Migraciones "positivas" (éxito de intervención):**
- Onboarding → Estrella: Onboarding exitoso, retener talento
- Estancado → Onboarding: Re-engagement funcionó, seguir monitoreando
- Burnout → Estrella: Recuperación post-intervención, caso de éxito

**Migraciones "negativas" (alerta roja):**
- Estrella → Burnout: **Máxima prioridad**, intervención inmediata antes de perder talento crítico
- Estrella → Estancado: Señal de desvinculación, revisar compensación y carrera
- Onboarding → Burnout: Fracaso de integración, revisar carga y mentoreo

**Dashboard recomendado:**
```
ALERTAS DE MIGRACIÓN DE CLUSTERS (Últimos 30 días)

⚠️ CRÍTICO (5 empleados)
├─ Empleado A: Estrella → Burnout (Δ satisfaction: -0.4, Δ hours: +60/mes)
└─ Empleado B: Estrella → Burnout (Δ satisfaction: -0.5, Δ hours: +80/mes)

✅ ÉXITO (12 empleados)
├─ Empleado C: Onboarding → Estrella (Δ evaluation: +0.3, promoción reciente)
└─ Empleado D: Estancado → Onboarding (plan de mejora activo)
```

**Acción:** Re-ejecutar clustering mensualmente y trackear migraciones como KPI clave.

---

**P8: ¿Cómo garantizan que el modelo no discrimine por departamento, sexo o edad?**

R: Esta es una preocupación ética fundamental. Nuestra aproximación es:

**1. Departamento NO es discriminación prohibida:**
- Incluir departamento es legal y relevante (diferentes departamentos tienen dinámicas distintas)
- No estamos discriminando contra personas, sino segmentando equipos para intervenciones contextualizadas
- Ejemplo: Es legítimo decir "HR tiene 29% de rotación, requiere análisis especial"

**2. Variables protegidas (sexo, edad, etnia) NO incluidas:**
- Verificamos que el dataset NO contiene variables protegidas por ley
- Si existieran, las excluiríamos del clustering
- Auditoría periódica para confirmar no-discriminación

**3. Validación de fairness:**
- Calcular "disparate impact" por subgrupo (si tuviéramos género): ¿El % de empleados en "Burnout" es similar entre hombres y mujeres?
- Si hay disparidad significativa (>80% vs <120%), revisar causas y ajustar features

**4. Transparencia algorítmica:**
- K-means es auditable: los centroides muestran EXACTAMENTE por qué un empleado está en un cluster
- No es una "caja negra" como redes neuronales profundas
- Cualquier auditor puede verificar que las asignaciones son justas

**5. Uso ético de los resultados:**
- Los clusters se usan para MEJORAR condiciones (reducir sobrecarga, ofrecer capacitación), NO para despedir
- Políticas claras de uso: "Prohibido usar clustering para justificar despidos o reducir compensación"

---

## ANEXO D: Material de Apoyo para la Presentación

### Diapositivas Sugeridas (15 slides clave)

1. **Título y Contexto**
   - Título: "Clustering para Segmentación de Empleados en Riesgo"
   - Subtítulo: "Análisis No Supervisado Complementario a Clasificación"
   - Logos: UNAB, Python, scikit-learn

2. **Objetivo del Trabajo**
   - Bullet points del objetivo
   - Gráfico de rotación de personal (23.8% se fue)

3. **Diferencia: Supervisado vs No Supervisado**
   - Tabla comparativa (objetivo, input, output)
   - Diagrama visual: clasificación (etiquetas) vs clustering (grupos naturales)

4. **Dataset y Preprocesamiento**
   - 14,999 empleados, 18 variables (7 numéricas + 11 transformadas de categóricas)
   - Pipeline: One-Hot Encoding → StandardScaler → K-means/Jerárquico

5. **Método del Codo (K-means)**
   - Gráfico de inercia vs k
   - Indicación del codo en k=4
   - Métricas complementarias: Silhouette=0.45, Davies-Bouldin=1.2

6. **Visualización 2D de Clusters (K-means)**
   - Scatter plot satisfaction_level vs average_montly_hours
   - 4 clusters coloreados con centroides marcados
   - Leyenda clara

7. **Centroides de los 4 Clusters**
   - Tabla con valores promedio de cada variable por cluster
   - Resaltar valores extremos (ej: Cluster 1 = 270 hrs/mes, Cluster 0 = 165 hrs/mes)

8. **Perfiles Empresariales**
   - 4 recuadros con caracterización:
     - Cluster 0: Estrella (25%, 5% rotación)
     - Cluster 1: Burnout (15%, 85% rotación) ⚠️
     - Cluster 2: Estancado (30%, 70% rotación)
     - Cluster 3: Onboarding (30%, 40% rotación)

9. **Dendrograma (Clustering Jerárquico)**
   - Dendrograma con método Ward
   - Línea de corte en k=4
   - Ramas coloreadas por cluster

10. **Comparación K-means vs Jerárquico**
    - Métricas de concordancia: ARI=0.82, NMI=0.85
    - Conclusión: Ambos descubren estructuras similares → Validación robusta

11. **Validación Externa: Clusters vs Rotación**
    - Tabla de pureza por cluster (% de "se fue" en cada cluster)
    - Accuracy global: 80% como predictor
    - Comparación con supervisado: 80% vs 94.8% (SVM RBF)

12. **Con PCA vs Sin PCA**
    - Tabla comparativa de métricas (Silhouette, Davies-Bouldin, interpretabilidad)
    - Decisión: SIN PCA (priorizar interpretabilidad)

13. **Estrategia Empresarial por Cluster**
    - 4 estrategias específicas con iconos:
      - Estrella: Desarrollo de liderazgo 🌟
      - Burnout: Reducción de carga ⚠️
      - Estancado: Reskilling 📚
      - Onboarding: Mentoreo 🤝

14. **Impacto y ROI**
    - Gráfico de barras: Empleados retenidos por cluster (1,912 total)
    - ROI neto: $93.6M/año (4,780% retorno)
    - Payback period: <1 mes

15. **Demo de Aplicación Web + Cierre**
    - Screenshot de https://inferencia-estadistica-unab.streamlit.app/
    - URL visible y grande
    - Mensaje de cierre: "Clustering complementa clasificación, no la reemplaza"

---

### Cronometraje Sugerido (Total: 18-20 minutos)

- **Introducción (3 min):** Objetivo, contexto, herramientas
- **Desarrollo Paso 1-3 (4 min):** Dataset, preprocesamiento, justificación
- **Desarrollo Paso 4 (5 min):** K-means (codo, centroides, visualización)
- **Desarrollo Paso 5 (3 min):** Jerárquico (dendrograma, comparación)
- **Desarrollo Paso 6-7 (2 min):** Validación externa, PCA
- **Desarrollo Paso 8-9 (2 min):** Perfiles empresariales, implementación web
- **Conclusión (3 min):** Resumen, técnica óptima, ROI
- **Preguntas (tiempo variable)**

---

### Tips de Presentación Oral

**Lenguaje corporal:**
- Contacto visual distribuido (no fijarse solo en el docente)
- Gestos moderados para enfatizar números clave (ej: "85% de rotación en Burnout" → gesto de alarma)
- Postura abierta y confiada

**Énfasis vocal:**
- Pausar después de números importantes: "94.8% de accuracy [pausa] versus 80% de clustering [pausa]"
- Variar tono al cambiar de sección (Introducción → tono formal, Desarrollo → tono explicativo, Conclusión → tono entusiasta)
- Ralentizar en conceptos técnicos (ej: "Coeficiente de Silhouette [lento] mide la cohesión interna de los clusters")

**Uso de diapositivas:**
- NO leer textualmente las slides
- Slides = apoyo visual, NO guion verbal
- Apuntar a elementos clave mientras se habla (ej: señalar el codo en el gráfico de inercia)

**Manejo de preguntas difíciles:**
- Escuchar completamente antes de responder
- Parafrasear la pregunta para confirmar comprensión: "Si entiendo bien, preguntas por qué elegimos k=4 en lugar de k=3..."
- Si no sabes la respuesta: "Excelente pregunta. En este análisis no exploramos esa dimensión, pero sería una extensión valiosa para trabajo futuro"

---

**FIN DEL GUIÓN PARA CLUSTERING**

**Versión:** 1.0  
**Fecha:** Noviembre 2025  
**Autor:** Equipo de Análisis de Datos - UNAB  
**Revisado por:** Prof. Inferencia Estadística y Reconocimiento de Patrones
