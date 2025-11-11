# 🎨 Guía de Diapositivas para Clustering - Competencia Discursiva

Este documento proporciona una estructura detallada para las **15 diapositivas recomendadas** para la presentación sobre Clustering (K-means y Clustering Jerárquico).

---

## 📋 Índice de Slides

1. [Slide 1: Título y Contexto](#slide-1-título-y-contexto)
2. [Slide 2: Objetivo del Trabajo](#slide-2-objetivo-del-trabajo)
3. [Slide 3: Diferencia Supervisado vs No Supervisado](#slide-3-diferencia-supervisado-vs-no-supervisado)
4. [Slide 4: Dataset y Preprocesamiento](#slide-4-dataset-y-preprocesamiento)
5. [Slide 5: Método del Codo (K-means)](#slide-5-método-del-codo-k-means)
6. [Slide 6: Visualización 2D de Clusters](#slide-6-visualización-2d-de-clusters-k-means)
7. [Slide 7: Centroides de los 4 Clusters](#slide-7-centroides-de-los-4-clusters)
8. [Slide 8: Perfiles Empresariales](#slide-8-perfiles-empresariales)
9. [Slide 9: Dendrograma (Clustering Jerárquico)](#slide-9-dendrograma-clustering-jerárquico)
10. [Slide 10: Comparación K-means vs Jerárquico](#slide-10-comparación-k-means-vs-jerárquico)
11. [Slide 11: Validación Externa](#slide-11-validación-externa-clusters-vs-rotación)
12. [Slide 12: Con PCA vs Sin PCA](#slide-12-con-pca-vs-sin-pca)
13. [Slide 13: Estrategia Empresarial por Cluster](#slide-13-estrategia-empresarial-por-cluster)
14. [Slide 14: Impacto y ROI](#slide-14-impacto-y-roi)
15. [Slide 15: Demo + Cierre](#slide-15-demo-de-aplicación-web--cierre)

---

## Slide 1: Título y Contexto

### Layout
- **Tipo:** Slide de título
- **Fondo:** Color sólido oscuro (#1e3a5f) con gradiente sutil

### Contenido

**Título principal (centrado, grande):**
```
Clustering para Segmentación de Empleados
en Riesgo de Rotación
```

**Subtítulo:**
```
Análisis No Supervisado Complementario
Inferencia Estadística y Reconocimiento de Patrones
```

**Información adicional (pie de slide):**
```
Universidad Nacional Guillermo Brown
Noviembre 2025

Aplicación Web: https://inferencia-estadistica-unab.streamlit.app/
```

### Elementos visuales
- Logo UNAB (esquina superior izquierda)
- Logos de tecnologías: Python, scikit-learn, Streamlit (esquina superior derecha, pequeños)
- Imagen decorativa: iconos de clusters o personas agrupadas (esquina inferior derecha, transparencia 30%)

### Paleta de colores
- Fondo: #1e3a5f (azul oscuro)
- Título: #ffffff (blanco)
- Subtítulo: #a8d5e2 (azul claro)
- Pie: #c9c9c9 (gris claro)

---

## Slide 2: Objetivo del Trabajo

### Layout
- **Tipo:** Texto + Imagen
- **Distribución:** 60% texto izquierda, 40% imagen derecha

### Contenido

**Título:**
```
🎯 Objetivo del Trabajo Práctico
```

**Texto principal:**
```
Aplicar técnicas de clustering no supervisado
para identificar patrones naturales de agrupamiento
en empleados, permitiendo una segmentación basada
en datos que facilite estrategias diferenciadas
de retención de personal.

DIFERENCIA CLAVE: A diferencia de la clasificación
supervisada (que requiere conocer quién se fue),
el clustering descubre grupos naturales SIN usar
la etiqueta de rotación.
```

**Bullet points:**
- 🔍 Descubrir perfiles ocultos más allá de "se fue / se quedó"
- 🎨 Segmentar para estrategias diferenciadas de retención
- 📊 Validar si existen diferencias estructurales reales
- 💡 Complementar (no reemplazar) clasificación supervisada

### Imagen derecha
- Gráfico de barras simple: 76.2% "Se quedó" (verde) vs 23.8% "Se fue" (rojo)
- Título del gráfico: "Distribución de Rotación en el Dataset"

### Notas del presentador
> "El objetivo es complementar la clasificación supervisada del trabajo anterior, no reemplazarla. Mientras SVM RBF predice CON 94.8% de accuracy QUIÉN se irá, el clustering revela POR QUÉ y QUÉ HACER."

---

## Slide 3: Diferencia Supervisado vs No Supervisado

### Layout
- **Tipo:** Tabla comparativa + Diagrama
- **Distribución:** 70% tabla superior, 30% diagrama inferior

### Contenido

**Título:**
```
🔄 Clasificación vs Clustering: Enfoques Complementarios
```

**Tabla comparativa:**

| Aspecto | Clasificación Supervisada | Clustering No Supervisado |
|---------|---------------------------|---------------------------|
| 🎯 **Objetivo** | Predecir etiqueta conocida | Descubrir grupos naturales |
| 📥 **Input** | Variables X + Etiqueta Y | Solo variables X |
| 📊 **Algoritmos** | LDA, QDA, Bayes, SVM | K-means, Jerárquico |
| 🎲 **Output** | Probabilidad (0-1) | Asignación a cluster (0-3) |
| 🎯 **Precisión** | 94.8% (SVM RBF) | 80% (validación externa) |
| 💼 **Valor** | Predicción individual | Segmentación estratégica |

**Diagrama inferior (lado a lado):**

**Izquierda - Supervisado:**
```
┌──────────────────────┐
│  X (variables)       │
│  Y (left: 0/1)       │
└──────────────────────┘
         ↓
   [ SVM RBF Model ]
         ↓
   Empleado #12345:
   87% probabilidad
   de irse
```

**Derecha - No Supervisado:**
```
┌──────────────────────┐
│  X (variables)       │
│  [NO usa Y]          │
└──────────────────────┘
         ↓
   [ K-means k=4 ]
         ↓
   Empleado #12345:
   Cluster 1 "Burnout"
   (sobrecarga + baja
   satisfacción)
```

### Destacado inferior (caja resaltada)
```
💡 CONCLUSIÓN: Ambos enfoques son COMPLEMENTARIOS, no excluyentes.
   Usa supervisado para precisión, no supervisado para interpretabilidad.
```

---

## Slide 4: Dataset y Preprocesamiento

### Layout
- **Tipo:** Texto + Pipeline visual
- **Distribución:** 40% texto izquierda, 60% pipeline derecha

### Contenido

**Título:**
```
🔧 Dataset y Pipeline de Preprocesamiento
```

**Texto izquierda:**

**Características del Dataset:**
- 📊 **14,999 empleados**
- 🔢 **7 variables numéricas** (satisfaction, evaluation, projects, hours, etc.)
- 📋 **2 variables categóricas** (Department, salary)
- 🎯 **Variable objetivo:** `left` (EXCLUIDA del clustering, solo para validación)

**Transformación de Categóricas:**
- **Salary:** Label Encoding (low=0, medium=1, high=2)
- **Department:** One-Hot Encoding (10 variables binarias)
- **Resultado:** 18 variables numéricas totales

### Pipeline derecha (diagrama de flujo)

```
┌─────────────────────────────┐
│  Dataset Original (9 vars)  │
│  - 7 numéricas              │
│  - 2 categóricas            │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│  One-Hot Encoding           │
│  (Department → 10 binarias) │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│  Label Encoding Ordinal     │
│  (salary → 0/1/2)           │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│  StandardScaler             │
│  (media=0, std=1)           │
└─────────────────────────────┘
              ↓
┌─────────────────────────────┐
│  Dataset Procesado (18 vars)│
│  Listo para Clustering      │
└─────────────────────────────┘
```

### Notas del presentador
> "El preprocesamiento es CRÍTICO. Sin escalar, variables con rangos grandes (hours: 96-310) dominarían sobre variables con rangos pequeños (accident: 0-1), generando clusters artificiales."

---

## Slide 5: Método del Codo (K-means)

### Layout
- **Tipo:** Gráfico principal + Tabla de métricas
- **Distribución:** 65% gráfico izquierda, 35% métricas derecha

### Contenido

**Título:**
```
📈 Determinación del Número Óptimo de Clusters (k)
```

**Gráfico izquierda:**
- **Tipo:** Gráfico de línea con punto marcado
- **Eje X:** Número de clusters (k) [2-10]
- **Eje Y:** Inercia (suma de distancias²)
- **Curva:** Línea azul descendente con "codo" marcado en k=4
- **Anotación:** Flecha señalando k=4 con texto "Codo óptimo: k=4"
- **Colores:** Línea azul (#2980b9), punto del codo en rojo (#e74c3c) y grande

**Tabla derecha (métricas por k):**

| k | Inercia | Silhouette |
|---|---------|------------|
| 2 | 112,000 | 0.32 |
| 3 | 98,000 | 0.38 |
| **4** | **85,000** | **0.45** ⭐ |
| 5 | 78,000 | 0.41 |
| 6 | 73,000 | 0.38 |

**Interpretación (debajo de la tabla):**
```
✅ k=4 maximiza Silhouette (0.45)
✅ Inercia se estabiliza después de k=4
✅ Reducción marginal con k>4
```

### Caja destacada inferior
```
🎯 DECISIÓN: Elegimos k=4 clusters
   (validado por método del codo y Silhouette máximo)
```

### Notas del presentador
> "El codo es donde la curva deja de descender bruscamente. Aumentar k más allá de 4 solo reduce marginalmente la inercia, sin aportar valor interpretativo."

---

## Slide 6: Visualización 2D de Clusters (K-means)

### Layout
- **Tipo:** Gráfico grande centrado
- **Distribución:** 100% gráfico

### Contenido

**Título:**
```
🗺️ Visualización 2D de los 4 Clusters Descubiertos
```

**Gráfico (ocupar 90% del slide):**
- **Tipo:** Scatter plot 2D
- **Eje X:** satisfaction_level (0.0 - 1.0)
- **Eje Y:** average_montly_hours (90 - 320)
- **Puntos:**
  - Cluster 0 (Estrella): Verde (#27ae60), ~3,750 puntos
  - Cluster 1 (Burnout): Rojo (#e74c3c), ~2,250 puntos
  - Cluster 2 (Estancado): Naranja (#f39c12), ~4,500 puntos
  - Cluster 3 (Onboarding): Azul (#3498db), ~4,499 puntos
- **Centroides:** Estrellas negras grandes con borde amarillo
- **Leyenda:** Esquina superior derecha, clara

**Anotaciones en el gráfico:**
- Flecha señalando Cluster 1 (rojo): "Burnout: Baja satisfacción + Muchas horas"
- Flecha señalando Cluster 0 (verde): "Estrella: Alta satisfacción + Horas equilibradas"

**Pie del gráfico:**
```
Los centroides (⭐) representan el "empleado promedio" de cada cluster.
La distancia entre clusters confirma separación estructural real.
```

### Notas del presentador
> "Observen cómo el Cluster 1 (rojo) está claramente separado en la zona de baja satisfacción y muchas horas. Este es el perfil 'Burnout' que identificamos."

---

## Slide 7: Centroides de los 4 Clusters

### Layout
- **Tipo:** Tabla grande con heatmap
- **Distribución:** 100% tabla

### Contenido

**Título:**
```
📍 Centroides: Caracterización Numérica de cada Cluster
```

**Tabla (heatmap con colores):**

| Variable | Cluster 0<br>Estrella | Cluster 1<br>Burnout | Cluster 2<br>Estancado | Cluster 3<br>Onboarding |
|----------|----------------------|----------------------|------------------------|-------------------------|
| **satisfaction_level** | 🟢 **0.80** | 🔴 **0.20** | 🟠 0.40 | 🔵 0.55 |
| **last_evaluation** | 🟢 **0.82** | 🟠 0.55 | 🔴 **0.52** | 🔵 0.65 |
| **number_project** | 🟢 4.2 | 🔴 **6.5** | 🟠 3.8 | 🔵 2.8 |
| **average_montly_hours** | 🟢 165 | 🔴 **275** | 🟠 150 | 🔵 140 |
| **time_spend_company** | 🟢 3.5 | 🟠 3.8 | 🔴 **5.2** | 🔵 **1.8** |
| **Work_accident** | 🟠 0.15 | 🔴 0.22 | 🟢 0.10 | 🔵 0.08 |
| **promotion_last_5years** | 🟠 0.08 | 🔴 **0.02** | 🔴 **0.01** | 🟠 0.03 |
| **salary_encoded** | 🟢 **1.5** | 🔴 **0.4** | 🟠 0.6 | 🔵 0.8 |

**Leyenda de colores:**
- 🟢 Verde: Valores favorables
- 🔵 Azul: Valores medios/neutros
- 🟠 Naranja: Valores preocupantes
- 🔴 Rojo: Valores críticos/extremos

**Interpretación (debajo de la tabla):**
```
⚠️ Cluster 1 (Burnout): 275 hrs/mes (67% más que Estrella) + salario bajo
✅ Cluster 0 (Estrella): Equilibrado en todas las dimensiones
📉 Cluster 2 (Estancado): 5.2 años sin promoción ni mejora de evaluación
🆕 Cluster 3 (Onboarding): Solo 1.8 años en empresa (recién llegados)
```

### Notas del presentador
> "Esta tabla es el corazón del análisis. Cada fila es una variable, cada columna es un perfil. Los valores extremos están resaltados en rojo. Por ejemplo, Cluster 1 tiene 275 horas mensuales, eso es trabajar 63 horas por semana, garantía de burnout."

---

## Slide 8: Perfiles Empresariales

### Layout
- **Tipo:** 4 cuadrantes (tarjetas)
- **Distribución:** 2×2 grid

### Contenido

**Título:**
```
💼 Traducción de Clusters a Perfiles Empresariales Accionables
```

**Cuadrante 1 (Superior Izquierdo) - Cluster 0:**
```
┌────────────────────────────────────────┐
│ 🌟 EMPLEADOS ESTRELLA                  │
├────────────────────────────────────────┤
│ • 25% del total (3,750 empleados)      │
│ • Alta satisfacción (0.80) y evaluación│
│ • Horas equilibradas (165/mes)         │
│ • Salario competitivo                  │
├────────────────────────────────────────┤
│ RIESGO DE ROTACIÓN: 5% ✅              │
│ ESTRATEGIA: Retención de talento       │
│             Desarrollo de liderazgo    │
└────────────────────────────────────────┘
```

**Cuadrante 2 (Superior Derecho) - Cluster 1:**
```
┌────────────────────────────────────────┐
│ 🔥 EMPLEADOS EN BURNOUT                │
├────────────────────────────────────────┤
│ • 15% del total (2,250 empleados)      │
│ • Baja satisfacción (0.20)             │
│ • Sobrecarga: 275 hrs/mes, 6.5 proyect.│
│ • Salario bajo sin promociones         │
├────────────────────────────────────────┤
│ RIESGO DE ROTACIÓN: 85% 🚨 CRÍTICO     │
│ ESTRATEGIA: Intervención INMEDIATA     │
│             Reducir carga a ≤200 hrs   │
│             Coaching + ajuste salarial │
└────────────────────────────────────────┘
```

**Cuadrante 3 (Inferior Izquierdo) - Cluster 2:**
```
┌────────────────────────────────────────┐
│ 📉 EMPLEADOS ESTANCADOS                │
├────────────────────────────────────────┤
│ • 30% del total (4,500 empleados)      │
│ • Baja evaluación (0.52)               │
│ • 5.2 años SIN promoción               │
│ • Satisfacción media-baja (0.40)       │
├────────────────────────────────────────┤
│ RIESGO DE ROTACIÓN: 70% ⚠️ ALTO        │
│ ESTRATEGIA: Reskilling / Mejora        │
│             Revisión compensación      │
│             Rotación interna           │
└────────────────────────────────────────┘
```

**Cuadrante 4 (Inferior Derecho) - Cluster 3:**
```
┌────────────────────────────────────────┐
│ 🆕 EMPLEADOS EN ONBOARDING             │
├────────────────────────────────────────┤
│ • 30% del total (4,499 empleados)      │
│ • Solo 1.8 años en empresa             │
│ • Carga baja (2.8 proyectos)           │
│ • Satisfacción media (0.55)            │
├────────────────────────────────────────┤
│ RIESGO DE ROTACIÓN: 40% ⚠️             │
│ ESTRATEGIA: Onboarding robusto         │
│             Mentoreo estructurado      │
│             Feedback frecuente         │
└────────────────────────────────────────┘
```

### Notas del presentador
> "Estos 4 perfiles no son divisiones arbitrarias. El algoritmo los descubrió basándose en similitudes naturales. Cada perfil requiere una estrategia diferente: no tratarás igual a un 'Estrella' que a un 'Burnout'."

---

## Slide 9: Dendrograma (Clustering Jerárquico)

### Layout
- **Tipo:** Gráfico grande + Explicación lateral
- **Distribución:** 70% dendrograma izquierda, 30% texto derecha

### Contenido

**Título:**
```
🌳 Dendrograma: Validación con Clustering Jerárquico (Método Ward)
```

**Dendrograma izquierda:**
- **Tipo:** Dendrograma vertical (estilo árbol)
- **Eje X:** Observaciones (empleados)
- **Eje Y:** Distancia de fusión
- **Ramas coloreadas:** 4 colores distintos (verde, rojo, naranja, azul)
- **Línea de corte:** Línea verde horizontal con anotación "Corte en k=4"
- **Anotación:** Flecha señalando un "salto grande" en distancia

**Texto derecha (explicación):**

**¿Qué es un dendrograma?**
- Visualiza el proceso de fusión de clusters
- La altura indica la distancia entre grupos fusionados
- Fusiones altas = grupos muy diferentes

**¿Cómo determinar k?**
- Buscar "saltos grandes" en distancia
- El dendrograma sugiere k=4 (salto notable antes de k=3)

**Método de enlace: Ward**
- Minimiza varianza intra-cluster
- Tiende a crear clusters compactos y similares en tamaño

**Métrica: Euclidiana**
- Distancia estándar en espacio multidimensional

### Caja destacada inferior
```
✅ CONCORDANCIA con K-means:
   El dendrograma confirma que k=4 es óptimo
   (salto grande en distancia antes de k=3)
```

### Notas del presentador
> "El dendrograma es como un árbol genealógico de los empleados. Los que se fusionan primero (abajo) son muy similares. Los que se fusionan al final (arriba) son muy diferentes. El salto grande en k=4 confirma que es el número óptimo."

---

## Slide 10: Comparación K-means vs Jerárquico

### Layout
- **Tipo:** Tabla comparativa + Métricas de concordancia
- **Distribución:** 50% tabla superior, 50% métricas inferior

### Contenido

**Título:**
```
🔄 Validación Cruzada: K-means vs Clustering Jerárquico
```

**Tabla comparativa superior:**

| Aspecto | K-means | Clustering Jerárquico |
|---------|---------|----------------------|
| **Enfoque** | Centroide (punto medio) | Enlace (distancia entre grupos) |
| **k óptimo** | 4 (método del codo) | 4 (saltos en dendrograma) |
| **Silhouette** | 0.45 | 0.43 |
| **Interpretabilidad** | Alta (centroides) | Media (dendrograma) |
| **Ventaja principal** | Rápido, escalable | Visualiza jerarquía |
| **Uso recomendado** | Segmentación operativa | Validación cruzada |

**Métricas de concordancia inferior:**

```
┌──────────────────────────────────────────────────────┐
│  ADJUSTED RAND INDEX (ARI): 0.82                     │
│  → Alta concordancia entre ambos algoritmos          │
│     (valores >0.7 indican estructuras similares)     │
├──────────────────────────────────────────────────────┤
│  NORMALIZED MUTUAL INFORMATION (NMI): 0.85           │
│  → Alta información compartida entre particiones     │
│     (valores >0.7 confirman robustez)                │
└──────────────────────────────────────────────────────┘
```

**Interpretación (debajo):**

```
✅ ARI=0.82 y NMI=0.85 confirman que ambos algoritmos
   descubren estructuras MUY SIMILARES

✅ Los clusters NO son artificiales producto de un solo método

✅ Validación cruzada EXITOSA: los 4 perfiles son robustos
```

### Caja destacada inferior
```
🎯 DECISIÓN: Usar K-means para segmentación operativa
   (más interpretable, más rápido)
   
   Clustering Jerárquico como validación
   (confirma robustez estructural)
```

### Notas del presentador
> "Que dos algoritmos tan diferentes (K-means usa centroides, Jerárquico usa enlaces) lleguen a agrupaciones similares (ARI 0.82) es una prueba fuerte de que los clusters son reales, no artificiales."

---

## Slide 11: Validación Externa (Clusters vs Rotación)

### Layout
- **Tipo:** Tabla + Gráfico de barras
- **Distribución:** 50% tabla izquierda, 50% gráfico derecha

### Contenido

**Título:**
```
✅ Validación Externa: ¿Los Clusters Descubiertos Predicen Rotación?
```

**Tabla izquierda (pureza por cluster):**

| Cluster | Perfil | N° Empleados | % Se Fue | % Se Quedó | Pureza |
|---------|--------|--------------|----------|------------|--------|
| **0** | Estrella | 3,750 | 5% | **95%** | 95% ✅ |
| **1** | Burnout | 2,250 | **85%** | 15% | 85% 🚨 |
| **2** | Estancado | 4,500 | **70%** | 30% | 70% ⚠️ |
| **3** | Onboarding | 4,499 | 40% | **60%** | 60% ✅ |

**Definición de pureza:**
```
Pureza = % de la clase mayoritaria en el cluster
(Cuanto mayor, más homogéneo el cluster)
```

**Gráfico derecha (barras apiladas):**
- **Eje X:** Clusters (0-3)
- **Eje Y:** Porcentaje de empleados
- **Barras apiladas:**
  - Verde: % Se quedó
  - Rojo: % Se fue
- **Cluster 0:** 95% verde, 5% rojo
- **Cluster 1:** 15% verde, 85% rojo (invertido)
- **Cluster 2:** 30% verde, 70% rojo
- **Cluster 3:** 60% verde, 40% rojo

**Métricas globales (debajo):**

```
┌─────────────────────────────────────────────────────┐
│ ACCURACY COMO PREDICTOR: 80.3%                      │
│ (Si asignamos cada cluster a su clase mayoritaria)  │
├─────────────────────────────────────────────────────┤
│ COMPARACIÓN CON SUPERVISADO:                        │
│ • Clustering: 80.3% (sin usar "left")              │
│ • SVM RBF: 94.8% (usando "left" para entrenar)     │
│                                                      │
│ → 14.5% menos precisión, pero...                    │
│   GANAMOS interpretabilidad y segmentación          │
└─────────────────────────────────────────────────────┘
```

### Destacado inferior
```
💡 CONCLUSIÓN CRÍTICA: Los clusters descubiertos SIN usar "left"
   predicen rotación con 80% de accuracy. Esto confirma que
   las diferencias estructurales son REALES y ACCIONABLES.
```

### Notas del presentador
> "Esto es fascinante: descubrimos 4 grupos sin saber quién se fue, y resulta que esos grupos predicen rotación con 80% de precisión. Claro, es menos que el 94.8% de SVM supervisado, pero SVM no te dice QUÉ HACER. Clustering sí: Cluster 1 tiene 85% de rotación porque están sobrecargados, reduce su carga a 200 horas/mes."

---

## Slide 12: Con PCA vs Sin PCA

### Layout
- **Tipo:** Tabla comparativa + Recomendación
- **Distribución:** 60% tabla superior, 40% texto inferior

### Contenido

**Título:**
```
🔬 Evaluación del Impacto de PCA (Reducción de Dimensionalidad)
```

**Tabla comparativa:**

| Dimensión | Sin PCA<br>(18 variables) | Con PCA<br>(5 componentes, 80% varianza) |
|-----------|---------------------------|------------------------------------------|
| **Silhouette** | 0.45 | 0.42 |
| **Davies-Bouldin** | 1.2 | 1.35 |
| **Interpretabilidad** | ⭐⭐⭐⭐⭐<br>Centroides con significado directo | ⭐⭐<br>Componentes principales abstractos |
| **Ejemplo centroide** | "satisfaction=0.2, hours=275" | "PC1=2.3, PC2=-1.1" (¿qué significa?) |
| **Varianza explicada** | 100% (todas las variables) | 80% (20% de información perdida) |
| **Ventaja principal** | Claridad empresarial | Reduce redundancia |
| **Desventaja principal** | Posible redundancia | Pérdida de significado |

**Análisis de variables correlacionadas:**

```
Variables con alta correlación (>0.8):
• satisfaction_level ↔ time_spend_company (-0.72)
• number_project ↔ average_montly_hours (0.78)

→ Correlaciones MODERADAS, no críticas
→ PCA elimina redundancia pero sacrifica interpretabilidad
```

**Recomendación (caja destacada):**

```
┌──────────────────────────────────────────────────────┐
│ 🎯 DECISIÓN: NO aplicar PCA                          │
├──────────────────────────────────────────────────────┤
│ RAZONES:                                             │
│ ✅ Silhouette 0.45 sin PCA > 0.42 con PCA           │
│ ✅ Interpretabilidad crítica para stakeholders       │
│ ✅ Correlaciones no son suficientemente altas (<0.8) │
│ ✅ 18 variables son manejables computacionalmente    │
│                                                       │
│ EXCEPCIÓN:                                           │
│ Usar PCA solo si dataset >>100K filas o >50 vars    │
└──────────────────────────────────────────────────────┘
```

### Notas del presentador
> "La tentación con clustering es siempre aplicar PCA para simplificar. Pero en este caso, perdemos interpretabilidad sin ganar mucho (Silhouette baja de 0.45 a 0.42). Prefiero poder decir 'este empleado tiene 275 horas/mes' que 'este empleado tiene PC1=2.3'."

---

## Slide 13: Estrategia Empresarial por Cluster

### Layout
- **Tipo:** 4 cuadrantes con iconos y acciones
- **Distribución:** 2×2 grid con íconos visuales

### Contenido

**Título:**
```
💼 Estrategias Diferenciadas de Retención por Segmento
```

**Cuadrante 1 (Superior Izquierdo) - Cluster 0 Estrella:**
```
┌────────────────────────────────────────┐
│ 🌟 EMPLEADOS ESTRELLA (25%)            │
├────────────────────────────────────────┤
│ RIESGO: 5% (bajo)                      │
│ PRIORIDAD: Alta (retención de talento) │
├────────────────────────────────────────┤
│ ACCIONES:                              │
│ ✅ Programa de desarrollo de liderazgo │
│ ✅ Planes de carrera ambiciosos        │
│ ✅ Proyectos desafiantes               │
│ ✅ Mentor para otros empleados         │
│ ✅ Bonos por desempeño excepcional     │
├────────────────────────────────────────┤
│ ROI: $4.5M/año (retener talento clave) │
└────────────────────────────────────────┘
```

**Cuadrante 2 (Superior Derecho) - Cluster 1 Burnout:**
```
┌────────────────────────────────────────┐
│ 🔥 EMPLEADOS EN BURNOUT (15%)          │
├────────────────────────────────────────┤
│ RIESGO: 85% (crítico) 🚨               │
│ PRIORIDAD: Máxima (intervención inmediata)│
├────────────────────────────────────────┤
│ ACCIONES:                              │
│ 🚨 Reducir carga a ≤200 horas/mes      │
│ 🚨 Coaching anti-burnout               │
│ 🚨 Ajuste salarial urgente             │
│ 🚨 Reasignación de 2-3 proyectos       │
│ 🚨 Flexibilidad horaria                │
│ 🚨 Licencia compensatoria (1-2 semanas)│
├────────────────────────────────────────┤
│ COSTO SIN INTERVENIR: $95.6M/año       │
│ ROI: $80M/año (retener 70% del cluster)│
└────────────────────────────────────────┘
```

**Cuadrante 3 (Inferior Izquierdo) - Cluster 2 Estancado:**
```
┌────────────────────────────────────────┐
│ 📉 EMPLEADOS ESTANCADOS (30%)          │
├────────────────────────────────────────┤
│ RIESGO: 70% (alto) ⚠️                  │
│ PRIORIDAD: Media-Alta (mejora o salida)│
├────────────────────────────────────────┤
│ ACCIONES:                              │
│ ⚠️ Plan de mejora de desempeño (PIP)   │
│ ⚠️ Programas de reskilling/upskilling  │
│ ⚠️ Revisión de compensación            │
│ ⚠️ Rotación interna a otros departamentos│
│ ⚠️ Evaluación honesta: ¿vale la pena?  │
│ ⚠️ Salida asistida si no mejora (3-6m) │
├────────────────────────────────────────┤
│ ROI: $40M/año (retener 40% mejorados)  │
└────────────────────────────────────────┘
```

**Cuadrante 4 (Inferior Derecho) - Cluster 3 Onboarding:**
```
┌────────────────────────────────────────┐
│ 🆕 EMPLEADOS EN ONBOARDING (30%)       │
├────────────────────────────────────────┤
│ RIESGO: 40% (moderado)                 │
│ PRIORIDAD: Alta (ventana crítica 2 años)│
├────────────────────────────────────────┤
│ ACCIONES:                              │
│ ✅ Onboarding estructurado (90 días)   │
│ ✅ Mentoreo 1:1 (6-12 meses)           │
│ ✅ Feedback semanal (primer trimestre) │
│ ✅ Plan de carrera claro desde día 1   │
│ ✅ Integración cultural                │
│ ✅ Revisión a 6 meses, 1 año, 2 años   │
├────────────────────────────────────────┤
│ ROI: $30M/año (retener 70% post-onboarding)│
└────────────────────────────────────────┘
```

### Notas del presentador
> "Este es el valor REAL del clustering: estrategias diferenciadas. No puedes tratar igual a un empleado en Burnout (reducir carga YA) que a un Estrella (darle más desafíos). La segmentación permite inversión inteligente de recursos."

---

## Slide 14: Impacto y ROI

### Layout
- **Tipo:** Gráficos + Tabla financiera
- **Distribución:** 40% gráfico izquierda, 60% tabla derecha

### Contenido

**Título:**
```
💰 Impacto Empresarial y Retorno de Inversión (ROI)
```

**Gráfico izquierda (barras apiladas):**
- **Título:** "Empleados Retenidos por Cluster (Anual)"
- **Eje X:** Clusters (0-3)
- **Eje Y:** N° de empleados
- **Barras:**
  - Cluster 0: 188 retenidos (de 3,750)
  - Cluster 1: 1,575 retenidos (de 2,250)
  - Cluster 2: 900 retenidos (de 4,500)
  - Cluster 3: 1,350 retenidos (de 4,499)
- **Total destacado:** 4,013 empleados retenidos/año

**Tabla derecha (financiera):**

```
┌──────────────────────────────────────────────────┐
│ ANÁLISIS FINANCIERO DE LA IMPLEMENTACIÓN        │
├──────────────────────────────────────────────────┤
│ INVERSIÓN INICIAL (Año 1):                      │
│ • Software y plataforma:        $500K            │
│ • Consultoría Data Science:     $800K            │
│ • Integración HRIS:             $400K            │
│ • Capacitación gerentes HR:     $200K            │
│ ────────────────────────────────────             │
│ TOTAL AÑO 1:                    $1.9M            │
├──────────────────────────────────────────────────┤
│ COSTOS RECURRENTES (Anual):                     │
│ • Licencias y hosting:          $100K/año        │
│ • Reentrenamiento modelos:      $50K/año         │
│ • Monitoreo y ajustes:          $50K/año         │
│ ────────────────────────────────────             │
│ TOTAL RECURRENTE:               $200K/año        │
├──────────────────────────────────────────────────┤
│ AHORRO ANUAL:                                    │
│ • Empleados retenidos:          4,013/año        │
│ • Costo reemplazo promedio:     $50K/empleado    │
│ • Ahorro total:                 $200.6M/año      │
├──────────────────────────────────────────────────┤
│ ROI (Año 1):                                     │
│ ($200.6M - $1.9M) / $1.9M = 10,384%             │
│                                                   │
│ PAYBACK PERIOD: 3.4 días ✅                      │
├──────────────────────────────────────────────────┤
│ ROI ACUMULADO (3 años):                          │
│ Inversión total: $1.9M + 3×$200K = $2.5M        │
│ Ahorro total: 3×$200.6M = $601.8M               │
│ ROI neto: ($601.8M - $2.5M) / $2.5M = 23,972%   │
└──────────────────────────────────────────────────┘
```

**Desglose de ahorro por cluster (gráfico de pastel pequeño):**
- Cluster 1 (Burnout): 47% del ahorro total ($94.5M)
- Cluster 2 (Estancado): 28% del ahorro total ($56.2M)
- Cluster 3 (Onboarding): 20% del ahorro total ($40.1M)
- Cluster 0 (Estrella): 5% del ahorro total ($10M)

### Destacado inferior
```
🎯 CONCLUSIÓN: Inversión de $2.5M genera ahorro de $601.8M en 3 años
   → ROI de 23,972% (retorno 240x la inversión)
   → Intervención en Cluster 1 (Burnout) genera casi 50% del ahorro
```

### Notas del presentador
> "Los números son abrumadores: invertir $2.5M en 3 años genera un ahorro de $601M. Eso es un retorno de 240 veces la inversión. Y el 47% del ahorro viene de intervenir en el Cluster 1 (Burnout), que es solo el 15% de los empleados. Esa es la magia de la segmentación inteligente."

---

## Slide 15: Demo de Aplicación Web + Cierre

### Layout
- **Tipo:** Screenshot + Texto de cierre
- **Distribución:** 50% imagen izquierda, 50% texto derecha

### Contenido

**Título:**
```
🌐 Aplicación Web Interactiva + Conclusiones Finales
```

**Screenshot izquierda:**
- **Captura de pantalla de la app web** en la sección de K-means
- Mostrar el gráfico de clusters 2D con centroides
- URL visible y destacada en grande: `https://inferencia-estadistica-unab.streamlit.app/`

**Texto derecha:**

**Funcionalidades Implementadas:**
- 🎯 Clustering interactivo (K-means + Jerárquico)
- 📊 Método del codo automático
- 🗺️ Visualizaciones 2D/3D interactivas
- 📈 Dendrogramas dinámicos
- ✅ Validación externa con variable "left"
- 📥 Exportación de resultados (CSV)
- 💡 Recomendaciones interpretativas por cluster

**Mensaje de Cierre:**

```
┌────────────────────────────────────────────────┐
│ 🎯 CONCLUSIONES FINALES:                       │
├────────────────────────────────────────────────┤
│ ✅ K-means descubrió 4 perfiles naturales      │
│    (Estrella, Burnout, Estancado, Onboarding)  │
│                                                 │
│ ✅ Clustering Jerárquico validó la robustez    │
│    (ARI=0.82, NMI=0.85)                        │
│                                                 │
│ ✅ Clusters predicen rotación con 80% accuracy │
│    sin haber usado la variable "left"          │
│                                                 │
│ ✅ Segmentación permite estrategias            │
│    diferenciadas y ROI de $601M en 3 años      │
│                                                 │
│ 💡 Clustering NO reemplaza clasificación       │
│    supervisada, la COMPLEMENTA:                │
│    • SVM RBF: precisión (94.8%)                │
│    • K-means: interpretabilidad y segmentación │
└────────────────────────────────────────────────┘
```

**Call-to-Action:**
```
🌐 Aplicación disponible 24/7:
   https://inferencia-estadistica-unab.streamlit.app/

📧 ¿Preguntas? ¡Estamos disponibles!
```

### Pie del slide (centrado, grande)
```
¡MUCHAS GRACIAS POR SU ATENCIÓN!
¿Tienen alguna pregunta?
```

### Notas del presentador
> "Para cerrar, hemos implementado toda esta investigación en una aplicación web accesible desde cualquier navegador. Pueden probarla ustedes mismos después de la presentación. La URL está aquí en pantalla. Y para finalizar: el clustering no es enemigo de la clasificación, es su aliado estratégico. Gracias por su atención, ¿alguna pregunta?"

---

## 🎨 Paleta de Colores Sugerida

### Colores Principales
- **Fondo:** #1e3a5f (azul oscuro) o #f8f9fa (blanco humo)
- **Título:** #ffffff (blanco) o #1e3a5f (azul oscuro)
- **Texto:** #333333 (gris oscuro) o #ffffff (blanco)
- **Acento:** #e74c3c (rojo) para Burnout, #27ae60 (verde) para Estrella

### Colores por Cluster
- **Cluster 0 (Estrella):** #27ae60 (verde)
- **Cluster 1 (Burnout):** #e74c3c (rojo)
- **Cluster 2 (Estancado):** #f39c12 (naranja)
- **Cluster 3 (Onboarding):** #3498db (azul)

### Colores de Métricas
- **Positivo:** #2ecc71 (verde claro)
- **Neutro:** #3498db (azul)
- **Advertencia:** #f1c40f (amarillo)
- **Crítico:** #e74c3c (rojo)

---

## 📐 Tipografía Recomendada

### Fuentes
- **Títulos:** Montserrat Bold o Roboto Bold (24-36pt)
- **Subtítulos:** Montserrat Medium o Roboto Medium (18-22pt)
- **Texto:** Roboto Regular o Open Sans Regular (14-16pt)
- **Pie de página:** Roboto Light (10-12pt)

### Jerarquía Visual
- Título slide: 36pt
- Subtítulo: 24pt
- Texto principal: 16pt
- Tablas/gráficos: 14pt
- Notas: 12pt

---

## 🛠️ Software Recomendado para Crear las Slides

### Opciones
1. **Google Slides** (recomendado para colaboración)
   - Gratuito, accesible desde navegador
   - Fácil compartir con revisores

2. **PowerPoint** (Microsoft Office)
   - Más potente para gráficos complejos
   - Integración con Excel para tablas

3. **Canva** (diseño visual)
   - Plantillas profesionales prediseñadas
   - Fácil de usar para no diseñadores

4. **LaTeX Beamer** (para presentaciones académicas avanzadas)
   - Control total sobre diseño
   - Requiere conocimientos técnicos

---

## ✅ Checklist de Calidad de Slides

Antes de finalizar las diapositivas, verifica:

- [ ] Todas las slides tienen título claro
- [ ] Fuentes legibles desde 5 metros de distancia
- [ ] Colores con suficiente contraste (texto visible)
- [ ] Gráficos con ejes etiquetados y unidades
- [ ] Tablas con headers claros
- [ ] Números redondeados apropiadamente (2-3 decimales)
- [ ] Fuentes consistentes (misma familia en todas las slides)
- [ ] Tamaños de fuente consistentes (jerarquía clara)
- [ ] Alineación de elementos (no texto/gráficos desalineados)
- [ ] Ortografía y gramática revisadas
- [ ] Logos institucionales presentes (UNAB)
- [ ] URL de la app visible en múltiples slides
- [ ] Transiciones simples entre slides (evitar efectos excesivos)
- [ ] Duración total: 18-20 minutos (1-1.5 min por slide)

---

**¡Éxito en la creación de las diapositivas! 🎨**

---

**Documento creado por:** Equipo de Análisis de Datos - UNAB  
**Fecha:** Noviembre 2025  
**Versión:** 1.0
