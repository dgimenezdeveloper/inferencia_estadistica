# Archivo de textos de ayuda y markdown para la app integrada

TEXTO_BAYES = r'''
**Bayes Ingenuo** es un algoritmo de clasificación supervisada basado en el Teorema de Bayes, con el supuesto de que las características son independientes entre sí dado la clase.

**Teorema de Bayes:**
$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

- $P(C|X)$: Probabilidad de la clase $C$ dado los atributos $X$.
- $P(X|C)$: Probabilidad de observar $X$ si la clase es $C$.
- $P(C)$: Probabilidad previa de la clase $C$.
- $P(X)$: Probabilidad de observar $X$.

**Supuesto ingenuo:**
$$
P(X|C) = \prod_{i=1}^n P(x_i|C)
$$
Esto simplifica el cálculo, aunque en la práctica las variables pueden estar correlacionadas.

**¿Cómo funciona?**
1. Calcula la probabilidad previa de cada clase y la probabilidad condicional de cada atributo dado la clase.
2. Para una nueva observación, multiplica las probabilidades y elige la clase con mayor probabilidad posterior.

**Ventajas:**
- Muy rápido y eficiente.
- Funciona bien incluso con pocos datos.
- Fácil de implementar.

**Desventajas:**
- El supuesto de independencia rara vez se cumple totalmente.
- No modela relaciones entre variables.

**Aplicaciones:**
- Clasificación de correos (spam/no spam), análisis de sentimientos, diagnóstico médico, etc.
'''

TEXTO_LDA_QDA = '''
**LDA (Análisis Discriminante Lineal)** y **QDA (Análisis Discriminante Cuadrático)** son algoritmos de clasificación supervisada que buscan separar clases en función de sus características.
- **LDA** asume que todas las clases tienen igual matriz de covarianza y genera fronteras lineales.
- **QDA** permite que cada clase tenga su propia matriz de covarianza y genera fronteras cuadráticas.
'''

TEXTO_PCA = '''
**PCA (Análisis de Componentes Principales)** es una técnica que transforma tus variables originales en nuevas variables llamadas *componentes principales*.

- Cada componente principal es una combinación de las variables originales.
- El **primer componente principal (PC1)** es la dirección donde los datos varían más.
- El **segundo componente principal (PC2)** es la siguiente dirección de máxima variabilidad, perpendicular a la primera.
- Así sucesivamente para los demás componentes.

**¿Para qué sirve?**
- Para reducir la cantidad de variables y simplificar el análisis.
- Para visualizar datos multidimensionales en 2D o 3D.
- Para eliminar redundancia si algunas variables están correlacionadas.

**Ejemplo visual:**
Si tienes datos de vinos con 10 características, PCA puede crear 2 componentes principales que expliquen el 80% de la variabilidad. Así puedes graficar y analizar los vinos en solo 2 dimensiones, sin perder casi nada de información.

**¿En qué se basa?**
- PCA calcula la matriz de covarianza de los datos y encuentra las direcciones (componentes) donde los datos varían más.
- Cada componente tiene un porcentaje de varianza explicada: indica cuánta información conserva ese componente.

**Interpretación de los porcentajes:**
- Si el primer componente explica el 60% y el segundo el 20%, juntos explican el 80% de la variabilidad de los datos.
- Puedes decidir cuántos componentes usar según la varianza acumulada.
'''

GUIA_USO = '''
### 🚀 Cómo usar esta aplicación

**1. Selecciona el tipo de análisis:**
- **Discriminante (LDA/QDA)**: Para clasificación supervisada lineal o cuadrática
- **Bayes Ingenuo**: Para clasificación supervisada basada en el teorema de Bayes
- **PCA**: Para reducción de dimensiones

**2. Carga tus datos:**
- Selecciona un archivo CSV de la carpeta de datos
- O sube tu propio archivo CSV

**3. Configura el análisis:**
- Selecciona las variables target y features
- Ajusta los parámetros según tus necesidades

**4. Interpreta los resultados:**
- Revisa las métricas y visualizaciones
- Usa las interpretaciones automáticas
- Explora las secciones expandibles para más detalles
'''

METRICAS_EXPLICADAS = '''
### 📊 Métricas de Clasificación

**Accuracy**: Proporción de predicciones correctas
- > 0.9: Excelente
- 0.8-0.9: Buena
- 0.7-0.8: Regular
- < 0.7: Necesita mejora

**Precision**: Exactitud de predicciones positivas
- Pregunta: "De los que predije positivos, ¿cuántos son realmente positivos?"

**Recall (Sensibilidad)**: Capacidad de encontrar casos positivos
- Pregunta: "De todos los casos positivos reales, ¿cuántos encontré?"

**F1-Score**: Media armónica entre precision y recall
- Balancea ambas métricas

**ROC-AUC**: Área bajo la curva ROC
- Mide capacidad discriminativa del modelo
- > 0.9: Excelente
- 0.8-0.9: Buena
- 0.7-0.8: Aceptable
- < 0.7: Pobre
'''

BAYES_EXPLICADO = '''
### 🤖 Bayes Ingenuo

- Algoritmo de clasificación supervisada basado en el Teorema de Bayes.
- Supone independencia entre las variables dado la clase.
- Muy eficiente y fácil de implementar.
- Útil para clasificación de texto, spam, análisis de sentimientos, etc.

**¿Cómo funciona?**
- Calcula la probabilidad de cada clase dado los atributos.
- Asigna la clase con mayor probabilidad posterior.

**Ventajas:**
- Rápido, robusto y funciona bien con pocos datos.

**Desventajas:**
- El supuesto de independencia rara vez se cumple totalmente.
'''

PCA_SIDEBAR = '''
### 📈 Componentes Principales

**PC1, PC2, PC3...**: Nuevas variables creadas
- PC1 explica la mayor varianza
- PC2 explica la segunda mayor varianza
- Son perpendiculares entre sí

**Varianza explicada**: Información conservada
- 80%+ acumulada es generalmente buena
- Ayuda a decidir cuántos componentes usar

**Matriz de componentes**: Cómo se construyen
- Cada fila = un componente
- Cada columna = una variable original
- Valores = importancia de cada variable

**Escalado**: Siempre recomendado
- Evita que variables de mayor rango dominen
- Permite comparación justa entre variables
'''

CONFIG_AVANZADA = '''
### 🛠️ Opciones Avanzadas

**Validación Cruzada**:
- Evalúa el modelo en diferentes subconjuntos
- Proporciona estimación más robusta
- K-fold típicamente entre 3-10

**Comparación de Modelos**:
- Evalúa los algoritmos estudiados (LDA, QDA, Bayes Ingenuo)
- Compara métricas lado a lado
- Proporciona recomendaciones

**Visualizaciones Interactivas**:
- Gráficos 3D con Plotly
- Hover para detalles
- Zoom y rotación disponibles

**Interpretaciones Automáticas**:
- Análisis de resultados
- Recomendaciones basadas en métricas
- Identificación de problemas comunes
'''
