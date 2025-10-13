# Guía Completa de PCA y Selección Automática

## ¿Qué es PCA y para qué sirve?
El Análisis de Componentes Principales (PCA, por sus siglas en inglés) es una técnica estadística de reducción de dimensionalidad. Su objetivo principal es transformar un conjunto de variables posiblemente correlacionadas en un conjunto más pequeño de variables no correlacionadas llamadas componentes principales. Cada componente principal es una combinación lineal de las variables originales y está ordenado de manera que el primero explica la mayor cantidad de varianza posible, el segundo la siguiente mayor cantidad, y así sucesivamente.

**¿Por qué es importante?**
- Reducción de dimensionalidad: Permite simplificar datasets con muchas variables, facilitando el análisis, la visualización y el modelado.
- Eliminación de redundancia: Al identificar y combinar variables correlacionadas, ayuda a eliminar información redundante.
- Mejora de modelos: Puede mejorar el rendimiento de algoritmos de machine learning al eliminar ruido y variables irrelevantes.
- Visualización: Permite representar datos complejos en 2D o 3D, facilitando la interpretación visual de la estructura de los datos.
- Evita el sobreajuste: Al reducir el número de variables, disminuye el riesgo de que los modelos se ajusten demasiado a los datos de entrenamiento.

**¿Cómo funciona PCA?**
1. Estandarización: Se recomienda escalar los datos para que todas las variables tengan media cero y varianza uno.
2. Cálculo de la matriz de covarianza: Se calcula cómo varían conjuntamente las variables.
3. Obtención de autovectores y autovalores: Se extraen los vectores propios (direcciones principales) y valores propios (importancia/varianza explicada).
4. Selección de componentes: Se eligen los componentes principales que explican la mayor parte de la varianza (por ejemplo, el 80% o 90%).
5. Proyección: Los datos originales se proyectan sobre estos nuevos ejes, obteniendo un dataset reducido.

---
## ¿Qué significa la varianza en PCA?
1. **Definición básica de varianza:**<br>
La varianza mide cuánto se dispersan los datos respecto a su media. Matemáticamente, es el promedio de las diferencias al cuadrado entre cada valor y la media del conjunto. Si la varianza es alta, los datos están muy dispersos; si es baja, los datos están más concentrados cerca de la media.

2. **Varianza en el contexto de PCA** <br>
En PCA, la varianza tiene un significado especial:
    - **Cada componente principal** es una nueva variable (una combinación lineal de las originales) que captura una cierta cantidad de la varianza total de los datos.
- **La varianza explicada por un componente** indica cuánta información (o estructura) de los datos originales está capturando ese componente.
    - **El primer componente principal** es la dirección en el espacio de los datos donde la varianza es máxima, es decir, donde los datos están más "extendidos".
    - **Componentes siguientes** capturan la máxima varianza posible en direcciones ortogonales (perpendiculares) a los anteriores.

3. ¿Por qué es importante la varianza en PCA?
- **Información útil**: En la mayoría de los datasets, la mayor parte de la información relevante está en las direcciones de mayor varianza. Por eso, los primeros componentes principales suelen ser los más informativos.
- **Reducción de dimensionalidad**: Al conservar solo los componentes que explican la mayor parte de la varianza, podemos reducir el número de variables sin perder mucha información.
- **Ruido vs. señal**: Componentes con muy poca varianza suelen corresponder a ruido o detalles poco relevantes.

4. **Interpretación práctica** <br>
    - Si un componente principal explica, por ejemplo, el 60% de la varianza, significa que ese solo componente resume el 60% de la información total de los datos originales.
    - Si los primeros dos componentes explican juntos el 90% de la varianza, puedes visualizar los datos en 2D y seguir capturando casi toda la estructura original.

5. **En resumen** <br>
    - **Varianza = información**: En PCA, más varianza explicada por un componente significa que ese componente es más importante para describir las diferencias entre los datos.
    - **Elegir componentes**: Se suelen elegir los componentes que, en conjunto, explican un alto porcentaje de la varianza (por ejemplo, 80-95%).

---
## ¿Qué significa la composición de cada componente principal?
En la sección "Composición de cada componente principal" para el ejemplo del dataset de géneros musicales se observan expresiones como:
```bash
    PC1 = +0.11 × Popularity +0.05 × danceability +0.53 × energy +0.02 × key +0.51 × loudness ...
```
Esto significa:
- **PC1** (Primer Componente Principal) es una combinación lineal de las variables originales (Popularity, danceability, energy, etc.).
- **El coeficiente** de cada variable indica cuánto "pesa" esa variable en el componente. Por ejemplo, en PC1, energy y loudness tienen coeficientes altos, por lo que este componente está muy influenciado por esas variables.
- Si una variable tiene un coeficiente positivo alto, contribuye fuertemente y en la misma dirección; si es negativo, contribuye en sentido opuesto.

### Interpretación práctica:<br>
PC1 resume la información de varias variables, pero sobre todo de aquellas con coeficientes más altos. Si, por ejemplo, energy y loudness dominan, entonces PC1 puede interpretarse como un "eje de energía/intensidad" de la música.

## ¿Qué significa el porcentaje de varianza explicada y acumulada?
- **Porcentaje de varianza explicada por cada componente:**
Por ejemplo, si PC1 explica el 20.4% y PC2 el 12.4%, significa que solo con esos dos componentes ya capturas el 32.8% de toda la variabilidad de las canciones en del dataset.

- **Varianza acumulada:**
Es la suma de la varianza explicada por los primeros N componentes. En tu caso, con 9 componentes llegas al 82.19%.
Esto significa que, aunque el dataset tenga muchas variables, solo necesitas 9 combinaciones (componentes principales) para conservar más del 80% de la información original.

## ¿Cómo se interpreta esto para la clasificación de géneros musicales?
- **Reducción de dimensionalidad:** <br>
Puedes trabajar con 9 componentes en vez de todas las variables originales, simplificando el modelo y acelerando el procesamiento.
- **Separación de géneros:** <br>
Si proyectas los datos en los primeros componentes (por ejemplo, PC1 vs PC2), puedes visualizar si los géneros musicales se agrupan o separan en el espacio reducido. Si ves agrupamientos claros, significa que los géneros tienen características musicales distintas que PCA logra capturar.
- **Importancia de las variables:**<br>
Analizando la composición de los componentes, puedes descubrir qué características musicales (por ejemplo, energía, tempo, danceability) son más relevantes para diferenciar géneros.

## Ejemplo concreto con lso resultados PCA de Géneros Musicales
- **PC1:** Si está dominado por energy y loudness, probablemente separa géneros "fuertes" (como Rock o Metal) de géneros "suaves" (como Acoustic o Pop).
- **PC2**: Si tiene peso en danceability y popularity, puede estar diferenciando géneros bailables de los menos bailables.
- **Varianza acumulada (82.19%)**: Usando los primeros 9 componentes, tu modelo o visualización conserva la mayor parte de la información relevante para distinguir géneros, descartando solo el "ruido" o detalles menos importantes.

## ¿Cómo interpretar los gráficos de PCA?

1. **Gráfico de varianza explicada**: 
    - **¿Qué muestra?** <br>
    Una barra por cada componente principal, indicando cuánta varianza explica.
    - **¿Cómo interpretarlo?** <br> 
    Componentes con mayor varianza explicada son más importantes. El "codo" en la curva acumulada sugiere cuántos componentes usar: después de ese punto, añadir más componentes aporta poca información adicional.
2. **Gráfico de dispersión  de componentes principales  (2D/3D)**: 
    - **¿Qué muestra?**<br>
    Los datos proyectados en los primeros dos o tres componentes principales.
    - **¿Cómo interpretarlo?**<br>
        - Agrupamientos: Si ves grupos separados, indica que las clases o categorías tienen diferencias claras en las variables originales.
        - Solapamiento: Si los puntos de diferentes clases se mezclan, las variables originales no permiten distinguir bien esas clases.
        - Outliers: Puntos alejados pueden indicar valores atípicos.
3. **Cargas de los componentes (loadings)**
    - **¿Qué muestra?**<br>
    Los coeficientes que indican cuánto contribuye cada variable original a cada componente principal.
    - **¿Cómo interpretarlo?**<br>
        - Un valor alto (positivo o negativo) indica que esa variable es importante para ese componente.
        - Analizando los loadings puedes interpretar el significado de cada componente (por ejemplo, "PC1 representa una combinación de alcohol y acidez").
4. **Matriz de Covarianza y Correlación**
    - **¿Qué muestra?**<br>
    Cómo varían conjuntamente las variables originales.
    - **¿Cómo interpretarlo?**<br>
    Valores altos indican variables redundantes; PCA tiende a combinar estas en un solo componente.
5. **Solapamiento de grupos**: <br> 
    si los puntos de diferentes clases se mezclan, significa que esas clases tienen características similares y serán más difíciles de separar.
    En resumen:

    - **Grupos mezclados = características similares = clasificación difícil**
    - **Grupos separados = características distintas = clasificación más fácil**

### Consejos para interpretar resultados de PCA
- No todos los componentes son igual de importantes: Concéntrate en los primeros, que explican la mayor parte de la varianza.
- El significado de los componentes depende de los loadings: Analiza qué variables tienen mayor peso en cada componente.
- La reducción de dimensionalidad puede perder información: Elige el número de componentes según el porcentaje de varianza que quieras conservar.
- Visualiza los datos en los nuevos ejes: Esto puede revelar agrupamientos, tendencias o anomalías que no eran evidentes en el espacio original.

---
## ¿Qué es la matriz de componentes y los loadings?
La matriz de componentes muestra cómo cada variable original contribuye a cada componente principal (los “loadings” o pesos).
- Un valor alto (positivo o negativo) indica que esa variable influye mucho en ese componente.
- El signo indica la dirección, pero no si “ayuda” o “perjudica” la clasificación.

---
## ¿Cómo elegir el número de componentes?
- Tradicionalmente, se elige el número de componentes que explica al menos el 80% de la varianza acumulada.
- Mejor aún: usar la auto-selección por validación cruzada (ver más abajo) para elegir el n que maximiza el rendimiento del clasificador.

---
## ¿Qué es la auto-selección de componentes PCA por validación cruzada (CV)?
Esta funcionalidad te ayuda a elegir **cuántos componentes principales (n)** usar en PCA, pero de forma **objetiva y automática**, basándose en el rendimiento real de un modelo de clasificación (no solo en la varianza explicada).

### ¿Cómo funciona?
1. Escala los datos.
2. Para cada posible n (número de componentes):
   - Aplica PCA con ese n.
   - Proyecta los datos.
   - Entrena y evalúa un clasificador (por defecto, LogisticRegression) usando validación cruzada (CV).
   - Guarda el promedio de la métrica elegida (accuracy o f1_macro).
3. Busca el n más pequeño cuya media de score esté dentro de la tolerancia (por ejemplo, 1%) del mejor resultado observado.
4. Te muestra una gráfica: eje X = n, eje Y = score. Marca el n recomendado y el n con mejor score.
5. Actualiza el slider de componentes con el n recomendado.

### ¿Qué es un clasificador? ¿Por qué LogisticRegression?
- Un **clasificador** es un modelo que predice a qué clase pertenece cada muestra (por ejemplo, género musical, tipo de vino, etc.).
- **LogisticRegression** es un modelo simple y estándar, ideal para comparar y medir la separabilidad de los datos tras PCA.
- No es el modelo final, solo una “regla de evaluación” para comparar cuántos componentes usar.

### ¿Qué significan las métricas?
- **accuracy**: Porcentaje de aciertos totales. Útil si las clases están balanceadas.
- **f1_macro**: Promedio del F1-score de cada clase. Mejor si tienes clases desbalanceadas o te importa el rendimiento en todas las clases por igual.

### ¿Qué es la tolerancia?
- Es el margen de “flexibilidad” para elegir el número de componentes.
- Ejemplo: si el mejor score se logra con 14 componentes, pero con 13 el score es solo 1% menor, la tolerancia de 0.01 (1%) permite elegir 13 (menos dimensiones, casi mismo rendimiento).

### ¿Por qué usar este método?
- Así eliges el número de componentes que realmente maximiza (o casi) el rendimiento de tu clasificador, no solo la varianza explicada.
- Evitas usar más dimensiones de las necesarias (menos sobreajuste, más interpretabilidad).
- Es un método objetivo, reproducible y adaptado a tu dataset y problema de clasificación.

---
## Ejemplo de justificación
“Seleccioné el número de componentes principales usando validación cruzada, eligiendo el n que maximiza el rendimiento de un clasificador estándar (LogisticRegression) según la métrica f1_macro. Así garantizo que la reducción de dimensionalidad no solo conserva la varianza, sino que también optimiza la capacidad de clasificación del modelo.”

---
## Preguntas frecuentes
**¿Puedo usar PCA con variables categóricas?**
No directamente. PCA requiere variables numéricas. Convierte las categóricas a numéricas primero.

**¿Qué pasa si todos los grupos se mezclan en el gráfico?**
Significa que las clases son muy similares en las variables elegidas. Prueba con otras variables o técnicas.

**¿El signo de los loadings indica si una variable ayuda o perjudica?**
No. Solo indica la dirección en el espacio de componentes. Lo importante es el valor absoluto (magnitud).

**¿Por qué a veces el mejor n de componentes no coincide con el 80% de varianza?**
Porque la varianza no siempre se traduce en mejor capacidad de clasificación. Por eso es mejor usar validación cruzada.

---
## Analogía práctica
Elegir el número de componentes en PCA es como elegir cuántos ingredientes usar en una receta: quieres los suficientes para que el plato tenga sabor (información), pero no tantos que se vuelva confuso o difícil de digerir (ruido, sobreajuste). La auto-selección por CV es como probar el plato con diferentes combinaciones y quedarte con la que mejor puntaje obtiene en una cata a ciegas.

---
## Consejos
- Si te preguntan “¿cuántos componentes usar?”, responde: “Elijo el n que maximiza el rendimiento de mi clasificador según validación cruzada, usando f1_macro si hay desbalance de clases, o accuracy si no”.
- Si te piden justificar: “No me baso solo en la varianza explicada, sino en el rendimiento real del modelo sobre los datos proyectados”.
- Puedes mostrar la gráfica de score vs n y explicar cómo se eligió el n recomendado.
