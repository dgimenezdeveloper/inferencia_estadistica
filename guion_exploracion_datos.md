# Guion para Video: Exploración de Datos – Dataset `regresión.csv`

## 1. Introducción y Presentación del Dataset

En este video vamos a explorar el dataset `regresión.csv` utilizando la app de Inferencia Estadística y Reconocimiento de Patrones. El objetivo es entender la estructura de los datos, detectar posibles problemas y preparar el terreno para el modelado.

**Variables del dataset:**

- `duracion`: Duración de la actividad o evento (numérica, rango amplio).
- `paginas`: Cantidad de páginas (numérica, valores bajos).
- `acciones`: Número de acciones realizadas (numérica).
- `valor`: Algún valor cuantitativo asociado (numérica, dispersión alta).
- `clase`: Variable objetivo (target), con tres categorías:
    - 0: Bajo Rendimiento
    - 1: Medio Rendimiento
    - 2: Alto Rendimiento

## 2. Estadísticos Descriptivos

Mostramos la tabla de estadísticos descriptivos para cada variable:

- **¿Por qué es importante?**
    - Permite detectar variables con escalas muy diferentes (por ejemplo, `duracion` tiene valores máximos de hasta 898, mientras que `paginas` llega solo a 9).
    - Ayuda a identificar posibles outliers y errores de carga.
    - Da una idea de la dispersión y simetría de los datos.

**Interpretación:**
- `duracion` tiene una media de 111 y una desviación estándar muy alta (202), lo que indica alta dispersión y posible presencia de valores extremos.
- `paginas`, `acciones` y `valor` también muestran asimetría y dispersión, pero en menor medida.
- La variable `clase` está codificada como 0, 1 y 2.

## 3. Valores Nulos y Atípicos

- **Nulos:** No se detectaron valores nulos en ninguna columna, lo que es ideal para el modelado.
- **Outliers:** Se detectaron algunos outliers (z-score > 3), pero en proporciones bajas (< 10%), por lo que no deberían afectar gravemente el análisis.

## 4. Matriz de Correlación

Se presenta la matriz de correlación entre todas las variables numéricas.

- **Hallazgos:**
    - Se detecta una correlación alta (0.86) entre `acciones` y `valor`.
    - También hay correlaciones moderadas entre otras variables.
- **Advertencia automática:** La app sugiere eliminar variables redundantes o aplicar PCA antes de usar algoritmos sensibles a la correlación (como Bayes Ingenuo).

## 5. Matriz de Covarianza

Se muestra la matriz de covarianza, que refuerza la presencia de relaciones fuertes entre algunas variables.

- **Hallazgos:**
    - Covarianza alta entre `duracion` y `valor`, `acciones` y `valor`, y `duracion` y `acciones`.
- **Advertencia automática:** Se recomienda escalar los datos o aplicar PCA para reducir la redundancia y mejorar la estabilidad del modelo.

## 6. Distribución de Clases

Se visualiza la distribución de la variable objetivo `clase`:

- 0 (Bajo Rendimiento): 86 casos
- 1 (Medio Rendimiento): 40 casos
- 2 (Alto Rendimiento): 44 casos

**Interpretación:**
- El dataset está moderadamente balanceado, aunque la clase 0 tiene el doble de casos que las otras.

## 7. Visualizaciones Básicas

Se presentan histogramas para cada variable numérica:

- Todas las variables muestran distribuciones asimétricas (sesgo a la derecha), con muchos valores bajos y pocos valores altos (típico de conteos o duraciones).
- Esto refuerza la necesidad de considerar escalado y/o transformaciones si se usan algoritmos sensibles a la escala.

## 8. Recomendaciones y Próximos Pasos

- La app recomienda aplicar PCA o eliminar variables redundantes si se usará Bayes Ingenuo.
- Recomienda escalar las variables si hay covarianzas muy altas.
- Justifica cada paso con base en la evidencia explorada.

## 9. Resumen para la Presentación

1. Presentar el dataset y explicar el significado de cada variable y clase.
2. Analizar los estadísticos descriptivos y resaltar diferencias de escala y dispersión.
3. Confirmar que no hay nulos y que los outliers no son un problema grave.
4. Interpretar la matriz de correlación y covarianza, explicando la importancia de PCA/escalado.
5. Mostrar la distribución de clases y discutir el balance.
6. Analizar los histogramas y sugiere transformaciones si es necesario.
7. Cerrar con las recomendaciones automáticas de la app y los próximos pasos para el modelado.

---

**Este guion puede ser leído o adaptado para el video, asegurando que cada sección se relacione con los gráficos y advertencias que muestra la app.**