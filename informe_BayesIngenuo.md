
# Informe de Resultados: Bayes Ingenuo sobre Rotación de Personal

## 1. Introducción

Este informe presenta los resultados del análisis realizado con el algoritmo Bayes Ingenuo (Naive Bayes) para predecir la rotación de personal en la misma base utilizada en los informes de LDA y QDA. El objetivo es comparar el desempeño de Bayes Ingenuo frente a los otros modelos y extraer conclusiones sobre su utilidad práctica.

## 2. Consideraciones sobre el dataset

El análisis exploratorio, la selección de variables y el tratamiento de datos se mantuvieron igual que en los informes anteriores (ver `informe_LDA.md`). Se utilizaron las mismas variables numéricas y la variable objetivo `left`.

## 3. Resultados del Modelo Bayes Ingenuo

### 3.1. Supuestos y matriz de correlación
Bayes Ingenuo asume independencia condicional entre las variables predictoras. La matriz de correlación muestra que, si bien hay algunas correlaciones bajas a moderadas, el supuesto de independencia no se cumple perfectamente, lo que puede afectar el desempeño del modelo.

### 3.2. Métricas globales

- **Accuracy:** 0.794
- **Precision (Macro):** 0.726
- **Recall (Macro):** 0.766
- **F1-Score (Macro):** 0.740
- **ROC-AUC:** 0.842

### 3.3. Métricas por clase

| Clase       | Precision | Recall | F1-Score | Soporte |
|-------------|-----------|--------|----------|---------|
| Permaneció  | 0.901     | 0.819  | 0.858    | 11.428  |
| Abandonó    | 0.552     | 0.712  | 0.622    | 3.571   |

**Interpretación:**
- El modelo predice con alta precisión la clase "Permaneció", pero tiene menor precisión y F1-score para la clase "Abandonó".
- El recall para "Abandonó" es aceptable (0.712), lo que indica que el modelo logra identificar una proporción razonable de quienes abandonan, aunque con bastantes falsos positivos.

### 3.4. Matriz de confusión

|              | Predicho: Permaneció | Predicho: Abandonó |
|--------------|---------------------|--------------------|
| Real: Permaneció | 9362                 | 2066               |
| Real: Abandonó   | 1028                 | 2543               |

**Mayor confusión:** El modelo tiende a confundir empleados que permanecieron con la clase "Abandonó" (2066 casos).

### 3.5. Curva ROC y AUC

El área bajo la curva ROC (AUC = 0.842) indica una buena capacidad discriminativa del modelo lo que significa que, en general, el modelo es capaz de distinguir entre las dos clases.

### 3.6. Predicción y probabilidades

El modelo permite obtener probabilidades asociadas a cada clase para una observación dada. Por ejemplo, para una observación típica, la probabilidad de "Permaneció" fue 0.734 y de "Abandonó" 0.266, mostrando una confianza moderada en la predicción.

## 4. Conclusiones y Comparación con LDA y QDA

- Bayes Ingenuo ofrece un desempeño aceptable, especialmente considerando su simplicidad y bajo costo computacional.
- El modelo es menos preciso que QDA y LDA, especialmente en la predicción de la clase minoritaria (Abandonó).
- El supuesto de independencia entre variables no se cumple totalmente, lo que limita el potencial del modelo en este caso.
- El AUC > 0.84 indica que el modelo es útil, pero no el más recomendable para este problema.

## 5. Recomendaciones y próximos pasos

- Bayes Ingenuo puede ser útil como modelo base o de referencia, pero no es el más adecuado para la toma de decisiones críticas en este contexto.
- Se recomienda priorizar modelos como QDA, que han mostrado mejor desempeño en todas las métricas.
- Puede ser interesante explorar variantes de Bayes (Bernoulli, Multinomial) si se incorporan variables categóricas.
- Continuar con la comparación frente a otros algoritmos y consolidar los resultados en un informe final.

---
**Este informe será integrado en el reporte comparativo final junto con los demás algoritmos.**