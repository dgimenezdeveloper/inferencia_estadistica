---
title: "Algoritmo de Bayes Ingenuo: Guía Teórica y Práctica"
author: "Universidad Nacional Guillermo Brown"
date: "25 de agosto de 2025"
<style>
/* Inspirado en la UNaB: Colores institucionales */
body {
teoria/bayes-ingenuo.md
  color: #222;
  background: #f7f7f7;
}
h1, h2, h3 {
  color: #0a3d62; /* Azul UNaB */
  font-family: 'Montserrat', Arial, sans-serif;
}
strong, b {
  color: #218c5a; /* Verde UNaB */
}
blockquote {
  border-left: 4px solid #218c5a;
  background: #eafaf1;
  padding: 0.5em 1em;
  color: #222;
}
code, pre {
  background: #e3eafc;
  color: #0a3d62;
  border-radius: 4px;
  padding: 2px 6px;
  font-family: 'Fira Mono', monospace;
}
hr {
  border: 1px solid #218c5a;
}
</style>

<!-- Para una mejor visualización de las fórmulas matemáticas, este documento utiliza sintaxis LaTeX compatible con MathJax/KaTeX. Al convertir a PDF, asegúrate de que el conversor soporte MathJax o KaTeX para que las ecuaciones se rendericen correctamente. -->

# Algoritmo de Bayes Ingenuo  
**Guía Teórica y Práctica**  
*Universidad Nacional Guillermo Brown*

---

## 1. Introducción

El **algoritmo de Bayes Ingenuo** (Naive Bayes) es un método de clasificación supervisada basado en el Teorema de Bayes, ampliamente utilizado en estadística, machine learning y minería de datos. Su simplicidad, eficiencia y buenos resultados en problemas reales lo convierten en una herramienta fundamental para estudiantes y profesionales.

---

## 2. Explicación Teórica

### 2.1. Teorema de Bayes


El Teorema de Bayes describe la probabilidad de que ocurra un evento, dado que ha ocurrido otro evento. Matemáticamente:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

Donde:

- $P(A \mid B)$: Probabilidad de A dado B.
- $P(B \mid A)$: Probabilidad de B dado A.
- $P(A)$: Probabilidad previa de A.
- $P(B)$: Probabilidad previa de B.

### 2.2. Supuesto de Independencia

El algoritmo se denomina "ingenuo" porque **asume que todas las variables predictoras son independientes entre sí**, lo cual rara vez es cierto en la práctica, pero simplifica enormemente los cálculos.

### 2.3. Funcionamiento del Algoritmo

1. **Entrenamiento:**  
   Se calcula la probabilidad de cada clase y la probabilidad de cada atributo dado cada clase, a partir de los datos de entrenamiento.

2. **Clasificación:**  
   Para una nueva instancia, se calcula la probabilidad de pertenecer a cada clase y se asigna la clase con mayor probabilidad.

3. **Fórmula General:**

$$
P(C \mid X) \propto P(C) \prod_{i=1}^{n} P(x_i \mid C)
$$

Donde:

- $C$: Clase.
- $X = (x_1, x_2, ..., x_n)$: Vector de atributos.

---

## 3. Requerimientos para Ejecutar el Algoritmo

### 3.1. Requisitos de Software

- **Python 3.x**  
- Bibliotecas recomendadas:
  - `numpy`
  - `pandas`
  - `scikit-learn` (opcional, para comparación)
  - `matplotlib` (opcional, para visualización)
- Jupyter Notebook (opcional, para ejecución interactiva)

### 3.2. Archivos Necesarios

- **Código:**  
  - `Codigo-bayes-ingenuo.ipynb`  
  - `algoritmo-de-bayes.ipynb`  
  - `algoritmos-de-bayes-2.ipynb`
- **Datos:**  
  - `ddbb-correr-bayes-ingenuo.csv`  
  - `Datos (2)/Datos/Wine.csv`

### 3.4. Crea un entorno virtual:
```bash
python -m venv env       # En Windows
python3 -m venv env      # En Linux/Mac

env\Scripts\activate     # En Windows
source env/bin/activate  # En Linux/Mac
```
---

### 3.4. Instalación de Dependencias

```bash
pip install numpy pandas matplotlib scikit-learn
```


## 4. Ejecución del Algoritmo

### 4.1. Usando Jupyter Notebook

1. Abre una terminal y navega a la carpeta `bayes-ingenuo/`.
2. Ejecuta:
   ```bash
   jupyter notebook
   ```
3. Abre el archivo `Codigo-bayes-ingenuo.ipynb` (o el que corresponda).
4. Ejecuta las celdas en orden. Si es necesario, ajusta la ruta de los archivos de datos.

### 4.2. Desde Consola (opcional)

Si tienes el código en un script `.py`, puedes ejecutarlo así:

```bash
python nombre_del_script.py
```

---

## 5. Interpretación de Resultados

Al finalizar la ejecución, el algoritmo mostrará:
- **Predicciones:** Clase asignada a cada instancia.
- **Matriz de confusión:** Tabla que compara predicciones vs. valores reales.
- **Métricas:** Precisión, recall, F1-score, exactitud, etc.

**¿Cómo interpretar?**

- Una alta exactitud indica buen desempeño.
- Analiza la matriz de confusión para ver en qué clases se equivoca más el modelo.
- Observa las probabilidades: si son muy bajas, puede haber problemas de datos o supuestos.



### ¿Cómo interpretar los resultados?

- **Exactitud (accuracy):** Proporción de predicciones correctas sobre el total. Un valor alto indica buen desempeño general, pero puede ser engañoso si las clases están desbalanceadas.
- **Matriz de confusión:** Permite ver en qué clases el modelo acierta o se equivoca. Por ejemplo:

  |            | Predicho: Clase A | Predicho: Clase B |
  |------------|-------------------|-------------------|
  | Real: A    |        50         |        10         |
  | Real: B    |        5          |        35         |

  Aquí, el modelo confunde 10 veces la clase A con B y 5 veces la B con A.

- **Precisión y recall:**
  - *Precisión:* De los elementos predichos como positivos, ¿cuántos lo son realmente?
  - *Recall:* De los elementos realmente positivos, ¿cuántos fueron detectados?
- **F1-score:** Media armónica entre precisión y recall, útil si hay desbalance de clases.
- **Probabilidades:** Si el modelo da probabilidades bajas para todas las clases, puede indicar que los datos no cumplen el supuesto de independencia o que hay ruido.

**Errores comunes:**
- Sobreajuste: El modelo aprende demasiado los datos de entrenamiento y falla en nuevos datos.
- Subajuste: El modelo es demasiado simple y no capta patrones relevantes.
- Datos desbalanceados: Si una clase es mucho más frecuente, el modelo puede sesgarse.

**Recomendación:** Siempre analiza la matriz de confusión y las métricas, no solo la exactitud.

---

## 6. Visualización de Resultados

Puedes agregar gráficos para analizar el desempeño:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Suponiendo que tienes y_test y y_pred
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Matriz de Confusión')
plt.show()
```

También puedes graficar la distribución de probabilidades, importancia de atributos, etc.

---

## 7. Cómo Cambiar Parámetros del Algoritmo

### 7.1. En código propio

Si implementas Bayes Ingenuo desde cero, puedes modificar:

- **Tipo de distribución:** Gaussiana, multinomial, bernoulli.
- **Atributos usados:** Selecciona o descarta columnas del dataset.
- **Tamaño del conjunto de entrenamiento/prueba.**

Ejemplo:

```python
# Cambiar columnas usadas
atributos = ['col1', 'col2', 'col3']
X = datos[atributos]

# Cambiar proporción de entrenamiento/prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

### 7.2. Usando scikit-learn

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Cambiar el tipo de modelo
modelo = GaussianNB()  # o MultinomialNB(), BernoulliNB()
modelo.fit(X_train, y_train)
```

---

## 8. Ejemplo de Interfaz Interactiva

Puedes crear una interfaz web sencilla con Streamlit para cargar datos y cambiar parámetros:

```python
import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('Clasificador Bayes Ingenuo')
uploaded_file = st.file_uploader('Sube tu archivo CSV', type='csv')
if uploaded_file:
  data = pd.read_csv(uploaded_file)
  st.write(data.head())
  atributos = st.multiselect('Selecciona atributos', data.columns.tolist())
  clase = st.selectbox('Selecciona la variable de clase', data.columns.tolist())
  if atributos and clase:
    X = data[atributos]
    y = data[clase]
    test_size = st.slider('Proporción de test', 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    st.write('Exactitud:', accuracy_score(y_test, y_pred))
```

---

## 8.1. ¿Se puede crear una SPA (Single Page Application) para ejecutar el algoritmo?

¡Sí! Es posible crear una aplicación web moderna (SPA) que permita cargar datos, ejecutar el algoritmo y visualizar resultados de forma interactiva.

### Arquitectura recomendada:

- **Frontend (SPA):**
  - Frameworks: React, Vue.js o Angular.
  - Permite cargar archivos, seleccionar parámetros, mostrar métricas y gráficos.
- **Backend (API):**
  - Python (Flask, FastAPI o Django REST Framework).
  - Expone endpoints para procesar los datos, entrenar el modelo y devolver resultados.
- **Comunicación:**
  - El frontend envía los datos y parámetros al backend vía API REST (JSON).
  - El backend ejecuta el algoritmo y responde con predicciones, métricas y gráficos (pueden enviarse como imágenes o datos para graficar en el frontend).

### Flujo típico de uso:
1. El usuario sube un archivo CSV desde la SPA.
2. Selecciona variables, parámetros y tipo de modelo.
3. La SPA envía los datos al backend.
4. El backend ejecuta Bayes Ingenuo y responde con resultados.
5. La SPA muestra métricas, matriz de confusión y gráficos interactivos.

### Ventajas de una SPA:
- Experiencia de usuario fluida y moderna.
- Permite cambiar parámetros y ver resultados en tiempo real.
- Puede integrarse con autenticación, exportación de reportes, etc.

**Ejemplo de stack:**
- Frontend: React + Material UI
- Backend: FastAPI (Python) + scikit-learn

**Recurso útil:**
- [FastAPI + React: ejemplo de integración](https://fastapi.tiangolo.com/advanced/async-react/)

---

## 9. Consejos Prácticos y Recomendaciones

- Verifica la calidad de los datos antes de entrenar.
- Prueba diferentes variantes de Bayes según el tipo de datos.
- Usa validación cruzada para evaluar el modelo.
- Documenta los cambios de parámetros y resultados obtenidos.
- Si usas una interfaz, permite exportar los resultados y gráficos.

---

## 10. Recursos y Referencias

- [Documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Teorema de Bayes - Wikipedia](https://es.wikipedia.org/wiki/Teorema_de_Bayes)
