# Inferencia Estadística - UNAB

Este repositorio contiene los materiales y scripts para la materia de Inferencia Estadística. Incluye implementaciones de algoritmos de Bayes Ingenuo, Análisis Discriminante Lineal y Cuadrático, y Análisis de Componentes Principales (PCA).

## Requisitos

- Python 3.8 o superior
- pip
- Recomendado: crear un entorno virtual


## Instalación de dependencias y entorno

Desde la raíz del proyecto, ejecuta:

```bash
python -m venv env
source env/bin/activate
pip install -r scripts/requirements.txt
```

Si no existe un `requirements.txt` global, instala manualmente:

```bash
pip install streamlit scikit-learn matplotlib seaborn pandas numpy plotly
```

## Estructura de carpetas

- `bayes-ingenuo/`: Algoritmo de Bayes Ingenuo y scripts relacionados
- `clase-02/`: Análisis Discriminante Lineal y Cuadrático
- `clase-03/`: Análisis de Componentes Principales (PCA)
- `scripts/`: App principal y utilidades
- `clases/`: Material multimedia y textos (no se suben al repo)



## Ejecución de la app principal (Streamlit)

La app principal se encuentra en `scripts/app_integrado.py` y permite ejecutar LDA, QDA, Bayes Ingenuo y PCA desde una sola interfaz interactiva.

### 1. Instala las dependencias necesarias

Activa tu entorno virtual y ejecuta:

```bash
pip install -r scripts/requirements.txt
# O instala manualmente los paquetes necesarios
```

### 2. Ejecuta la app con Streamlit

Desde la raíz del proyecto:

```bash
streamlit run scripts/app_integrado.py
```

Esto abrirá la app en tu navegador. Si no se abre automáticamente, copia la URL que aparece en la terminal.

### 3. ¿Qué puedes hacer en la app?

- Elegir entre Análisis Discriminante (LDA/QDA/Bayes Ingenuo) y Reducción de Dimensiones (PCA)
- Subir tu propio archivo CSV o usar los datasets de ejemplo (`datos/`)
- Visualizar y comparar resultados de PCA con y sin escalado
- Explorar visualizaciones interactivas 2D/3D
- Recibir interpretaciones automáticas y recomendaciones sobre tus datos

### 4. Notas y recomendaciones

- Para visualizaciones 3D y FAQ interactivo, asegúrate de tener `plotly` instalado.
- Si tienes problemas con dependencias, revisa los mensajes de error en la terminal y ejecuta `pip install` para instalar los paquetes faltantes.

---


## Ejecución de scripts individuales

Cada carpeta contiene scripts independientes para probar cada algoritmo:

- Bayes Ingenuo:
  ```bash
  python bayes-ingenuo/scripts/bayes_ingenuo.py
  ```
- Análisis Discriminante:
  ```bash
  python clase-02/scripts/lda_ejemplo.py
  python clase-02/scripts/qda_ejemplo.py
  ```
- PCA:
  ```bash
  python clase-03/scripts/pca_ejemplo.py
  ```


## Notas

- Los archivos de datos (`.csv`) están incluidos en la carpeta `datos/` y subcarpetas específicas de cada algoritmo.
- Los videos y archivos `.txt` de la carpeta `clases` no se suben al repositorio.
- Si tienes dudas, revisa los archivos `README.md` de cada subcarpeta para detalles específicos.

---

**¡Listo para comenzar!**

---

**¡Listo para comenzar!**
