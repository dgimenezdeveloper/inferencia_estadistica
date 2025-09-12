# Inferencia Estadística - UNAB

Este repositorio contiene los materiales y scripts para la materia de Inferencia Estadística. Incluye implementaciones de algoritmos de Bayes Ingenuo, Análisis Discriminante Lineal y Cuadrático, y Análisis de Componentes Principales (PCA).

## Requisitos

- Python 3.8 o superior
- pip
- Recomendado: crear un entorno virtual

## Instalación de dependencias

Desde la raíz del proyecto, ejecuta:

```bash
python -m venv env
source env/bin/activate
pip install -r bayes-ingenuo/requirements.txt
pip install -r clase-02/requirements.txt
pip install -r clase-03/requirements.txt
```

## Estructura de carpetas

- `bayes-ingenuo/`: Algoritmo de Bayes Ingenuo y scripts relacionados
- `clase-02/`: Análisis Discriminante Lineal y Cuadrático
- `clase-03/`: Análisis de Componentes Principales (PCA)
- `scripts/`: App principal y utilidades
- `clases/`: Material multimedia y textos (no se suben al repo)

## Ejecución de la app principal

La app principal se encuentra en `scripts/app_integrado.py` y permite correr los tres algoritmos desde una sola interfaz.

Para ejecutarla:

```bash
python scripts/app_integrado.py
```

Sigue las instrucciones en pantalla para seleccionar el algoritmo y el dataset.

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

- Los archivos de datos (`.csv`) están incluidos en las carpetas correspondientes.
- Los videos y archivos `.txt` de la carpeta `clases` no se suben al repositorio.
- Si tienes dudas, revisa los archivos `README.md` de cada subcarpeta para detalles específicos.

---

**¡Listo para comenzar!**
