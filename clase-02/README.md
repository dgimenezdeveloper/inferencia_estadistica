# Proyecto: Análisis Discriminante Lineal y Cuadrático (LDA & QDA)

## Estructura

```
clase-02/
    teoria/
        discriminante_lineal_cuadratico.md
    scripts/
        lda_ejemplo.py
        qda_ejemplo.py
        app_discriminante.py
    datos/
        # Aquí puedes agregar datasets
    otros/
        discriminante_l_q.pdf
        chats.txt
        videos/
    README.md
    requirements.txt
```

## Instalación de dependencias

Requiere Python 3.8+ y pip. Ejecuta:

```bash
pip install -r requirements.txt
```
### Dependencias específicas por clase

Si trabajas con scripts o apps de otras clases, instala sus dependencias usando el archivo `requirements.txt` correspondiente. Por ejemplo:

```bash
pip install -r bayes-ingenuo/requirements.txt
```

Esto asegura que cada conjunto de algoritmos tenga sus librerías necesarias y evita conflictos.

## Ejecución de scripts

### Ejemplo LDA
```bash
python scripts/lda_ejemplo.py
```

### Ejemplo QDA
```bash
python scripts/qda_ejemplo.py
```

### App interactiva con Streamlit
```bash
streamlit run scripts/app_discriminante.py
```

## Notas
- El dataset Iris se carga automáticamente desde scikit-learn.
- Puedes modificar los scripts para usar otros datasets.
- Consulta la teoría en `teoria/discriminante_lineal_cuadratico.md`.

## Referencias
- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [Streamlit documentation](https://docs.streamlit.io/)
