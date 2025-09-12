# Proyecto: Algoritmo de Bayes Ingenuo

## Estructura de carpetas
- `teoria/`: Documentos teóricos y PDFs
- `scripts/`: Scripts de Python para análisis y visualización
- `datos/`: Archivos de datos CSV
- `visualizaciones/`: Resultados y gráficos generados

## Cómo usar los scripts

### 1. Visualización web interactiva
Usa la app de Streamlit para cargar tus datos y visualizar resultados:
```bash
streamlit run scripts/app.py
```
- Sube tu archivo CSV.
- Selecciona la variable de clase y los atributos.
- Visualiza métricas, gráficos y explicaciones automáticas.

### 2. Ejecución por consola
Ejecuta el script principal para ver resultados en terminal:
```bash
python scripts/bayes_ingenuo.py
```
- Usa el archivo `datos/ddbb-correr-bayes-ingenuo.csv` por defecto.
- Imputa valores faltantes automáticamente.
- Muestra exactitud, matriz de confusión y reporte de clasificación.

### 3. Análisis de valores faltantes
Para revisar los datos antes de entrenar el modelo:
```bash
python scripts/datos_faltantes.py
```
- Muestra columnas y filas con valores faltantes.

## Recomendaciones para visualizar correctamente
- Usa la app web para una experiencia interactiva y visual.
- Si prefieres Jupyter, puedes convertir los scripts en notebooks y agregar celdas explicativas.
- Guarda los gráficos generados en la carpeta `visualizaciones/`.

## Notas
- Si tienes archivos teóricos en PDF o Markdown, consúltalos en la carpeta `teoria/`.
- Si tienes nuevos datos, colócalos en la carpeta `datos/` y actualiza la ruta en los scripts si es necesario.

---
**¡Listo para aprender y experimentar con Bayes Ingenuo de forma ordenada y visual!**
