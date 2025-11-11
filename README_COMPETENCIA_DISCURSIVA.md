# 📋 Documentación para Competencia Discursiva - UNAB

Este directorio contiene todos los materiales preparados para la **Competencia Discursiva** de la materia **Inferencia Estadística y Reconocimiento de Patrones** de la Universidad Nacional Guillermo Brown.

---

## 📂 Archivos Disponibles

### 1. **Guión Completo** 
📄 `guion_competencia_discursiva.md`

**Descripción:** Guión detallado y completo para la presentación oral (15-20 minutos).

**Estructura:**
- ✅ Introducción (objetivo, contexto, herramientas)
- ✅ Desarrollo (10 pasos metodológicos con organizadores textuales)
- ✅ Conclusión (resumen, técnica óptima, lecciones aprendidas)
- ✅ Anexos (organizadores textuales, notas, FAQ)

**Características:**
- Cumple 100% con los requisitos del texto modelo
- Incluye organizadores textuales explícitos en cada sección
- Profundidad técnica adecuada para evaluación académica
- Justificaciones metodológicas rigurosas

**Uso recomendado:** 
- Documento de referencia principal
- Base para preparar la presentación
- Material de estudio completo

---

### 2. **Guión Conciso para Presentación**
📄 `guion_presentacion_oral_conciso.md`

**Descripción:** Versión reducida y memorizable del guión (10-12 minutos).

**Ventajas:**
- ✅ Estructura simplificada y clara
- ✅ Puntos clave destacados con emojis
- ✅ Tiempos estimados por sección
- ✅ Tips de presentación oral incluidos
- ✅ Checklist de organizadores textuales

**Características:**
- Fácil de memorizar
- Ritmo y pausas sugeridas
- Gestos recomendados
- Énfasis vocal marcado

**Uso recomendado:**
- Memorización para la presentación en vivo
- Tarjetas de apoyo (cue cards)
- Ensayo con cronómetro

---

### 3. **Guía de Diapositivas**
📄 `guia_diapositivas_sugeridas.md`

**Descripción:** Estructura completa sugerida para 19 slides de PowerPoint/Google Slides.

**Contenido de cada slide:**
- Título y contenido textual
- Visualizaciones sugeridas
- Paleta de colores recomendada
- Tipografía profesional
- Notas de diseño

**Slides incluidas:**
1. Título
2. Objetivo del trabajo
3. Herramientas y tecnologías
4. Características del dataset
5. Preprocesamiento - Transformaciones
6. Pipeline de preprocesamiento
7. Resultados comparativos
8. Impacto de variables categóricas
9. Poder predictivo de categóricas
10. Validación metodológica
11. Implementación - App Web
12. Conclusiones - Técnica óptima
13. Lecciones aprendidas
14. Recomendaciones de implementación
15. Impacto empresarial proyectado
16. Demostración en vivo
17. Preguntas frecuentes anticipadas
18. Cierre
19. Contacto y referencias

**Uso recomendado:**
- Crear presentación visual profesional
- Apoyo durante la exposición oral
- Material para compartir con audiencia

---

### 4. **Documento Oral Original**
📄 `documento_oral_completo.md`

**Descripción:** Documento extenso y detallado usado para el parcial (incluye análisis completo).

**Características:**
- Análisis exhaustivo de todos los algoritmos
- Comparativas con y sin PCA
- Justificación del impacto de variables categóricas
- Resultados técnicos completos
- Recomendaciones empresariales detalladas

**Uso recomendado:**
- Consulta técnica profunda
- Respaldo para preguntas complejas
- Material de estudio complementario

---

## 🎯 Estructura del Guión según Requisitos

Todos los guiones cumplen con la estructura requerida por la cátedra:

### **INTRODUCCIÓN**
1. **Objetivo del trabajo** - ¿Qué se quiere lograr?
2. **Herramientas utilizadas** - Lenguajes, bibliotecas, plataforma

Organizadores: 
- "El objetivo de este trabajo práctico es..."
- "En este contexto..."
- "Para llevar a cabo este análisis..."

### **DESARROLLO**
Pasos organizados secuencialmente:

1. **Preparación de datos** - Carga y exploración del dataset
2. **Verificación de calidad** - Valores nulos, outliers, tipos
3. **Transformación de variables** - Encoding de categóricas
4. **Escalado de datos** - StandardScaler, PCA opcional
5. **Particionamiento** - Train/test split estratificado
6. **Entrenamiento de algoritmos** - LDA, QDA, Bayes, SVM
7. **Evaluación de resultados** - Métricas y validación cruzada
8. **Análisis comparativo** - Ranking y descubrimientos
9. **Validación metodológica** - Corrección de sobreajuste
10. **Implementación** - Deployment en Streamlit

Organizadores:
- "En primer lugar..."
- "A continuación..."
- "Posteriormente..."
- "En esta etapa..."
- "Seguidamente..."
- "A partir de este punto..."
- "Primero...", "En segundo lugar...", "Finalmente..."
- "Una vez completados..."
- "Adicionalmente..."
- "Para profundizar..."
- "Por último..."

### **CONCLUSIÓN**
1. **Resumen de conclusiones generales** - ¿Qué se descubrió?
2. **Técnica más útil y justificación** - SVM RBF 94.8% accuracy
3. **Lecciones aprendidas** - Importancia del análisis iterativo

Organizadores:
- "En resumen..."
- "La técnica más útil es..."
- "Es importante destacar..."
- "Además..."
- "Para finalizar..."
- "En conclusión..."

---

## 📊 Resultados Clave a Comunicar

### Ranking Final de Algoritmos (Dataset Completo - Sin PCA)
```
1. SVM RBF          94.8% ⭐️ GANADOR ABSOLUTO
2. QDA              90.5% 🥈 EXCELENTE ALTERNATIVA
3. SVM Linear       76.0% 🥉 MEJORA NOTABLE
4. LDA              75.7% 📈 LÍNEA BASE
5. Bayes Ingenuo    71.1% ⚠️ NO RECOMENDADO
```

### Mejora con Variables Categóricas
```
SVM RBF:     85.5% → 94.8%  (+9.4%) 🚀 MAYOR MEJORA
QDA:         85.8% → 90.5%  (+5.5%) ✅
SVM Linear:  70.1% → 76.0%  (+8.3%) ��
LDA:         73.9% → 75.7%  (+1.8%) ✅
Bayes:       79.2% → 71.1%  (-10.3%) ⚠️ EMPEORA
```

### Impacto Empresarial
```
ROI Anual: $900K - $1.5M (empresa 15,000 empleados)
Reducción de rotación: 20-30%
Identificación correcta: 94.8% de riesgos
Falsos negativos: Solo 5.2%
```

---

## 🎤 Recomendaciones para la Presentación

### Antes de Presentar
1. ✅ Memorizar el guión conciso
2. ✅ Preparar diapositivas según la guía
3. ✅ Ensayar con cronómetro (ajustar a tiempo límite)
4. ✅ Probar la demo en vivo de la app
5. ✅ Revisar FAQ anticipadas
6. ✅ Preparar backup (sin internet para demo)

### Durante la Presentación
- 🎯 Mantener contacto visual con la audiencia
- 🎯 Usar organizadores textuales claramente
- 🎯 Pausar después de números importantes
- 🎯 Señalar visualizaciones relevantes
- 🎯 Controlar el tiempo (15-20 min máximo)

### Después de la Presentación
- 💡 Responder preguntas con seguridad
- 💡 Usar el documento completo como respaldo técnico
- 💡 Compartir URL de la app si hay interés
- 💡 Agradecer y cerrar profesionalmente

---

## 🔗 Recursos Adicionales

### Aplicación Web Deployada
**URL:** https://inferencia-estadistica-unab.streamlit.app/

**Funcionalidades:**
- Carga de datasets personalizados
- Comparativa interactiva de algoritmos (LDA, QDA, Bayes, SVM)
- Visualizaciones en tiempo real
- Matrices de confusión
- Curvas ROC multiclase
- Análisis exploratorio automatizado
- Predicciones individuales

### Repositorio GitHub
Todos los archivos están versionados y disponibles en el repositorio del proyecto.

---

## 📚 Referencias Metodológicas

### Guía Oficial de la Cátedra
📎 **Texto modelo:** `Guion para una presentación oral.pdf`
- Estructura: Introducción → Desarrollo → Conclusión
- Uso de organizadores textuales
- Jerarquización de información

### Videos de Referencia
1. **Video 1:** Deconstrucción del guion (introducción, desarrollo, conclusión)
2. **Video 2:** Proceso de escritura (planificación, redacción, revisión)

---

## ✅ Checklist de Entrega

**Documentos para revisar antes de la presentación:**

- [x] Guión completo estructurado (`guion_competencia_discursiva.md`)
- [x] Guión conciso memorizado (`guion_presentacion_oral_conciso.md`)
- [x] Diapositivas preparadas según guía (`guia_diapositivas_sugeridas.md`)
- [ ] Cronometraje de ensayo (ajustar a 15-20 min)
- [ ] Demo de la app funcionando
- [ ] Respuestas a FAQ preparadas
- [ ] Contacto visual y lenguaje corporal ensayados
- [ ] Backup plan si falla tecnología

---

## 💡 Puntos Críticos a Recordar

### Mensajes Clave
1. **SVM RBF alcanza 94.8% de accuracy** (validación cruzada rigurosa)
2. **Variables categóricas son transformadoras** (+9.4% en SVM RBF)
3. **No todos los algoritmos se benefician igual** (Bayes empeora -10.3%)
4. **Validación rigurosa es fundamental** (evitar sobreajuste optimista)
5. **Impacto empresarial real** ($900K-1.5M ROI anual)

### Errores a Evitar
- ❌ NO leer las diapositivas textualmente
- ❌ NO exceder el tiempo límite
- ❌ NO omitir organizadores textuales
- ❌ NO usar jerga sin explicar
- ❌ NO ignorar a la audiencia

---

## 📞 Contacto y Soporte

**Para dudas sobre el contenido:**
- Revisar el documento oral completo (`documento_oral_completo.md`)
- Consultar los informes técnicos específicos (LDA, QDA, SVM, etc.)
- Probar la aplicación web interactiva

**Para ajustes al guión:**
- Los archivos están en formato Markdown (fácil edición)
- Se pueden personalizar según tiempo disponible
- Mantener la estructura (introducción/desarrollo/conclusión)

---

## 🎓 Sobre Este Trabajo

**Materia:** Inferencia Estadística y Reconocimiento de Patrones  
**Universidad:** Universidad Nacional Guillermo Brown (UNAB)  
**Tipo:** Trabajo Práctico - Competencia Discursiva  
**Tema:** Análisis Comparativo de Algoritmos de Clasificación  
**Caso de estudio:** Predicción de Rotación de Personal  

**Algoritmos evaluados:**
- Análisis Discriminante Lineal (LDA)
- Análisis Discriminante Cuadrático (QDA)
- Bayes Ingenuo (Gaussian Naive Bayes)
- Máquinas de Vectores de Soporte - Lineal (SVM Linear)
- Máquinas de Vectores de Soporte - RBF (SVM RBF)

**Técnicas de preprocesamiento:**
- Análisis de Componentes Principales (PCA)
- Escalado con StandardScaler
- Codificación de variables categóricas (Label Encoding, One-Hot Encoding)

---

**¡Éxito en la presentación! 🎉**

---

_Última actualización: [Fecha actual]_
