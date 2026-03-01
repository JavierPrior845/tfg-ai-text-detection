# Guía de Análisis Exploratorio de Datos (EDA) para Detección de Texto Generado por IA

Este documento detalla el flujo de análisis exploratorio diseñado para el estudio de detección de titulares generados por inteligencia artificial frente a titulares reales. Se explica el propósito de cada fase, qué se espera encontrar y cómo interpretar los resultados, además de ofrecer una visión general de la solidez de este enfoque en el ámbito del Procesamiento del Lenguaje Natural (PLN).

---

## Flujo del Análisis Exploratorio

### 1. Estadísticas de Superficie y Análisis Diferencial
**Scripts:** `dataAnalysisTitle.py` y `diferentialAnalysisTitle.py`

*   **Propósito:** Esta fase inicial busca entender las características más básicas y superficiales del texto. Analiza métricas como la longitud de los titulares, la cantidad de palabras, la longitud media de las palabras y la presencia de caracteres atípicos o "artefactos" (como dobles espacios, secuencias de escape inusuales, etc.).
*   **¿Qué se espera?** 
    *   Frecuentemente, los modelos de lenguaje tienden a generar textos con una estructura más uniforme y predecible. Es posible que los titulares de IA tengan una desviación estándar menor en longitud que los reales.
    *   El modelo de IA podría, dependiendo del prompt, ser más prolijo o, por el contrario, más robótico y estructurado. 
    *   El análisis diferencial busca errores de generación: caracteres raros, puntuación excesiva o mal colocada que los humanos raramente producen de forma sistemática.
*   **Interpretación:** 
    *   Si los histogramas (`eda_word_distribution_titles.png`) muestran curvas muy diferentes (p. ej., la IA produce siempre 8-10 palabras, mientras el humano oscila entre 5 y 15), esto es una señal clara de sesgo generativo.
    *   Si el análisis diferencial arroja "artefactos" muy presentes en la IA, pueden usarse como características rudimentarias pero efectivas para un modelo de clasificación simple o como heurísticas.

### 2. Estilometría de N-gramas
**Script:** `ngramAnalysis.py`

*   **Propósito:** Analiza la frecuencia de secuencias de 1, 2 o N palabras (n-gramas). Esto captura el estilo, el vocabulario preferido y las estructuras sintácticas locales repetitivas.
*   **¿Qué se espera?** 
    *   Los modelos de IA tienden a sufrir de "colapso de vocabulario" o a usar ciertas frases hechas con mucha frecuencia (p. ej., "En un giro sorprendente de los eventos", "Descubre cómo..."). 
    *   Los humanos suelen tener una "cola larga" (distribución de Zipf) mucho más rica en términos de vocabulario raro o combinaciones inusuales.
*   **Interpretación:** 
    *   Busca n-gramas que estén sobre-representados en la clase IA frente a la clase Real (y viceversa). Esto ayuda a entender si la IA está abusando de ciertas fórmulas periodísticas genéricas, lo cual es un excelente insumo para entender qué delata a la IA.

### 3. Diversidad Léxica (Type-Token Ratio - TTR)
**Script:** `ttrAnalysisTitle.py`

*   **Propósito:** Mide la riqueza del vocabulario utilizado evaluando la proporción de palabras únicas (types) respecto al total de palabras (tokens) en un titular. Un TTR más alto indica mayor variabilidad léxica.
*   **¿Qué se espera?** 
    *   Los modelos de lenguaje generativos, al intentar maximizar la probabilidad y seguir patrones comunes, tienden a reciclar un conjunto más limitado de vocabulario, resultando en un TTR más bajo.
    *   Los humanos, en cambio, emplean un vocabulario más rico y variado para atraer la atención en los titulares, lo que se traduce en un TTR mayor.
*   **Interpretación:** 
    *   Si los gráficos (boxplot e histograma de densidad) muestran que los titulares humanos tienen un TTR significativamente desplazado hacia la derecha (valores más altos) en comparación con los de IA, esta métrica será un excelente indicador discriminativo para diferenciar ambas fuentes.

### 4. Perplejidad (Perplexity)
**Script:** `perplexityAnalysisTitle.ipynb`

*   **Propósito:** La perplejidad es una métrica intrínseca de los modelos de lenguaje que mide qué tan "sorprendido" está el modelo (en este caso, GPT-2 u otro) al leer una secuencia de palabras. Una perplejidad baja significa que el texto es muy predecible para el modelo.
*   **¿Qué se espera?** 
    *   **Resultado clave:** El texto generado por IA suele tener una perplejidad **menor** (es más predecible) que el texto escrito por humanos. Esto se debe a que la IA generativa produce secuencias maximizando la probabilidad condicional de las palabras, tendiendo hacia el texto "promedio" o "más probable". Los humanos somos más creativos introduciendo ruido, metáforas o combinaciones impredecibles.
*   **Interpretación:** 
    *   El gráfico de distribución de perplejidad debería mostrar dos picos (una distribución bimodal) o al menos dos medias claramente separadas. 
    *   Si la perplejidad de los titulares de IA es mucho más baja y consistente que la de los humanos, esta es una característica fundamental para la detección.

### 5. Espacio Latente (Embeddings)
**Script:** `visualizingEmbeddingsTitle.py`

*   **Propósito:** Transforma el texto en vectores densos (embeddings) usando un modelo semántico (como SBERT) y reduce su dimensionalidad (con t-SNE o PCA) a 2D para visualizarlos. Esto capta el significado profundo y el contexto de la oración.
*   **¿Qué se espera?** 
    *   Idealmente, se espera ver clústeres separados. El modelo de lenguaje podría estar generando titulares que, aunque traten de temas diversos, comparten una "firma semántica" o un tono que los agrupa en el espacio latente.
    *   Los titulares humanos podrían observarse más dispersos (reflejando mayor diversidad estilística y semántica).
*   **Interpretación:** 
    *   Si el gráfico de t-SNE muestra dos colores (IA vs Real) relativamente separados o en regiones concentradas distintas de la gráfica, significa que los modelos de embeddings profundos son capaces de discernir la diferencia. Esto indica que un clasificador (como una red neuronal o una regresión logística sobre los embeddings) funcionará muy bien.
    *   Si están muy mezclados, la detección será un desafío mayor, requiriendo modelos más complejos o basándose más en las métricas de varianza y perplejidad.


