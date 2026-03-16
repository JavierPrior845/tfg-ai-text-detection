# TFG: Detección Multimodal de Noticias Sintéticas mediante Modelos de Lenguaje y Visión

## # Preambulo

### Agradecimientos

* Espacio dedicado a tutores, familia y colaboradores.

### Resumen

* **Contexto:** Crecimiento de los VLMs y la necesidad de verificar la autoría de la información.
* **Metodología:** Creación de un dataset multimodal y evaluación mediante arquitecturas Encoder-only, Decoder-only y Encoder-Decoder.
* **Resultados:** Breve mención a la eficacia del modelo DeBERTa-v3 frente a técnicas de parafraseo y generación de titulares.

### Declaración de originalidad

* Certificación de autoría propia del trabajo.

---

## # 1. Introducción

* **1.1. Motivación:** El auge de la desinformación generada por IA (Deepfakes de texto e imagen).
* **1.2. Contexto del Problema:** Dificultad de detección en modelos de última generación (Gemini 2.0 Flash-Lite).
* **1.3. Estructura del Documento:** Guía de los capítulos que componen el trabajo.

---

## # 2. Estado del Arte

* **2.1. Modelos de Lenguaje de Gran Escala (LLMs):** Evolución desde BERT hasta Llama 3.1.
* **2.2. Modelos Multimodales (VLMs):** Arquitecturas que integran texto y visión.
* **2.3. Técnicas de Detección de Contenido Sintético:**
* Métodos estadísticos (Ley de Zipf, Perplejidad).
* Métodos basados en aprendizaje supervisado (Clasificadores Transformer).


* **2.4. Datasets Existentes:** Comparativa con el dataset propio desarrollado en este trabajo.

---

## # 3. Análisis de Objetivos y Metodología

* **3.1. Objetivos del Trabajo:**
* General: Desarrollar un clasificador multimodal robusto.
* Específicos: Recolección de datos reales, generación sintética controlada y análisis de interpretabilidad (XAI).


* **3.2. Metodología de Investigación:**
* Diseño experimental siguiendo el ciclo de vida de un proyecto de ML.
* Aproximación empírica mediante *Hyperparameter Tuning* (Sweeps Bayesianos).



---

## # 4. Diseño y Resolución del Trabajo

* **4.1. Adquisición y Construcción del Dataset:**
* Scraping de noticias reales (NewsData API) y limpieza con `trafilatura`.
* Generación de titulares sintéticos con Gemini 2.0 Flash-Lite basado en contexto.


* **4.2. Ingeniería de Datos y Post-procesamiento:**
* Prevención de *Data Leakage* mediante `GroupShuffleSplit`.
* Técnicas de barajado (*shuffling*) y sanitización de texto.


* **4.3. Análisis Exploratorio de Datos (EDA) Avanzado:**
* Análisis de N-gramas y Solapamiento de Jaccard.
* Estudio de Perplejidad ($PPL$) con GPT-2.
* Visualización del espacio latente (UMAP/t-SNE).


* **4.4. Fase de Modelado (Entrenamiento):**
* Encoder-only: Configuración de DeBERTa-v3.
* Decoder-only: Inferencia y entropía con Llama 3.1.
* Pruebas Multimodales: Clasificación combinada de texto e imagen.


* **4.5. Resultados y Evaluación:**
* Métricas: F1-Score, Precision-Recall y AUC-ROC.
* Análisis de Explicabilidad (SHAP/LIME): Por qué el modelo detecta la IA.



---

## # 5. Conclusión y Vías Futuras

* **5.1. Conclusiones:** Resumen de hallazgos sobre la predictibilidad de Gemini frente al estilo periodístico humano.
* **5.2. Limitaciones:** Desafíos encontrados en el entrenamiento y sesgos del dataset.
* **5.3. Trabajo Futuro:** Escalado a modelos de mayor tamaño, análisis de "Burstiness" y robustez frente a ataques adversarios.
