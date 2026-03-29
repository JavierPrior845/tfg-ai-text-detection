# TFG: Detección Multimodal de Noticias Sintéticas mediante Modelos de Lenguaje y Visión

## # Preámbulo

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

### 4.1. Adquisición y Construcción del Dataset

#### 4.1.1. Recolección de Noticias Reales (*Web Scraping*)

La adquisición de noticias reales se realiza mediante un pipeline automatizado compuesto por dos scripts complementarios, ubicados en `scraping/data_collection/`:

* **`collector.py`** — Script principal de ingestión. Consulta la API de [NewsData.io](https://newsdata.io) con paginación automática (`nextPage` token), cubriendo 6 categorías temáticas: *business, technology, science, politics, environment* y *top*. Para cada artículo candidato:
  1. Comprueba colisión de `article_id` contra un conjunto de IDs ya almacenados (O(1) lookup vía `set`).
  2. Extrae el contenido completo del artículo mediante `trafilatura.fetch_url()` + `trafilatura.extract()`, que limpia el HTML y devuelve texto plano de alta calidad.
  3. Filtra por longitud mínima (≥ 300 palabras) para descartar artículos superficiales.
  4. Persiste cada entrada válida en modo *append* al fichero `real_news.jsonl` con el esquema: `{article_id, title, content, word_count, image_url, category, source, pub_date, is_real: True}`.
  5. Implementa *checkpoint* del token de paginación en `last_token.txt` para reanudar ejecuciones interrumpidas.
  6. Aplica un *courtesy delay* de 1 segundo entre llamadas API para respetar los *rate limits*.

* **`collectorCategories.py`** — Variante mejorada con iteración por categoría. Rota entre las 6 categorías, rompiendo el bucle de una categoría ante errores HTTP (e.g., cuota agotada) y continuando con la siguiente. Incorpora `removeduplicate=1` como parámetro adicional de la API y manejo robusto de excepciones (`try/except`).

* **`scraping/Article-Web-Scraping/`** — Contiene un notebook experimental (`Article Web Scraping.ipynb`) y un script `app.py` utilizados en fases tempranas de prototipado del scraping.

**Resultado:** Fichero `real_news.jsonl` con noticias reales en inglés y su versión deduplicada `real_news_no_duplicates.jsonl`.

#### 4.1.2. Limpieza y Deduplicación

El script `EDA/dataCleaning.py` implementa un mecanismo de deduplicación basado en *hashing*:

* Genera un **hash MD5** del campo `content` (texto normalizado) de cada entrada.
* Compara contra un conjunto de hashes ya vistos (`seen_hashes`).
* Elimina entradas con contenido idéntico, independientemente de que tengan `article_id` distinto (duplicados cross-fuente).
* El análogo para titulares es `EDA/dataCleaningByTitle.py`, que opera sobre el campo `title`.

**Resultado:** Dataset filtrado sin duplicados semánticos.

#### 4.1.3. Generación de Contenido Sintético

El pipeline de generación se encuentra en `generation/` y opera en tres niveles de granularidad:

##### A) Generación de Texto Completo — `datasetGenerator.py`

Implementa la clase `DatasetGenerator` que orquesta el pipeline completo:

1. **Generación textual** (`generate_fake_text`): Envía el título y cuerpo de la noticia real a **Gemini 2.0 Flash-Lite** con un prompt de parafraseo que impone restricciones de longitud ($\pm 10\%$ del original) y densidad de detalle. Utiliza respuesta estructurada en JSON con schema Pydantic (`SyntheticNews: {headline, content, technique}`).
2. **Generación de imagen** (`generate_fake_image`): Usa el titular sintético como prompt para **FLUX** (vía Hugging Face `InferenceClient`) y guarda la imagen generada en `dataset/fake_images/{article_id}_fake.png`.
3. **Pipeline integrado** (`process_pipeline`): Para cada noticia real, genera la versión sintética (texto + imagen) de forma atómica — si falla cualquiera de los dos, no se incluye el par. Checkpoints mediante `_load_processed_ids()`.

> También incluye `TextOnlyGenerator` (generación solo de texto, sin imagen) e `ImageBackfiller` (generación retroactiva de imágenes para entradas que ya tienen texto sintético pero carecen de imagen).

**Dataset resultante:** `multimodal_dataset.jsonl` con esquema `{group_id, is_real, title, content, image_path, technique, model}`.

##### B) Generación Solo de Titulares — `datasetTitleGenerator.py`

Enfoque más ligero centrado exclusivamente en titulares:

1. Calcula la **longitud media en palabras** de los titulares reales del dataset (`_calculate_avg_title_length`).
2. Para cada noticia real, envía su `content` a **Gemini 2.0 Flash-Lite** con el prompt: *"Write a compelling and accurate headline of approximately N words"*.
3. Guarda pares emparejados `{group_id, title, is_real}` en `dataset/titles_data.jsonl`.

##### C) Enriquecimiento Multimodal (Imagen + Img-to-Text) — `imageProcessor.py`

La clase `ImageProcessor` enriquece `titles_data.jsonl` con información visual:

1. **Imagen Real:** Extrae la `image_url` del artículo original en `real_news_no_duplicates.jsonl` y la asigna como `img_path`.
2. **Imagen Sintética:** Genera una imagen con **FLUX** usando el titular sintético como prompt, guardándola en `dataset/fake_images/{group_id}_fake.png`.
3. **Img-to-Text (Captioning):** Utiliza **Gemini multimodal** para describir textualmente cada imagen:
   * Para URLs: `img_to_text_from_url()` envía la URL directamente al modelo.
   * Para archivos locales: `img_to_text_from_file()` hace upload inline del archivo.
4. Guarda el resultado en `img_text`, generando el campo de *caption* que alimentará los enfoques multimodales.

**Dataset resultante:** `titles_img_data.jsonl` con esquema `{group_id, title, is_real, img_path, img_text}` — 877 pares (1754 registros).

##### D) Utilidades Auxiliares

* **`generator.py`** — Prototipo de generación rápida de noticias sintéticas (prompt en castellano) para pruebas iniciales.
* **`extractSynthetic.py` / `extractSyntheticTitle.py`** — Scripts de backup que extraen únicamente las entradas sintéticas (`is_real == 0`) del dataset multimodal o de títulos, respectivamente, guardándolas en `synthetic_news.jsonl` / `synthetic_news_titles.jsonl`.
* **`repair_dataset.py`** — Script de reparación/sanitización del dataset.

---

### 4.2. Ingeniería de Datos y Post-procesamiento

* **Prevención de *Data Leakage*** mediante `GroupShuffleSplit`: Los pares real/sintético que comparten `group_id` se asignan íntegramente al mismo subconjunto (train/val/test), evitando que el modelo memorice correspondencias durante el entrenamiento.
* **Técnicas de barajado (*shuffling*)** y sanitización de texto para eliminar artefactos residuales del scraping o de la generación.

---

### 4.3. Análisis Exploratorio de Datos (EDA) Avanzado

El directorio `EDA/` contiene un conjunto de análisis organizados en dos familias: análisis sobre **cuerpo completo** (`multimodal_dataset.jsonl`) y análisis sobre **titulares** (`titles_data.jsonl`).

#### 4.3.1. Estadísticas Descriptivas y Distribución de Longitud

* **`dataAnalysis.py` / `dataAnalysisTitle.py`**: Calculan estadísticas descriptivas (media, mediana, percentiles) de `word_count`, `char_count` y `avg_word_length` agrupadas por clase (Real vs. AI). Generan histogramas comparativos y boxplots (`eda_word_distribution.png`, `eda_summary_stats.csv`).

#### 4.3.2. Análisis de N-gramas y Solapamiento de Jaccard

* **`ngramAnalysis.py`**: Extrae los top-K n-gramas (uni-, bi-, tri-gramas) más frecuentes para cada clase usando `CountVectorizer` de sklearn. Genera gráficos de barras comparativos.
* **`ngramAnalysisOverlap.py`**: Extiende el análisis calculando:
  * **Coeficiente de Jaccard**: $J = \frac{|N_{real} \cap N_{AI}|}{|N_{real} \cup N_{AI}|}$ — mide el solapamiento léxico entre clases.
  * **Porcentaje de overlap**: Proporción de n-gramas de la IA presentes en el corpus real.
  * Muestra ejemplos de n-gramas compartidos para análisis cualitativo.

#### 4.3.3. Diversidad Léxica (Type-Token Ratio)

* **`ttrAnalysisTitle.py`**: Calcula el **TTR** (Type-Token Ratio) para cada titular: $TTR = \frac{|\text{types}|}{|\text{tokens}|}$, donde un $TTR = 1.0$ indica máxima diversidad (todas las palabras son únicas). Genera boxplots y densidades comparativas (`ttr_distribution_titles.png`, `ttr_density_titles.png`). La tokenización se realiza con NLTK `word_tokenize`.

#### 4.3.4. Análisis de Artefactos Diferenciales

* **`diferentialAnalysis.py` / `diferentialAnalysisTitle.py`**: Detectan **artefactos textuales** que distinguen el texto generado del humano, como patrones de caracteres especiales (`\n\n`, espacios dobles, puntuación diferencial). Comparan la frecuencia de caracteres no alfanuméricos entre clases y reportan los 15 con mayor diferencia a favor de la IA.

#### 4.3.5. Estudio de Perplejidad ($PPL$) con GPT-2

* **`perplexityAnalysis.py` / `perplexityAnalysisTitle.ipynb`**: Calculan la **perplejidad** de cada texto usando GPT-2 como modelo de referencia:

$$PPL(x) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

  * Hipótesis: Los textos generados por IA presentan menor perplejidad (mayor predictibilidad) que los textos humanos.
  * Implementación: Tokenización con max 512 tokens, inferencia `@torch.no_grad()` en GPU, cálculo de NLL (Negative Log-Likelihood) del modelo.
  * Limpieza de outliers al percentil 95 para visualización.
  * Genera histogramas de densidad con líneas de media para cada clase (`perplexity_histogram.png`).

#### 4.3.6. Visualización del Espacio Latente (t-SNE)

* **`visualizingEmbeddings.py` / `visualizingEmbeddingsTitle.py`**: Generan embeddings semánticos con **Sentence-BERT** (`all-MiniLM-L6-v2`) y los proyectan a 2D mediante **t-SNE** (perplexity=30, init PCA). El scatter plot resultante muestra la separabilidad (o superposición) entre las clases Real y Sintética en el espacio semántico (`semantic_distribution_tsne.png`).

---

### 4.4. Fase de Modelado (Entrenamiento y Evaluación)

El directorio `models/` implementa tres familias arquitectónicas de Transformers, evaluadas sobre tres enfoques de entrada progresivamente multimodales.

#### 4.4.1. Enfoques de Entrada

| Enfoque | Input | Dataset | Notebooks |
|---|---|---|---|
| **Solo Titulares** | `title` | `titles_data.jsonl` | `*-Titles*.ipynb` |
| **Titulares + Img-to-Text** | `title` + `img_text` (*early fusion* por concatenación) | `titles_img_data.jsonl` | `*-Img*.ipynb` |
| **Similitud CLIP** | Embeddings visuales + textuales (zero-shot) | `titles_img_data.jsonl` | `CLIP_Semantic_Alignment.ipynb` |

En los enfoques supervisados (Titulares y Multimodal), la partición del dataset emplea `GroupShuffleSplit` sobre `group_id` para evitar *data leakage*, asegurando que el par real/sintético nunca se reparta entre subconjuntos.

#### 4.4.2. Arquitectura Encoder-Only (DeBERTa-v3)

**Directorio:** `models/encoder-only/`

* **`DeBERTa.ipynb`** — Fine-tuning del modelo DeBERTa-v3-base para clasificación binaria (Real vs. Sintético).
* **`RoBERTa.ipynb`** — Variante con RoBERTa como baseline comparativo.
* **`Encoder-Only-Sweep.ipynb`** — Hyperparameter sweep bayesiano sobre el dataset de contenido completo, integrando W&B (Weights & Biases) para tracking de métricas.
* **`Encoder-Only-Sweep-Titles.ipynb`** — Sweep sobre el enfoque de solo titulares.
* **`Encoder-Only-Sweep-Img.ipynb`** — Sweep sobre el enfoque multimodal (titulares + img-to-text concatenados mediante *early fusion*).
* **`Encoder-Only.ipynb`** — Entrenamiento base sobre contenido completo.

**Arquitectura:** `AutoModelForSequenceClassification` con cabeza de clasificación binaria. Tokenización con max 256 tokens (titulares) o 512 tokens (contenido completo).

#### 4.4.3. Arquitectura Decoder-Only (Llama 3.1)

**Directorio:** `models/decoder-only/`

* **`Decoder-Only.ipynb`** — Inferencia y clasificación con Llama 3.1 sobre contenido completo.
* **`Decoder-Only-Titles.ipynb`** — Enfoque de solo titulares.
* **`Decoder-Only-Img.ipynb`** — Enfoque multimodal (titulares + img-to-text).

**Metodología:** Utiliza la API de Hugging Face (`huggin_face_api.py`) o inferencia local según la configuración. Evaluación basada en la entropía de la salida generativa y clasificación por prompting.

#### 4.4.4. Arquitectura Encoder-Decoder (T5/Flan-T5)

**Directorio:** `models/encoder-decoder/`

* **`Encoder-Decoder.ipynb`** — Fine-tuning sobre contenido completo.
* **`Encoder-Decoder-Titles.ipynb`** — Enfoque de solo titulares.
* **`Encoder-Decoder-Titles-Sweep.ipynb`** — Sweep bayesiano sobre titulares.
* **`Encoder-Decoder-Img-Sweep.ipynb`** — Sweep bayesiano sobre el enfoque multimodal.
* **`Sweep-Configuratio.ipynb`** — Configuración de sweeps compartida.

**Arquitectura:** Formulación Text-to-Text con T5. El texto de entrada se prefija con una instrucción de tarea y el modelo genera la etiqueta como secuencia de texto.

#### 4.4.5. Alineación Semántica Multimodal con CLIP (Zero-Shot)

**Directorio:** `models/clip/`

* **`CLIP_Semantic_Alignment.ipynb`** — Análisis de consistencia semántica imagen–titular sin entrenamiento supervisado, usando `openai/clip-vit-base-patch32`.

**Dos enfoques complementarios:**

* **Enfoque A — Fidelidad Original:** Evalúa la similitud coseno entre la **imagen real** y ambos titulares. Hipótesis: $\cos(E_{Img\_Real}, E_{T\_Real}) > \cos(E_{Img\_Real}, E_{T\_Sint})$ → el titular real preserva mayor afinidad semántica.
* **Enfoque B — Sesgo de Generación (*Prompt-Adherence Bias*):** Evalúa la similitud coseno entre la **imagen sintética** (generada por FLUX) y ambos titulares. Hipótesis: $\cos(E_{Img\_Sint}, E_{T\_Sint}) > \cos(E_{Img\_Sint}, E_{T\_Real})$ → una correlación artificialmente alta es indicio de contenido generado.

**Sistema de decisión:** Clasifica como "Real" el titular con mayor score en el Enfoque A. Documenta los casos de confusión donde el Prompt-Adherence Bias del Enfoque B induce error.

**Visualizaciones:** Histogramas de distribución de similitud, histogramas de $\Delta$ (deltas), matrices de confusión (absoluta y normalizada), scatter plot $\Delta_A$ vs. $\Delta_B$.

---

### 4.5. Resultados y Evaluación

* **Métricas:** F1-Score, Precision-Recall y AUC-ROC.
* **Análisis de Explicabilidad (SHAP/LIME):** Interpretación de qué features textuales activan la predicción de contenido sintético.
* **Comparativa entre enfoques:** Evaluación cruzada de los tres enfoques de entrada (solo titulares, multimodal, CLIP zero-shot) sobre las tres familias arquitectónicas.

---

## # 5. Conclusión y Vías Futuras

* **5.1. Conclusiones:** Resumen de hallazgos sobre la predictibilidad de Gemini frente al estilo periodístico humano.
* **5.2. Limitaciones:** Desafíos encontrados en el entrenamiento y sesgos del dataset.
* **5.3. Trabajo Futuro:** Escalado a modelos de mayor tamaño, análisis de "Burstiness" y robustez frente a ataques adversarios.
