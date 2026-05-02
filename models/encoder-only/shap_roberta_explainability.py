import torch
import shap
import pandas as pd
import numpy as np
from transformers import pipeline

# =========== 1. PIPELINE SETUP ===========
# Creamos el pipeline de clasificación. Usamos top_k=None (el estándar moderno para return_all_scores=True)
# Esto asegura que nos devuelva un vector completo de probabilidades para cada input (IA y Human).
device_id = 0 if torch.cuda.is_available() else -1

pred_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device_id,
    top_k=None
)

# =========== 2. EXPLAINER INITIALIZATION ===========
# Mapeamos textualmente el tokenizador para que sirva de máscara.
# Utilizamos explícitamente "partition" explainer (PartitionExplainer), optimizado 
# para interacciones O(N^2) jerárquicas NLP en vez de usar KernelSHAP (que es O(2^N) y muy lento).
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(pred_pipeline, masker=masker, algorithm="partition")

# =========== 3. SAMPLING (Aciertos y Errores) ===========
# Extraemos el test completo a pandas para categorizar
df_test_full = dataset_dict['test'].to_pandas()

print(f"Obteniendo predicciones para seleccionar los {len(df_test_full)} casos del Test (TP, TN, FP, FN)...")
# Pipeline computation en batch (maneja la VRAM eficientemente)
preds = pred_pipeline(df_test_full['title'].tolist(), batch_size=16)

# El label con mayor score es la predicción:
pred_labels = []
for p in preds:
    best_pred = max(p, key=lambda x: x['score'])
    pred_labels.append(0 if best_pred['label'] == 'IA' else 1)
df_test_full['pred_label'] = pred_labels

# Agrupamos por TP, TN, FP, FN (Target 0: IA/Sintético)
subset_tp = df_test_full[(df_test_full['is_real'] == 0) & (df_test_full['pred_label'] == 0)]
subset_tn = df_test_full[(df_test_full['is_real'] == 1) & (df_test_full['pred_label'] == 1)]
subset_fp = df_test_full[(df_test_full['is_real'] == 1) & (df_test_full['pred_label'] == 0)]
subset_fn = df_test_full[(df_test_full['is_real'] == 0) & (df_test_full['pred_label'] == 1)]

# Balanceamos la submuestra de ~50 observaciones recuperando algo de todas (si están disponibles)
samples = []
for df_sub in [subset_tp, subset_tn, subset_fp, subset_fn]:
    samples.append(df_sub.sample(min(len(df_sub), 13), random_state=42))

sample_df = pd.concat(samples).drop_duplicates()
# Si faltan muestras para llegar a 50 (por ej. si no hay falsos positivos), rellenamos random
if len(sample_df) < 50:
    remaining = df_test_full.drop(sample_df.index).sample(50 - len(sample_df), random_state=42)
    sample_df = pd.concat([sample_df, remaining])

sample_texts = sample_df['title'].tolist()

# =========== 4. COMPUTATION ===========
print("Generando SHAP values sobre la submuestra...")
shap_values = explainer(sample_texts)

# FIX VISUALIZACIÓN BPE: Eliminar la métrica Ġ que arrastra RoBERTa 
# para representar espacios y que interfiere con el shap.plots.text
for i in range(len(shap_values)):
    shap_values.data[i] = np.array([str(t).replace("Ġ", " ") for t in shap_values.data[i]])

# Recuperamos el ID de indexación de la clase IA
class_ia_idx = "IA" if "IA" in shap_values.output_names else 0

# =========== 5. VISUALIZATION & INSIGHTS ===========
# Activamos JavaScript si estuviéramos en una celda Jupyter
shap.initjs()

# A. Gráfico Textual Individual: Buscamos un texto Sintético (TP o FN, es decir is_real = 0)
synthetic_loc_idx = sample_df['is_real'].tolist().index(0)

print(f"\n[A] Render de Explicación Textual - Observación en Submuestra: #{synthetic_loc_idx}")
# Dibujamos exactamente qué sumó scores a IA para ESTE titular en concreto
display(shap.plots.text(shap_values[synthetic_loc_idx, :, class_ia_idx]))

# B. Gráfico Global Barras (Top 10 Activaciones Hacia "IA")
print("\n[B] Top 10 tokens globalmente contributivos (Promedio Local/Cohorte):")
shap.plots.bar(shap_values[:, :, class_ia_idx], max_display=10, show=True)

# =========== 6. EXPORT / DATA TABULATION ===========
print("\nExportando tokens principales y sus impactos al CSV 'shap_impact_gemini_tokens.csv'...")
token_data = []

# Iteramos la matriz SHAP dimensionada: (num_docs, num_tokens)
for i in range(len(shap_values)):
    tks = shap_values.data[i]
    # Extraemos solo el logit marginal/aditivo de la clase IA
    vals = shap_values.values[i, :, class_ia_idx]
    
    for t_raw, v in zip(tks, vals):
        t_clean = t_raw.strip().lower()
        if len(t_clean) > 1: # Filtro antiruido de comas / espacios
            token_data.append({'token': t_clean, 'impact_ia': float(v)})

df_tokens = pd.DataFrame(token_data)

# Agregaciones: mean() dictará cuánto inclina este token a que el clasificador falle o acierte hacia la IA
df_shap_export = df_tokens.groupby('token').agg(
    mean_impact_ia=('impact_ia', 'mean'),
    abs_impact_ia=('impact_ia', lambda x: np.mean(np.abs(x))),
    appearances=('impact_ia', 'count')
).reset_index()

# Filtramos tokens residuales y mostramos los líderes naturales
df_shap_export = df_shap_export[df_shap_export['appearances'] > 1]
df_shap_export = df_shap_export.sort_values(by='mean_impact_ia', ascending=False)
df_shap_export.to_csv("shap_impact_gemini_tokens.csv", index=False)

print("\n(Extracto) - TOP 5 Palabras que empujan el score a 'Sintético' (IA):")
print(df_shap_export.head(5))

# ==============================================================================
# NOTA SOBRE LA FIDELIDAD (Fidelity) DE SHAP PARA TRANSFORMERS:
# ==============================================================================
# La "fidelidad" en SHAP nos da un teorema matemático garantizado respecto al modelo. 
# El Partition Explainer calcula valores aproximados de Shapley aglomerando la jerarquía  
# del texto. Lo crucial es la Propiedad de Exactitud Local (Local Accuracy):
# 
# Σ (SHAP_Values_tokens) + Expected_Base_Value_IA == Logit(Output) de IA real medido
#
# Esto se debe a que la matemática detrás no está aproximando una región (como LIME),
# sino dividiendo linealmente el valor marginal real. Las métricas impresas en el dataframe 
# ('mean_impact_ia') son completamente fieles a las ecuaciones de los transformadores 
# internamente mapeadas: NO exigen volver a entrenar el modelo, son transparentes al vector C (Logits).
# ==============================================================================
