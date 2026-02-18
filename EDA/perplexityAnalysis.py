import json
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
DATASET_PATH = Path("dataset/multimodal_dataset.jsonl")
MODEL_ID = "gpt2" # Modelo estándar para medir entropía/perplejidad

def calculate_perplexity(text, model, tokenizer, device):
    """Calcula la perplejidad de un texto usando GPT-2."""
    if not text or len(str(text).strip()) < 10:
        return None
    
    # GPT-2 tiene un límite de 1024, pero para noticias 512 suele bastar y es más rápido
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Negative Log Likelihood
        
    return torch.exp(loss).item()

def main():
    # 1. Cargar Datos
    if not DATASET_PATH.exists():
        print(f"Error: No se encuentra el archivo en {DATASET_PATH}")
        return

    print(f"[*] Cargando dataset desde {DATASET_PATH}...")
    data = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    # Mapeo de labels para el gráfico
    df['label'] = df['is_real'].map({1: "Real", 0: "AI"})

    # 2. Configurar Modelo y GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Usando dispositivo: {device.upper()}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # 3. Cálculo de Perplejidad
    print("[*] Calculando perplejidad (esto puede tardar unos minutos)...")
    tqdm.pandas()
    df['perplexity'] = df['content'].progress_apply(
        lambda x: calculate_perplexity(x, model, tokenizer, device)
    )

    # 4. Limpieza de Outliers para visualización
    # La PPL puede tener valores astronómicos en textos muy raros. 
    # Usamos el percentil 95 para que el histograma no se vea "aplastado".
    upper_limit = df['perplexity'].quantile(0.95)
    df_plot = df[df['perplexity'] < upper_limit].copy()

    # 5. Generación del Histograma
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # Gráfico de densidad e histograma solapados
    plot = sns.histplot(
        data=df_plot, 
        x="perplexity", 
        hue="label", 
        kde=True, 
        element="step", 
        palette={"AI": "#FF5733", "Real": "#2E86C1"},
        alpha=0.5
    )

    plt.title("Paso 2: Distribución de Perplejidad (GPT-2 PPL)", fontsize=16, fontweight='bold')
    plt.xlabel("Perplejidad (Menor = Texto más predecible/IA)", fontsize=12)
    plt.ylabel("Frecuencia de noticias", fontsize=12)
    
    # Añadir líneas de media para la defensa del TFG
    mean_ai = df[df['is_real'] == 0]['perplexity'].mean()
    mean_real = df[df['is_real'] == 1]['perplexity'].mean()
    plt.axvline(mean_ai, color='#FF5733', linestyle='--', label=f'Media AI: {mean_ai:.2f}')
    plt.axvline(mean_real, color='#2E86C1', linestyle='--', label=f'Media Real: {mean_real:.2f}')
    
    plt.legend()
    
    # Guardar gráfico
    output_img = "EDA/perplexity_histogram.png"
    plt.savefig(output_img)
    print(f"\n[+] Histograma guardado en: {output_img}")
    plt.show()

    # 6. Guardar resultados para el Paso 3
    df.to_csv("dataset/dataset_with_perplexity.csv", index=False)
    print("[+] Datos con métricas guardados en dataset/dataset_with_perplexity.csv")

if __name__ == "__main__":
    main()