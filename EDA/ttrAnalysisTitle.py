import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
import os

# Descargar punkt para tokenización si es necesario
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Configuración de rutas
DATASET_PATH = 'dataset/titles_data.jsonl'
OUTPUT_DIR = 'EDA'

def calculate_ttr(text):
    """Calcula el ratio Type-Token para un texto dado."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    # Tokenización simple (convertir a minúsculas para unificar)
    tokens = word_tokenize(text.lower())
    
    # Filtrar tokens no alfabéticos (opcional, pero recomendado para TTR)
    tokens = [t for t in tokens if t.isalpha()]
    
    if not tokens:
        return 0.0
        
    types = set(tokens)
    return len(types) / len(tokens)

def perform_ttr_analysis():
    # 1. Cargar datos
    data = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    print("Calculando Type-Token Ratio (TTR) por titular...")
    # 2. Calcular TTR
    df['ttr'] = df['title'].apply(calculate_ttr)
    
    # Añadir etiqueta textual para las visualizaciones
    df['Label'] = df['is_real'].map({1: 'Real', 0: 'AI (Synthetic)'})
    
    # 3. Estadísticas descriptivas
    ttr_stats = df.groupby('Label')['ttr'].describe()
    print("\n--- Estadísticas de Type-Token Ratio (TTR) ---")
    print(ttr_stats)
    
    # 4. Visualización
    plt.figure(figsize=(10, 6))
    
    # Usando seaborn para un boxplot
    sns.boxplot(x='Label', y='ttr', data=df, hue='Label', palette=['#1f77b4', '#ff7f0e'], legend=False)
    plt.title('Distribución de Diversidad Léxica (TTR) en Titulares')
    plt.ylabel('Type-Token Ratio (1.0 = todas las palabras son únicas)')
    plt.xlabel('Clase')
    
    # Guardar gráfico
    output_path = os.path.join(OUTPUT_DIR, 'ttr_distribution_titles.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado en {output_path}")
    
    # Opcional: Graficar histograma apilado/superpuesto
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='ttr', hue='Label', element='step', common_norm=False, stat='density', fill=True, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Densidad de TTR: Real vs AI')
    plt.xlabel('Type-Token Ratio')
    plt.ylabel('Densidad')
    output_path_hist = os.path.join(OUTPUT_DIR, 'ttr_density_titles.png')
    plt.savefig(output_path_hist, dpi=300, bbox_inches='tight')
    print(f"Histograma de densidad guardado en {output_path_hist}")

if __name__ == "__main__":
    # Asegurarse de que estamos en el directorio correcto o que las rutas son válidas
    perform_ttr_analysis()
