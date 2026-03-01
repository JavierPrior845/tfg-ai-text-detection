import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Configuración de rutas (ajusta a tu BASE_DIR si es necesario)
dataset_path = Path("dataset/titles_data.jsonl")

def perform_eda(file_path):
    # 1. Carga de datos
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    # 2. Feature Engineering para el EDA
    df['char_count'] = df['title'].str.len()
    df['word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    df['avg_word_length'] = df['title'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

    # 3. Estadísticas Descriptivas por Clase
    stats = df.groupby('is_real')[['word_count', 'char_count', 'avg_word_length']].describe()
    print("--- Descriptive Statistics (Human [1] vs AI [0]) ---")
    print(stats)

    # 4. Visualización de Distribuciones
    plt.figure(figsize=(12, 5))

    # Histograma de Word Count
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='word_count', hue='is_real', kde=True, element="step")
    plt.title('Distribution of Word Count')
    plt.xlabel('Number of Words')

    # Boxplot para comparar medianas y outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='is_real', y='word_count')
    plt.title('Word Count Boxplot (0: AI, 1: Human)')
    
    plt.tight_layout()
    plt.savefig("eda_word_distribution_titles.png")
    print("\nGráfico guardado como eda_word_distribution_titles.png")

    # 5. Guardar resumen para la memoria del TFG
    stats.to_csv("eda_summary_stats_titles.csv")
    return df

if __name__ == "__main__":
    if dataset_path.exists():
        df_analyzed = perform_eda(dataset_path)
    else:
        print(f"Archivo no encontrado en {dataset_path}")
