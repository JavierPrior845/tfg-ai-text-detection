import pandas as pd
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de rutas
DATASET_PATH = Path("dataset/multimodal_dataset.jsonl")

def visualize_semantic_space():
    # Cargar datos
    texts, labels = [], []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj['content'])
            labels.append('Real' if obj['is_real'] == 1 else 'AI (Synthetic)')

    print(f"Generando embeddings para {len(texts)} textos...")
    
    # 2. Generar Embeddings con SBERT (Modelo 'all-MiniLM-L6-v2' es balanceado y rápido)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    # 3. Reducción de dimensionalidad con t-SNE
    # Perplexity suele ir entre 5 y 50. Como tienes pocos datos ahora, 30 está bien.
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    # 4. Plotting
    plt.figure(figsize=(10, 7))
    df_tsne = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'Label': labels
    })

    sns.scatterplot(data=df_tsne, x='x', y='y', hue='Label', style='Label', palette='viridis', alpha=0.7)
    
    plt.title('Semantic Space Visualization: Real vs Synthetic News (t-SNE)')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("semantic_distribution_tsne.png")
    plt.show()

if __name__ == "__main__":
    visualize_semantic_space()