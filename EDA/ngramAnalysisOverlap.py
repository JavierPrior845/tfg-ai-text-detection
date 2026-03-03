import os

import pandas as pd

import json

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns

import matplotlib.pyplot as plt



def calculate_ngram_overlap(real_titles, ai_titles, n=2):

    """Calcula el solapamiento de n-gramas y el Coeficiente de Jaccard."""

    vec = CountVectorizer(ngram_range=(n, n), stop_words='english')

    

    # Obtenemos el conjunto de n-gramas únicos para cada clase

    vec.fit(real_titles)

    real_ngrams = set(vec.get_feature_names_out())

    

    vec.fit(ai_titles)

    ai_ngrams = set(vec.get_feature_names_out())

    

    intersection = real_ngrams.intersection(ai_ngrams)

    union = real_ngrams.union(ai_ngrams)

    

    jaccard = len(intersection) / len(union) if len(union) > 0 else 0

    overlap_pct = (len(intersection) / len(ai_ngrams)) * 100 if len(ai_ngrams) > 0 else 0

    

    return {

        "jaccard": jaccard,

        "overlap_pct": overlap_pct,

        "common_examples": list(intersection)[:10] # Ejemplos de n-gramas comunes

    }



def analyze_ngram_comparison(path, n=2, top_k=15):

    # 1. Cargar datos

    if not os.path.exists(path):

        print(f"❌ Error: El archivo {path} no existe.")

        return

        

    data = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]

    df = pd.DataFrame(data)

    

    # 2. Separar Real vs AI

    real_titles = df[df['is_real'] == 1]['title']

    ai_titles = df[df['is_real'] == 0]['title']

    

    # 3. Cálculo de Solapamiento (Overlap)

    stats = calculate_ngram_overlap(real_titles, ai_titles, n)

    print(f"\n" + "="*40)

    print(f"📊 ESTADÍSTICAS DE SOLAPAMIENTO ({n}-gramas)")

    print(f"="*40)

    print(f"• Coeficiente de Jaccard: {stats['jaccard']:.4f}")

    print(f"• % de N-gramas AI presentes en Real: {stats['overlap_pct']:.2f}%")

    print(f"• Ejemplos comunes: {stats['common_examples']}")

    print(f"="*40 + "\n")



    # 4. Extraer N-gramas para visualización

    vec = CountVectorizer(ngram_range=(n, n), stop_words='english')

    

    def get_top_ngrams(corpus):

        bag_of_words = vec.fit_transform(corpus)

        sum_words = bag_of_words.sum(axis=0) 

        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

        return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]



    real_top = get_top_ngrams(real_titles)

    ai_top = get_top_ngrams(ai_titles)

    

    # 5. Visualización Pro

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    

    sns.barplot(

        x=[x[1] for x in real_top], y=[x[0] for x in real_top], 

        ax=ax1, palette='Blues_r', hue=[x[0] for x in real_top], legend=False

    )

    ax1.set_title(f'Top {n}-gramas: Noticias Reales', fontsize=14)



    sns.barplot(

        x=[x[1] for x in ai_top], y=[x[0] for x in ai_top], 

        ax=ax2, palette='Oranges_r', hue=[x[0] for x in ai_top], legend=False

    )

    ax2.set_title(f'Top {n}-gramas: Titulares Gemini (AI)', fontsize=14)



    plt.tight_layout()

    output_img = f"EDA/plots/ngram_{n}_comparison.png"

    os.makedirs("EDA/plots", exist_ok=True)

    plt.savefig(output_img, dpi=300)

    print(f"[*] Gráfico guardado en: {output_img}")



if __name__ == "__main__":

    # Ejecutamos para Bigramas (n=2)

    analyze_ngram_comparison("dataset/titles_data.jsonl", n=3)