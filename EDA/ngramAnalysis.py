import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_ngram_comparison(path, n=2, top_k=15):
    # 1. Cargar datos
    data = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]
    df = pd.DataFrame(data)
    
    # 2. Separar Real vs AI
    real_titles = df[df['is_real'] == 1]['title']
    ai_titles = df[df['is_real'] == 0]['title']
    
    # 3. Extraer N-gramas
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english')
    
    def get_top_ngrams(corpus):
        bag_of_words = vec.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

    real_top = get_top_ngrams(real_titles)
    ai_top = get_top_ngrams(ai_titles)
    
    # 4. Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.barplot(x=[x[1] for x in real_top], y=[x[0] for x in real_top], ax=ax1, palette='Blues_r')
    ax1.set_title(f'Top {n}-gramas Reales')
    
    sns.barplot(x=[x[1] for x in ai_top], y=[x[0] for x in ai_top], ax=ax2, palette='Oranges_r')
    ax2.set_title(f'Top {n}-gramas Gemini (AI)')
    
    plt.tight_layout()
    plt.show()

# Uso: analyze_ngram_comparison("dataset/titles_data.jsonl", n=2)