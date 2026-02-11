import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd

def analyze_artifacts(file_path):
    real_chars = Counter()
    ai_chars = Counter()
    
    # Patrón para caracteres no alfanuméricos (excluyendo espacios)
    pattern = re.compile(r'[^a-zA-Z0-9\s]')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            content = obj['content']
            
            # Buscamos secuencias específicas como \n\n, espacios dobles, etc.
            escapes = re.findall(r'\\n|\\t|\s{2,}', content)
            
            if obj['is_real'] == 1:
                real_chars.update(pattern.findall(content))
                real_chars.update(escapes)
            else:
                ai_chars.update(pattern.findall(content))
                ai_chars.update(escapes)

    # Creamos un DataFrame para comparar
    df_real = pd.DataFrame.from_dict(real_chars, orient='index', columns=['Real'])
    df_ai = pd.DataFrame.from_dict(ai_chars, orient='index', columns=['AI'])
    
    comparison = df_real.join(df_ai, how='outer').fillna(0)
    comparison['Diff'] = comparison['AI'] - comparison['Real']
    
    # Mostramos los 15 caracteres con mayor diferencia a favor de la IA
    print("--- Top Artifacts detectados en IA vs Real ---")
    print(comparison.sort_values(by='Diff', ascending=False).head(15))

if __name__ == "__main__":
    analyze_artifacts("dataset/multimodal_dataset.jsonl")