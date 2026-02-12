import pandas as pd
from pathlib import Path

def deduplicate_by_title(input_file, output_file):
    if not Path(input_file).exists():
        print("Archivo de entrada no encontrado.")
        return

    # Cargamos el JSONL
    df = pd.read_json(input_file, lines=True)
    before = len(df)

    # Eliminamos duplicados por título (Canonicalization)
    df_cleaned = df.drop_duplicates(subset=['title'], keep='first')
    after = len(df_cleaned)

    # Guardamos el resultado limpio
    df_cleaned.to_json(output_file, orient='records', lines=True, force_ascii=False)
    
    print(f"--- Reporte de Deduplicación ---")
    print(f"Originales: {before}")
    print(f"Únicas: {after}")
    print(f"Eliminadas: {before - after}")

if __name__ == "__main__":
    INPUT = "scraping/data_collection/real_news.jsonl"
    OUTPUT = "scraping/data_collection/real_news_no_duplicates.jsonl"
    deduplicate_by_title(INPUT, OUTPUT)