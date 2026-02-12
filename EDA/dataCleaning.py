import json
import hashlib
from pathlib import Path

def deduplicate_news(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    seen_hashes = set()  # Almacena los hashes únicos
    unique_count = 0
    duplicate_count = 0

    print(f"--- Iniciando limpieza de: {input_path.name} ---")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            
            entry = json.loads(line)
            # Normalizamos el texto (opcional: quitar espacios extra)
            content = entry.get('content', '').strip()
            
            # Generamos un hash MD5 único del contenido
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                # Es la primera vez que vemos este texto, lo guardamos
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                seen_hashes.add(content_hash)
                unique_count += 1
            else:
                # Es un duplicado (mismo contenido, diferente ID)
                duplicate_count += 1

    print(f"Proceso finalizado.")
    print(f"Noticias únicas guardadas: {unique_count}")
    print(f"Duplicados eliminados: {duplicate_count}")

if __name__ == "__main__":
    # Ajusta las rutas a tu estructura de carpetas
    INPUT = "scraping/data_collection/real_news.jsonl"
    OUTPUT = "scraping/data_collection/real_news_cleaned.jsonl"
    deduplicate_news(INPUT, OUTPUT)