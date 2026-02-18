import json
from pathlib import Path

# Configuración
REAL_NO_DUPS = Path("scraping/data_collection/real_news_no_duplicates.jsonl")
MULTIMODAL_FILE = Path("dataset/multimodal_dataset.jsonl")
PENDING_OUT = Path("scraping/data_collection/pending_real_news.jsonl")

def extract_pending():
    # 1. Obtener IDs ya procesados del dataset multimodal
    processed_ids = set()
    if MULTIMODAL_FILE.exists():
        with open(MULTIMODAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["group_id"])
                except: continue

    # 2. Buscar en el archivo de "No Duplicados" las que falten
    pending_count = 0
    if not REAL_NO_DUPS.exists():
        print(f"Error: No se encuentra {REAL_NO_DUPS}")
        return

    with open(REAL_NO_DUPS, "r", encoding="utf-8") as f_in, \
         open(PENDING_OUT, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            data = json.loads(line)
            if data["article_id"] not in processed_ids:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                pending_count += 1

    print(f"📊 Resumen de sincronización:")
    print(f"   - Noticias ya procesadas: {len(processed_ids)}")
    print(f"   - Noticias pendientes encontradas: {pending_count}")
    print(f"   - Nuevo archivo creado: {PENDING_OUT}")

if __name__ == "__main__":
    extract_pending()