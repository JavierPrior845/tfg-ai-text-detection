import json
from pathlib import Path

# Configuración
MULTIMODAL_FILE = Path("dataset/multimodal_dataset.jsonl")
SYNTHETIC_OUT = Path("generation/synthetic_news.jsonl")

def backup_synthetic():
    if not MULTIMODAL_FILE.exists():
        print("Error: No existe el dataset multimodal.")
        return

    synthetic_count = 0
    with open(MULTIMODAL_FILE, "r", encoding="utf-8") as f_in, \
         open(SYNTHETIC_OUT, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            data = json.loads(line)
            # Filtramos solo las que no son reales
            if data.get("is_real") == 0:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                synthetic_count += 1
                
    print(f"✅ Backup completado: {synthetic_count} noticias sintéticas guardadas en {SYNTHETIC_OUT}")

if __name__ == "__main__":
    backup_synthetic()