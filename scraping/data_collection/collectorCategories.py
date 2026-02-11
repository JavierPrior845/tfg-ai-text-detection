import os
import json
import time
import requests
import trafilatura
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuración de rutas absoluta para evitar líos de directorios
SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_FILE = SCRIPT_DIR / "last_token.txt"
OUTPUT_FILE = SCRIPT_DIR / "real_news.jsonl"

categories = ["environment", "top", "business", "technology", "science", "politics"]

def get_last_token():
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None

def save_last_token(token):
    if token:
        TOKEN_FILE.write_text(str(token))

def get_existing_ids(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return {json.loads(line)["article_id"] for line in f}

def process_automated_ingestion(goal_new_articles=100):
    existing_ids = get_existing_ids(OUTPUT_FILE)
    new_articles_count = 0
    next_page_token = get_last_token()
    
    print(f"Dataset actual: {len(existing_ids)} noticias. Objetivo: +{goal_new_articles}")

    for cat in categories:
        if new_articles_count >= goal_new_articles:
            break
            
        print(f"\n--- Iniciando categoría: {cat.upper()} ---")
        
        # Resetear el token si cambias de categoría (opcional, pero NewsData suele ligar tokens a queries)
        # Si prefieres seguir el hilo global, no toques next_page_token aquí.

        while new_articles_count < goal_new_articles:
            params = {
                "apikey": os.getenv("NEWSDATA_API_KEY"),
                "language": "en",
                "size": 10,
                "category": cat,
                "removeduplicate": 1
            }
            if next_page_token:
                params["page"] = next_page_token

            try:
                resp = requests.get("https://newsdata.io/api/1/latest", params=params)
                if resp.status_code != 200:
                    print(f"Error {resp.status_code} en {cat}: {resp.text}")
                    break # Salta a la siguiente categoría si hay error de cuota
                
                data = resp.json()
                results = data.get('results', [])
                next_page_token = data.get("nextPage")
                save_last_token(next_page_token) # Checkpoint inmediato
                
                if not results:
                    print(f"No más resultados en {cat}.")
                    break

                for item in results:
                    aid = item.get('article_id')
                    if aid in existing_ids:
                        continue
                    
                    # Scraping
                    downloaded = trafilatura.fetch_url(item.get('link'))
                    content = trafilatura.extract(downloaded) if downloaded else None
                    
                    if content and len(content.split()) >= 300:
                        entry = {
                            "article_id": aid,
                            "title": item.get("title"),
                            "content": content,
                            "word_count": len(content.split()),
                            "image_url": item.get("image_url"),
                            "category": cat,
                            "source": item.get("source_id"),
                            "pub_date": item.get("pubDate"),
                            "is_real": True
                        }
                        
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
                        existing_ids.add(aid)
                        new_articles_count += 1
                        print(f"[{new_articles_count}/{goal_new_articles}] Guardado: {item.get('title')[:50]}...")
                    
                    if new_articles_count >= goal_new_articles:
                        break
                
                if not next_page_token:
                    break
                    
                time.sleep(1) # Courtesy delay entre llamadas API

            except Exception as e:
                print(f"Error crítico: {e}")
                break

if __name__ == "__main__":
    process_automated_ingestion(goal_new_articles=200)