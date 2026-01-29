import os
import json
import time
import requests
import trafilatura
from dotenv import load_dotenv

load_dotenv()

def get_existing_ids(filepath):
    """Carga los IDs ya guardados para evitar duplicados (Set para O(1) lookup)."""
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8") as f:
        return {json.loads(line)["article_id"] for line in f}

def process_automated_ingestion(goal_new_articles=50):
    output_file = "real_news.jsonl"
    existing_ids = get_existing_ids(output_file)
    new_articles_count = 0
    next_page_token = None
    
    print(f"Dataset actual: {len(existing_ids)} noticias. Objetivo: +{goal_new_articles}")

    while new_articles_count < goal_new_articles:
        params = {
            "apikey": os.getenv("NEWSDATA_API_KEY"),
            "language": "en",
            "size": 10  # Maximizamos créditos
        }
        if next_page_token:
            params["page"] = next_page_token

        resp = requests.get("https://newsdata.io/api/1/latest", params=params)
        if resp.status_code != 200:
            print(f"Error o límite de créditos: {resp.text}")
            break
        
        data = resp.json()
        results = data.get('results', [])
        next_page_token = data.get("nextPage") # El "puntero" a la siguiente página

        for item in results:
            aid = item.get('article_id')
            
            # EVITAR COLISIÓN
            if aid in existing_ids:
                continue
            
            # SCRAPING (Solo si no es duplicado)
            content = trafilatura.extract(trafilatura.fetch_url(item.get('link')))
            
            if content and len(content.split()) >= 300:
                entry = {
                    "article_id": aid,
                    "title": item.get("title"),
                    "content": content,
                    "word_count": len(content.split()),
                    "image_url": item.get("image_url"),
                    "category": item.get("category")[0] if item.get("category") else "general",
                    "source": item.get("source_id"),
                    "pub_date": item.get("pubDate"),
                    "is_real": True
                }
                
                # Guardado inmediato (Append mode)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                existing_ids.add(aid)
                new_articles_count += 1
                print(f"[{new_articles_count}/{goal_new_articles}] Guardado: {aid}")
            
            if new_articles_count >= goal_new_articles:
                break
        
        if not next_page_token:
            print("No hay más páginas disponibles.")
            break
        
        time.sleep(1) # Courtesy delay

if __name__ == "__main__":
    process_automated_ingestion(goal_new_articles=5)