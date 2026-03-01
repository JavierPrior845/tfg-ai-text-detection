import json
import os
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE RUTAS ---
REAL_NEWS_PATH = Path("scraping/data_collection/real_news_no_duplicates.jsonl")
OUTPUT_PATH = Path("dataset/titles_data.jsonl")

# Esquema para la respuesta estructurada
class SyntheticTitle(BaseModel):
    headline: str

class TitleGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Configura la variable de entorno GEMINI_API_KEY")
        
        self.client = genai.Client(api_key=self.api_key)
        self.avg_title_length = self._calculate_avg_title_length()
        self.processed_ids = self._get_processed_ids()

    def _calculate_avg_title_length(self):
        """Calcula la media de palabras de los titulares reales."""
        lengths = []
        with open(REAL_NEWS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                lengths.append(len(data['title'].split()))
        avg = int(np.mean(lengths))
        print(f"[*] Media de longitud detectada: {avg} palabras.")
        return avg

    def _get_processed_ids(self):
        """Evita duplicados si el script se detiene."""
        ids = set()
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    ids.add(json.loads(line)["group_id"])
        return ids

    def generate_title(self, content):
        """Llama a Gemini para generar un titular basado en el cuerpo de la noticia."""
        prompt = f"""
        Based on the following news content, write a compelling and accurate headline.
        The headline MUST HAVE a length of approximately {self.avg_title_length} words.
        Maintain a professional journalistic tone.

        CONTENT: {content[:4000]}  # Limitamos entrada para ahorrar tokens
        
        OUTPUT FORMAT: JSON {{headline}}
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SyntheticTitle,
                    max_output_tokens=100
                ),
            )
            return response.parsed.headline if response.parsed else None
        except Exception as e:
            print(f"❌ Error API: {e}")
            return None

    def run(self, limit=None):
        count = 0
        with open(REAL_NEWS_PATH, "r", encoding="utf-8") as f_in, \
            open(OUTPUT_PATH, "a", encoding="utf-8") as f_out:
            
            for line in tqdm(f_in, desc="Generando titulares"):
                real_data = json.loads(line)
                group_id = real_data["article_id"]

                if group_id in self.processed_ids:
                    continue

                # 1. Guardar Titular Real
                real_entry = {
                    "group_id": group_id,
                    "title": real_data["title"],
                    "is_real": 1
                }
                f_out.write(json.dumps(real_entry, ensure_ascii=False) + "\n")

                # 2. Generar y Guardar Titular Sintético
                synthetic_title = self.generate_title(real_data["content"])
                if synthetic_title:
                    fake_entry = {
                        "group_id": group_id,
                        "title": synthetic_title,
                        "is_real": 0
                    }
                    f_out.write(json.dumps(fake_entry, ensure_ascii=False) + "\n")
                
                count += 1
                if limit and count >= limit:
                    break

if __name__ == "__main__":
    generator = TitleGenerator()
    generator.run(limit=250)