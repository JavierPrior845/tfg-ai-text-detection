import os
import json
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from pydantic import BaseModel

load_dotenv()

# --- ESQUEMAS ---
class SyntheticNews(BaseModel):
    headline: str
    content: str
    technique: str

# 1. Obtiene la carpeta donde está ESTE script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_NEWS_FILE = os.path.join(BASE_DIR, "..", "scraping", "data_collection", "real_news.jsonl")
REAL_NEWS_FILE = os.path.normpath(REAL_NEWS_FILE)
FINAL_DATASET_FILE = "multimodal_dataset.jsonl"
IMAGES_DIR = "dataset"
os.makedirs(IMAGES_DIR, exist_ok=True)

class DatasetGenerator:
    def __init__(self):
        # Clientes
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.hf_client = InferenceClient(api_key=os.getenv('FIRST_HF_TK'))
        
        # Estado (para evitar duplicados en el proceso de generación)
        self.processed_ids = self._load_processed_ids()

    def _load_processed_ids(self):
        if not os.path.exists(FINAL_DATASET_FILE): return set()
        with open(FINAL_DATASET_FILE, "r") as f:
            return {json.loads(line)["parent_article_id"] for line in f}

    def generate_fake_text(self, real_title, real_content):
        """Genera texto falso usando Gemini."""
        prompt = f"""
        Eres un redactor de noticias. Basándote en esta noticia REAL, crea una noticia FALSA mediante PARAFRASEO y manipulación de datos.
        TITULAR REAL: {real_title}
        CONTENIDO REAL: {real_content[:1500]}
        
        REGLAS:
        1. Longitud similar a la original.
        2. Mantén el tono periodístico.
        3. Devuelve JSON: headline, content, technique.
        """
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SyntheticNews,
                    max_output_tokens=800
                ),
            )
            return response.parsed
        except Exception as e:
            print(f"Error LLM: {e}")
            return None

    def generate_fake_image(self, fake_headline, article_id):
        """Genera imagen sintética basada en el titular falso."""
        image_path = f"{IMAGES_DIR}/{article_id}_fake.png"
        try:
            # Usamos FLUX o SDXL para mayor calidad
            image = self.hf_client.text_to_image(
                prompt=f"Photojournalism style news photo, high quality, realistic: {fake_headline}",
                model="black-forest-labs/FLUX.1-schnell", # Alternativa Pro
            )
            image.save(image_path)
            return image_path
        except Exception as e:
            print(f"Error Imagen: {e}")
            return None

    def process_pipeline(self):
        with open(REAL_NEWS_FILE, "r") as f:
            for line in f:
                real_data = json.loads(line)
                group_id = real_data["article_id"] # Usamos el ID original como nexo
                
                if group_id in self.processed_ids:
                    continue

                # 1. Generar la noticia falsa (Texto + Imagen)
                fake_text = self.generate_fake_text(real_data["title"], real_data["content"])
                if not fake_text: continue
                
                fake_image_path = self.generate_fake_image(fake_text.headline, group_id)

                # 2. GUARDAR EL PAR (Dos entradas distintas, mismo group_id)
                
                # Entrada REAL
                entry_real = {
                    "group_id": group_id,
                    "is_real": 1,
                    "title": real_data["title"],
                    "content": real_data["content"],
                    "image_path": real_data["image_url"], # URL o path local
                    "model": "human"
                }

                # Entrada FAKE
                entry_fake = {
                    "group_id": group_id,
                    "is_real": 0,
                    "title": fake_text.headline,
                    "content": fake_text.content,
                    "image_path": fake_image_path,
                    "model": "gemini-1.5-flash"
                }

                # Escribimos ambas en el dataset final
                with open(FINAL_DATASET_FILE, "a", encoding="utf-8") as out:
                    out.write(json.dumps(entry_real, ensure_ascii=False) + "\n")
                    out.write(json.dumps(entry_fake, ensure_ascii=False) + "\n")
                
                self.processed_ids.add(group_id)
                print(f"Par guardado para el grupo: {group_id}")

if __name__ == "__main__":
    gen = DatasetGenerator()
    gen.process_pipeline()