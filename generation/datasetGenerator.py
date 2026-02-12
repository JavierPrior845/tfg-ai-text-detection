import os
import json
import time
from pathlib import Path
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

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_ROOT = ROOT_DIR / "dataset"
IMAGES_DIR = DATASET_ROOT / "fake_images"
FINAL_DATASET_FILE = DATASET_ROOT / "multimodal_dataset.jsonl"
REAL_NEWS_FILE = ROOT_DIR / "scraping" / "data_collection" / "real_news_no_duplicates.jsonl"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class DatasetGenerator:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.hf_client = InferenceClient(api_key=os.getenv('FIRST_HF_TK'))
        self.processed_ids = self._load_processed_ids()

    def _load_processed_ids(self):
        """Carga IDs ya procesados para evitar re-trabajo y gasto de API."""
        if not FINAL_DATASET_FILE.exists():
            return set()
        with open(FINAL_DATASET_FILE, "r", encoding="utf-8") as f:
            # Usamos group_id para verificar si el par ya existe
            return {json.loads(line)["group_id"] for line in f}

    def generate_fake_text(self, real_title, real_content):
        
        target_words = len(real_content.split())
        prompt = f"""
            [CRITICAL INSTRUCTION]
            You must paraphrase the following news article. 
            The generated content MUST HAVE between {int(target_words*0.9)} and {int(target_words*1.1)} words.
            Do not be concise. Mimic the original news length and detail density.

            REAL TITULAR: {real_title}
            REAL CONTENT: {real_content}
            
            OUTPUT FORMAT: JSON {{headline, content, technique}}
            """
        
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SyntheticNews,
                    max_output_tokens=1500
                ),
            )
            return response.parsed
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def generate_fake_image(self, fake_headline, article_id):
        """Genera imagen sintética y devuelve la ruta relativa para el JSONL."""
        file_name = f"{article_id}_fake.png"
        image_path = IMAGES_DIR / file_name
        if not image_path.exists():
            try:
                image = self.hf_client.text_to_image(
                    prompt=f"Professional photojournalism, high quality, realistic news photo: {fake_headline}",
                    model="black-forest-labs/FLUX.1-schnell",
                )
                image.save(image_path)
            except Exception as e:
                print(f"Image Error: {e}")
                return None
        return f"dataset/fake_images/{file_name}"
        
    def process_pipeline(self, goal=10):
        if not REAL_NEWS_FILE.exists():
            print(f"Source file not found: {REAL_NEWS_FILE}")
            return

        new_pairs_count = 0
        print(f"Starting pipeline. Already processed: {len(self.processed_ids)} groups. Goal: +{goal}")

        with open(REAL_NEWS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if new_pairs_count >= goal:
                    break
                
                real_data = json.loads(line)
                group_id = real_data["article_id"]
                
                if group_id in self.processed_ids:
                    continue

                print(f"[*] Processing pair {new_pairs_count + 1}/{goal}: {group_id}")

                # 1. Generar Texto Fake (English)
                fake_data = self.generate_fake_text(real_data["title"], real_data["content"])
                if not fake_data:
                    continue
                
                # 2. Generar Imagen Fake
                rel_image_path = self.generate_fake_image(fake_data.headline, group_id)
                if not rel_image_path:
                    continue

                # 3. Guardar registros (Pairwise format)
                entry_real = {
                    "group_id": group_id,
                    "is_real": 1,
                    "title": real_data["title"],
                    "content": real_data["content"],
                    "image_path": real_data["image_url"],
                    "model": "human"
                }

                entry_fake = {
                    "group_id": group_id,
                    "is_real": 0,
                    "title": fake_data.headline,
                    "content": fake_data.content,
                    "image_path": rel_image_path,
                    "technique": fake_data.technique,
                    "model": "gemini-2.0-flash-lite"
                }

                with open(FINAL_DATASET_FILE, "a", encoding="utf-8") as out:
                    out.write(json.dumps(entry_real, ensure_ascii=False) + "\n")
                    out.write(json.dumps(entry_fake, ensure_ascii=False) + "\n")
                
                self.processed_ids.add(group_id)
                new_pairs_count += 1
                
                # Courtesy delay para no saturar APIs
                time.sleep(1)

        print(f"Finished. Generated {new_pairs_count} new pairs.")

class TextOnlyGenerator(DatasetGenerator):
    def process_pipeline(self, goal=10):
        if not REAL_NEWS_FILE.exists():
            print(f"Source file not found: {REAL_NEWS_FILE}")
            return

        new_pairs_count = 0
        with open(REAL_NEWS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if new_pairs_count >= goal: break
                
                real_data = json.loads(line)
                group_id = real_data["article_id"]
                
                if group_id in self.processed_ids: continue

                print(f"[*] Processing text for pair {new_pairs_count + 1}/{goal}: {group_id}")

                fake_text = self.generate_fake_text(real_data["title"], real_data["content"])
                if not fake_text: continue

                entry_real = {
                    "group_id": group_id, "is_real": 1, "title": real_data["title"],
                    "content": real_data["content"], "image_path": real_data["image_url"], "model": "human"
                }

                entry_fake = {
                    "group_id": group_id, "is_real": 0, "title": fake_text.headline,
                    "content": fake_text.content, "image_path": None, # Marcamos como pendiente
                    "technique": fake_text.technique, "model": "gemini-2.0-flash-lite"
                }

                with open(FINAL_DATASET_FILE, "a", encoding="utf-8") as out:
                    out.write(json.dumps(entry_real, ensure_ascii=False) + "\n")
                    out.write(json.dumps(entry_fake, ensure_ascii=False) + "\n")
                
                self.processed_ids.add(group_id)
                new_pairs_count += 1
                time.sleep(1)


class ImageBackfiller:
    def __init__(self):
        self.hf_client = InferenceClient(api_key=os.getenv('FIRST_HF_TK'))

    def run(self):
        temp_file = FINAL_DATASET_FILE.with_suffix(".tmp")
        updated_count = 0
        stop_execution = False

        with open(FINAL_DATASET_FILE, "r", encoding="utf-8") as f_in, \
            open(temp_file, "w", encoding="utf-8") as f_out:
            
            for line in f_in:
                data = json.loads(line)
                
                # Si no hemos sido bloqueados por la API y falta la imagen
                if not stop_execution and data["is_real"] == 0 and (data["image_path"] is None or data["image_path"] == ""):
                    print(f"[*] Generating missing image for: {data['group_id']}")
                    file_name = f"{data['group_id']}_fake.png"
                    img_path = IMAGES_DIR / file_name
                    
                    try:
                        if not img_path.exists():
                            image = self.hf_client.text_to_image(
                                prompt=f"Professional news photo: {data['title']}",
                                model="black-forest-labs/FLUX.1-schnell",
                            )
                            image.save(img_path)
                        
                        data["image_path"] = f"dataset/fake_images/{file_name}"
                        updated_count += 1
                    except Exception as e:
                        print(f"❌ Error en imagen {data['group_id']}: {e}")
                        if "402" in str(e):
                            print("🛑 Cuota agotada. Guardando progreso y saliendo...")
                            stop_execution = True # Marcamos para dejar de intentar, pero seguimos copiando el resto del archivo
                
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        temp_file.replace(FINAL_DATASET_FILE)
        print(f"✅ Proceso completado. Imágenes añadidas: {updated_count}")

if __name__ == "__main__":
    # Puedes cambiar el número aquí para controlar cada ejecución
    GEN_GOAL = 55
    #gen = DatasetGenerator()
    #gen.process_pipeline(goal=GEN_GOAL)

    gen = TextOnlyGenerator()
    gen.process_pipeline(goal=GEN_GOAL)

    #filler = ImageBackfiller()
    #filler.run()