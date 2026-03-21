"""
imageProcessor.py
-----------------
Enriquece titles_data.jsonl con:
  - img_path:  URL (para reales) o ruta local en dataset/fake_images/ (para sintéticas)
  - img_text:  texto generado a partir de la imagen (img-to-text via Gemini)

Las noticias se procesan SIEMPRE como pares (real + fake). Si alguna falla, no se escribe ninguna.
"""

import os
import json
import time
import httpx
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient

load_dotenv()

# ─── RUTAS ───────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
ROOT_DIR   = BASE_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"

TITLES_FILE      = DATASET_DIR / "titles_data.jsonl"
REAL_NEWS_FILE   = ROOT_DIR / "scraping" / "data_collection" / "real_news_no_duplicates.jsonl"
OUTPUT_FILE      = DATASET_DIR / "titles_img_data.jsonl"
FAKE_IMAGES_DIR  = DATASET_DIR / "fake_images"

FAKE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ─── HELPERS ─────────────────────────────────────────────────────────
def load_processed_ids() -> set:
    """Devuelve los group_id ya presentes en el dataset de salida."""
    ids = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["group_id"])
    return ids


def load_real_news_index() -> dict:
    """Indexa article_id → image_url del fichero de noticias reales."""
    index = {}
    with open(REAL_NEWS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                index[data["article_id"]] = data.get("image_url", "")
    return index


def group_titles(path: Path) -> dict:
    """
    Agrupa las entradas del JSONL de títulos por group_id.
    Devuelve  {group_id: {"real": {...}, "fake": {...}}}
    """
    groups: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            gid = entry["group_id"]
            if gid not in groups:
                groups[gid] = {}
            if entry["is_real"] == 1:
                groups[gid]["real"] = entry
            else:
                groups[gid]["fake"] = entry
    return groups


# ─── PROCESADOR ──────────────────────────────────────────────────────
class ImageProcessor:
    def __init__(self):
        self.gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.hf     = InferenceClient(api_key=os.getenv("FIRST_HF_TK"))

    # ── Img-to-text ──────────────────────────────────────────────────
    def img_to_text_from_url(self, image_url: str) -> str | None:
        """Describe una imagen referenciada por URL usando Gemini multimodal."""
        prompt = (
            "Describe this news image in a single detailed paragraph. "
            "Focus on the subjects, setting, actions, and any visible text or logos."
        )
        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    types.Part.from_uri(file_uri=image_url, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(max_output_tokens=300),
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"  ⚠️  img-to-text URL error: {e}")
            return None

    def img_to_text_from_file(self, file_path: Path) -> str | None:
        """Describe una imagen local usando Gemini multimodal (upload inline)."""
        prompt = (
            "Describe this news image in a single detailed paragraph. "
            "Focus on the subjects, setting, actions, and any visible text or logos."
        )
        try:
            img_bytes = file_path.read_bytes()
            mime = "image/png"
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    prompt,
                ],
                config=types.GenerateContentConfig(max_output_tokens=300),
            )
            return response.text.strip() if response.text else None
        except Exception as e:
            print(f"  ⚠️  img-to-text file error: {e}")
            return None

    # ── Generación de imagen ─────────────────────────────────────────
    def generate_fake_image(self, headline: str, group_id: str) -> Path | None:
        """Genera imagen con FLUX y la guarda en fake_images/."""
        file_name = f"{group_id}_fake.png"
        img_path  = FAKE_IMAGES_DIR / file_name

        if img_path.exists():
            return img_path

        try:
            image = self.hf.text_to_image(
                prompt=f"Professional photojournalism, high quality, realistic news photo: {headline}",
                model="black-forest-labs/FLUX.1-schnell",
            )
            image.save(img_path)
            return img_path
        except Exception as e:
            print(f"  ⚠️  Image generation error: {e}")
            return None

    # ── Pipeline principal ───────────────────────────────────────────
    def run(self, goal: int | None = None):
        print("═" * 60)
        print("  IMAGE PROCESSOR – Titles → Img-to-Text Dataset")
        print("═" * 60)

        # 1. Cargar datos
        processed_ids   = load_processed_ids()
        real_news_index = load_real_news_index()
        title_groups    = group_titles(TITLES_FILE)

        total_groups = len(title_groups)
        skipped      = 0
        ok_count     = 0
        fail_count   = 0

        print(f"  Total pares en titles_data : {total_groups}")
        print(f"  Ya procesados             : {len(processed_ids)}")
        if goal:
            print(f"  Objetivo esta ejecución   : {goal}")
        print("─" * 60)

        for idx, (gid, pair) in enumerate(title_groups.items(), start=1):
            # ── Control de ejecución ──
            if goal and ok_count >= goal:
                print(f"\n✅ Objetivo alcanzado: {ok_count} pares nuevos.")
                break

            # ── Ya procesado ──
            if gid in processed_ids:
                skipped += 1
                continue

            # ── Verificar que el par esté completo ──
            if "real" not in pair or "fake" not in pair:
                print(f"  [{idx}/{total_groups}] ❌ Par incompleto para {gid}, saltando.")
                fail_count += 1
                continue

            real_entry = pair["real"]
            fake_entry = pair["fake"]

            print(f"  [{idx}/{total_groups}] Procesando par: {gid[:16]}…")

            # ═══════════════ REAL ═══════════════
            image_url = real_news_index.get(gid, "")
            if not image_url:
                print(f"    → Sin image_url para la real. Par descartado.")
                fail_count += 1
                continue

            real_img_text = self.img_to_text_from_url(image_url)
            if not real_img_text:
                print(f"    → img-to-text REAL falló. Par descartado.")
                fail_count += 1
                continue

            # ═══════════════ FAKE ═══════════════
            fake_img_path = self.generate_fake_image(fake_entry["title"], gid)
            if not fake_img_path:
                print(f"    → Generación de imagen FAKE falló. Par descartado.")
                fail_count += 1
                continue

            fake_img_text = self.img_to_text_from_file(fake_img_path)
            if not fake_img_text:
                print(f"    → img-to-text FAKE falló. Par descartado.")
                fail_count += 1
                continue

            # ═══════════════ ESCRIBIR PAR ═══════════════
            out_real = {
                "group_id":  gid,
                "title":     real_entry["title"],
                "is_real":   1,
                "img_path":  image_url,
                "img_text":  real_img_text,
            }
            out_fake = {
                "group_id":  gid,
                "title":     fake_entry["title"],
                "is_real":   0,
                "img_path":  f"dataset/fake_images/{gid}_fake.png",
                "img_text":  fake_img_text,
            }

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_real, ensure_ascii=False) + "\n")
                f.write(json.dumps(out_fake, ensure_ascii=False) + "\n")

            processed_ids.add(gid)
            ok_count += 1
            print(f"    ✅ Par guardado ({ok_count} nuevos)")

            # Delay cortesía entre pares
            time.sleep(2)

        # ── Resumen final ──
        print("\n" + "═" * 60)
        print(f"  RESUMEN")
        print(f"    Pares nuevos escritos : {ok_count}")
        print(f"    Pares descartados     : {fail_count}")
        print(f"    Pares saltados (ya ok): {skipped}")
        print("═" * 60)


# ─── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GOAL = 100   # Cambia este valor para controlar cuántos pares procesar
    processor = ImageProcessor()
    processor.run(goal=GOAL)
