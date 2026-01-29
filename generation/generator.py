import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

class SyntheticNews(BaseModel):
    headline: str
    content: str
    technique: str

def run_generation(real_headline: str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
    Eres un redactor de noticias experto.
    Crea una noticia sintética basada en este titular real: "{real_headline}"

    REGLAS:
    1. Estilo profesional y neutral (Castellano de España).
    2. Manipulación sutil de los hechos.
    3. Devuelve JSON con llaves: headline, content, technique.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",  # o el modelo que tengas habilitado
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SyntheticNews,  # <- usa tu Pydantic
                max_output_tokens=512,
            ),
        )

        data: SyntheticNews = response.parsed  # ya tipado
        print("\n--- NOTICIA GENERADA CON ÉXITO ---")
        print(f"Titular: {data.headline}")
        print(f"Técnica: {data.technique}")
        return data

    except Exception as e:
        print(f"Error en la generación: {e}")

if __name__ == "__main__":
    titular_test = "El IBEX 35 cierra en verde tras la reunión del Eurogrupo"
    run_generation(titular_test)
