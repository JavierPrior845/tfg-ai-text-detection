import os
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

# Definimos el contrato de datos
class SyntheticNews(BaseModel):
    headline: str
    content: str
    technique: str

def run_generation(real_headline: str):
    # Inicializamos el nuevo cliente
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
        # En el nuevo SDK, se usa models.generate
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite', # Usamos la última versión disponible en 2026
            contents='que es un prompt',
            config={
                'response_mime_type': 'application/json',
            }
        )

        # Acceso directo al objeto parseado
        data = response.parsed
        print("\n--- NOTICIA GENERADA CON ÉXITO ---")
        print(f"Titular: {data.headline}")
        print(f"Técnica: {data.technique}")
        return data

    except Exception as e:
        print(f"Error en la generación: {e}")

if __name__ == "__main__":
    titular_test = "El IBEX 35 cierra en verde tras la reunión del Eurogrupo"
    run_generation(titular_test)