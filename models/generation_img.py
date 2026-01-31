import requests
import os
from dotenv import load_dotenv

load_dotenv()

model_id = "kandinsky-community/kandinsky-2-2-decoder"
api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

headers = {
    "Authorization": f"Bearer {os.getenv('FIRST_HF_TK')}",
    "Content-Type": "application/json"
}

prompt = "Un paisaje de fantasía con castillo y dragones, ultra detallado"

payload = {
    "inputs": prompt,
    "parameters": {
        "guidance_scale": 7.5,
        "num_inference_steps": 50
    }
}

response = requests.post(api_url, headers=headers, json=payload)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("Imagen generada y guardada en output.png")
else:
    print("Error:", response.status_code, response.text)
