import os
import json
from dotenv import load_dotenv
import requests

load_dotenv()

def test_newsdata_api():
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        print("NEWSDATA_API_KEY no encontrada en .env")
        return
    
    print("Probando NewsData.io API...")
    
    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": api_key,
        "category": "business",
        "language": "en",
        "size": 5,
        "removeduplicate": 1
    }
    
    resp = requests.get(url, params=params)
    
    print(f"Satatus: {resp.status_code}")
    
    if resp.status_code == 200:
        try:
            data = resp.json()
            print("¡¡API OK!!")
            
            # Guardar respuesta cruda
            with open("api_test_response.json", "w") as f:
                json.dump(data, f, indent=2)
            
            print("Respuesta guardada: api_test_response.json")
            print(f"\nResumen:")
            print(f"   Total results: {len(data.get('results', []))}")
            print(f"   Next page: {'Sí' if data.get('nextPage') else 'No'}")
            
        except json.JSONDecodeError as e:
            print(f"Error JSON: {e}")
    else:
        print(f"Error: {resp.text}")

if __name__ == "__main__":
    test_newsdata_api()