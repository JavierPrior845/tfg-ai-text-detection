import json
from pathlib import Path

def repair_dataset(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"Archivo no encontrado: {file_path}")
        return

    repaired_lines = []
    print(f"🛠️ Iniciando reparación de {path.name}...")

    with open(path, "r", encoding="utf-8") as f:
        # Leemos todo y limpiamos caracteres nulos que Git o crashes suelen insertar
        content = f.read().replace('\0', '')
    
    # Separamos objetos pegados: cada '}{' debería ser '}\n{'
    fixed_content = content.replace('}{', '}\n{')
    
    for i, line in enumerate(fixed_content.split('\n')):
        line = line.strip()
        if not line: continue
        try:
            json.loads(line) # Validamos formato JSON
            repaired_lines.append(line)
        except json.JSONDecodeError:
            print(f"⚠️ Omitiendo fragmento corrupto en línea {i+1}: {line[:50]}...")

    # Sobreescribimos con la versión limpia
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(repaired_lines) + "\n")
    
    print(f"✅ Reparación finalizada. {len(repaired_lines)} líneas rescatadas.")

if __name__ == "__main__":
    repair_dataset("dataset/multimodal_dataset_fixed.jsonl")