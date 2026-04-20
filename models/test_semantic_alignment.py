import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@torch.no_grad()
def encode_texts(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def cosine_sim(a, b):
    return (a @ b.T).item()

df = pd.read_json("dataset/titles_img_data.jsonl", lines=True)

pairs = {}
for _, row in df.iterrows():
    gid = row["group_id"]
    label = "real" if row["is_real"] == 1 else "fake"
    if gid not in pairs:
        pairs[gid] = {}
    pairs[gid][label] = row.to_dict()

complete_pairs = {k: v for k, v in pairs.items() if "real" in v and "fake" in v}
print("Pares completos:", len(complete_pairs))

results = []
for gid, pair in tqdm(list(complete_pairs.items())):
    r_row = pair["real"]
    f_row = pair["fake"]
    
    t_real = r_row["title"]
    t_sint = f_row["title"]
    i_real = r_row["img_text"]
    i_sint = f_row["img_text"]
    
    texts = [t_real, t_sint, i_real, i_sint]
    emb = encode_texts(texts)
    e_t_real = emb[0:1]
    e_t_sint = emb[1:2]
    e_i_real = emb[2:3]
    e_i_sint = emb[3:4]
    
    sim_A_real = cosine_sim(e_i_real, e_t_real)
    sim_A_sint = cosine_sim(e_i_real, e_t_sint)
    
    sim_B_real = cosine_sim(e_i_sint, e_t_real)
    sim_B_sint = cosine_sim(e_i_sint, e_t_sint)
    
    results.append({
        "group_id": gid,
        "delta_A": sim_A_real - sim_A_sint,
        "delta_B": sim_B_sint - sim_B_real,
        "sim_A_real": sim_A_real,
        "sim_A_sint": sim_A_sint,
        "sim_B_real": sim_B_real,
        "sim_B_sint": sim_B_sint,
    })

res_df = pd.DataFrame(results)

wins_A_real = (res_df["delta_A"] >= 0).sum()
wins_B_sint = (res_df["delta_B"] >= 0).sum()

total = len(res_df)

print("ENFOQUE A - Imagen_Text Real")
print(f"Gana T_Real: {wins_A_real/total:.2%}")

print("\nENFOQUE B - Imagen_Text Sintetica")
print(f"Gana T_Sint: {wins_B_sint/total:.2%}")

y_true = []
y_pred = []

for _, row in res_df.iterrows():
    pred_real_is_winner = row["delta_A"] >= 0
    
    y_true.append(1)
    y_pred.append(1 if pred_real_is_winner else 0)
    
    y_true.append(0)
    y_pred.append(0 if pred_real_is_winner else 1)

print("\nAccuracy Enfoque A:")
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["Sintético", "Real"]))
