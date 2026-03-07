import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

# 1. Carga del Dataset desde el archivo JSONL
# Asumiendo que tu archivo se llama 'data.jsonl'
dataset = load_dataset("json", data_files="data.jsonl", split="train")

# Renombramos 'is_real' a 'label' para que el modelo lo reconozca automáticamente
dataset = dataset.rename_column("is_real", "label")

# 2. Configuración del Tokenizer para DeBERTa-v3
MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
# use_fast=False es necesario para DeBERTa-v3 por su tokenizador SentencePiece
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=False)

def tokenize_function(examples):
    # Padding=False aquí porque el DataCollator lo hará dinámicamente por batch
    return tokenizer(examples["title"], truncation=True, max_length=128)

# Aplicamos el tokenizado de forma eficiente (batched)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Inicialización del Modelo para Clasificación Binaria
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, 
    num_labels=2
)

# 4. Configuración de Entrenamiento (Training Loop)
training_args = TrainingArguments(
    output_dir="./deberta-v3-detector",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(), # Aceleración por hardware
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# 5. Data Collator para Padding Dinámico
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Definición del Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 7. Ejecución
trainer.train()

# 8. Guardado del modelo resultante
trainer.save_model("./final_model")