import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

print("🔄 Загружаем модель...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer.pad_token = tokenizer.eos_token

print("📚 Загружаем датасет...")
with open("formatted_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def format_conversation(item):
    messages = item["conversations"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = Dataset.from_list(data)
dataset = dataset.map(format_conversation)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("🚀 Начинаем обучение...")
trainer.train()

print("💾 Сохраняем модель...")
model.save_pretrained("./mistral-finetuned-final")
tokenizer.save_pretrained("./mistral-finetuned-final")

print("✅ Обучение завершено!")