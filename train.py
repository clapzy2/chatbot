import torch
import json
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

print("🔄 Загружаем модель...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("🔧 Добавляем LoRA адаптеры...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

print("📚 Загружаем датасет...")
with open("formatted_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def format_conversation(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_conversation)

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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

print("🚀 Начинаем обучение...")
trainer.train()

print("💾 Сохраняем модель...")
model.save_pretrained("./mistral-finetuned-final")
tokenizer.save_pretrained("./mistral-finetuned-final")

print("✅ Обучение завершено!")