import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

print("🔄 Загружаем модель в 4-bit...")
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.gradient_checkpointing_enable()

# LoRA конфигурация
lora_config = LoraConfig(
    r=16,                      # увеличил ранг
    lora_alpha=32,             # увеличил alpha
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("📚 Загружаем датасет...")
with open("formatted_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Создаём текст для обучения
def create_text(item):
    messages = item["conversations"]
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]
    text = f"<s>[INST] {user_content} [/INST] {assistant_content} </s>"
    return {"text": text}

dataset = Dataset.from_list(data)
dataset = dataset.map(create_text)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["conversations", "text"])
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})

# Улучшенные параметры обучения
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    num_train_epochs=5,              # увеличил эпохи
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=20,                 # больше прогрева
    logging_steps=5,
    save_steps=50,
    learning_rate=1e-4,              # уменьшил learning rate
    fp16=True,
    logging_dir="./logs",
    report_to="none",
    save_total_limit=2,
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