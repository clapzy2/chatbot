from unsloth import FastLanguageModel
import torch

# Загружаем дообученную модель
print("🔄 Загружаем дообученную модель...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "./mistral-finetuned-final",
    max_seq_length=2048,
    load_in_4bit=True,
)

def ask_trained_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Убираем оригинальный промпт из ответа
    if prompt in answer:
        answer = answer[len(prompt):].strip()
    return answer