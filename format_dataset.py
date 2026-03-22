import json

# Загружаем датасет
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Форматируем в формат для Mistral
formatted_data = []

for item in data:
    formatted = {
        "conversations": [
            {
                "role": "user",
                "content": f"""Контекст:
{item['context']}

Вопрос: {item['question']}"""
            },
            {
                "role": "assistant",
                "content": item['answer']
            }
        ]
    }
    formatted_data.append(formatted)

# Сохраняем
with open("formatted_dataset.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

print(f"✅ Создано {len(formatted_data)} примеров для обучения")