import json
import os
import time
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== НАСТРОЙКИ ==========
TEXT_FILE = "docs/s1.txt"  # файл с текстом
OUTPUT_FILE = "dataset.json"  # куда сохранить датасет
QUESTIONS_PER_CHUNK = 3  # сколько вопросов на фрагмент

# ========== ЗАГРУЖАЕМ МОДЕЛЬ ==========
print("🤖 Загружаем Mistral для генерации вопросов...")
llm = Ollama(model="mistral:7b-instruct")  # ← ИСПРАВЛЕНО!

# ========== ЧИТАЕМ ТЕКСТ ==========
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

print(f"📄 Загружен текст: {len(text)} символов")

# ========== РАЗБИВАЕМ НА ФРАГМЕНТЫ ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_text(text)
print(f"🔪 Разбито на {len(chunks)} фрагментов")

# ========== ГЕНЕРИРУЕМ ВОПРОСЫ ==========
dataset = []

for i, chunk in enumerate(chunks):
    if len(chunk.strip()) < 100:
        continue

    print(f"\n📝 Обрабатываю фрагмент {i + 1}/{len(chunks)}...")
    print(f"   {chunk[:150]}...")

    prompt = f"""Ты — эксперт по созданию обучающих данных. Прочитай фрагмент текста и сгенерируй {QUESTIONS_PER_CHUNK} вопросов с ответами.
Вопросы должны проверять понимание ключевых деталей: что произошло, кто, когда, почему, сколько и т.д.
Ответы должны быть КРАТКИМИ (1-2 предложения) и строго из текста.

Фрагмент:
{chunk}

Формат вывода (строго):
Вопрос 1: ...
Ответ 1: ...
Вопрос 2: ...
Ответ 2: ...
Вопрос 3: ...
Ответ 3: ...
"""

    try:
        response = llm.invoke(prompt)

        # Парсим ответ
        lines = response.strip().split('\n')
        for j in range(len(lines)):
            if lines[j].startswith('Вопрос'):
                question = lines[j].replace('Вопрос', '').strip(': ').strip()
                # Ищем следующий ответ
                for k in range(j + 1, min(j + 5, len(lines))):
                    if lines[k].startswith('Ответ'):
                        answer = lines[k].replace('Ответ', '').strip(': ').strip()
                        dataset.append({
                            "question": question,
                            "answer": answer,
                            "context": chunk
                        })
                        break

        print(f"   ✅ Сгенерировано вопросов для этого фрагмента")

        # Небольшая пауза, чтобы не перегружать модель
        time.sleep(1)

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

# ========== СОХРАНЯЕМ ==========
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n🎉 ГОТОВО! Создано {len(dataset)} пар вопрос-ответ")
print(f"📁 Файл сохранён: {OUTPUT_FILE}")

# Показываем примеры
print("\n📖 ПРИМЕРЫ:")
for i, item in enumerate(dataset[:5]):
    print(f"{i + 1}. Вопрос: {item['question']}")
    print(f"   Ответ: {item['answer']}\n")