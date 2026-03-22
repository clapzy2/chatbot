from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import warnings
import traceback
import subprocess
import re
from collections import Counter

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Глобальные переменные
retriever = None
vectorstore = None
available_files = set()

# Выбери модель здесь! Убедись, что она скачана через `ollama list`
MODEL_NAME = "mistral:7b"  # или "qwen2.5:7b-instruct", "gemma2:9b", "llama3.1:8b"

print("🔄 Загружаем базу данных...")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

    # Получаем список всех файлов
    all_docs = vectorstore.get()
    available_files = set()
    for metadata in all_docs['metadatas']:
        if metadata:
            file_name = metadata.get('source_file')
            if not file_name:
                source = metadata.get('source', '')
                file_name = os.path.basename(source) if source else None
            if file_name and file_name != 'Неизвестно':
                available_files.add(file_name)

    print(f"✅ База данных загружена. Найдено файлов: {len(available_files)}")
    print(f"📚 Доступные файлы: {', '.join(sorted(available_files))}")
    print(f"🤖 Используется модель: {MODEL_NAME}")

except Exception as e:
    print(f"❌ Ошибка загрузки базы: {e}")
    retriever = None
    vectorstore = None
    available_files = set()

def extract_keywords(text, top_n=5):
    """Извлекает ключевые слова из текста (имена, значимые слова)"""
    stop_words = {'это', 'что', 'как', 'так', 'вот', 'там', 'тут', 'его', 'её', 'их', 'вас', 'вам', 'для', 'без', 'было', 'будет', 'когда', 'тогда', 'потом', 'теперь'}
    words = re.findall(r'[А-Яа-яёЁA-Za-z]{4,}', text)
    words = [w.lower() for w in words if w.lower() not in stop_words]
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]

def filter_docs_by_keywords(docs, keywords):
    """Оставляет только те документы, которые содержат хотя бы одно ключевое слово"""
    if not keywords:
        return docs[:5]
    filtered = []
    for doc in docs:
        content = doc.page_content.lower()
        if any(kw in content for kw in keywords):
            filtered.append(doc)
    if len(filtered) < 2 and len(docs) > 2:
        filtered.extend(docs[:2])
    return filtered[:5]

def ask_ollama(prompt):
    """Отправляет промпт в Ollama"""
    try:
        process = subprocess.Popen(
            ['ollama', 'run', MODEL_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt, timeout=120)
        if process.returncode == 0:
            return stdout.strip()
        else:
            return f"Ошибка: {stderr}"
    except Exception as e:
        return f"Исключение: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/files')
def get_files():
    if vectorstore is None:
        return jsonify([])
    return jsonify(sorted(list(available_files)))

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '')
        selected_file = data.get('file', 'all')

        if not query:
            return jsonify({'error': 'Пустой запрос'}), 400

        # === Приветствия ===
        greetings = ['привет', 'здравствуй', 'здравствуйте', 'добрый день', 'добрый вечер', 'как дела']
        is_greeting = any(g in query.lower() for g in greetings)
        if is_greeting:
            prompt = "Ты — дружелюбный ассистент. Ответь на приветствие кратко и вежливо."
            answer = ask_ollama(prompt)
            return jsonify({
                'answer': answer,
                'sources': [],
                'fragments': [],
                'fragments_count': 0
            })

        if retriever is None or vectorstore is None:
            return jsonify({'error': 'База данных не загружена'}), 500

        # === Поиск фрагментов ===
        if selected_file == 'all':
            docs = retriever.invoke(query)
        else:
            filter_criteria = {"source_file": selected_file}
            docs = vectorstore.similarity_search(query, k=30, filter=filter_criteria)

        if not docs:
            return jsonify({
                'answer': 'НЕТ ИНФОРМАЦИИ',
                'sources': [],
                'fragments': [],
                'fragments_count': 0
            })

        # Убираем дубликаты
        unique_docs = []
        seen_content = set()
        for doc in docs:
            content_key = doc.page_content[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append(doc)
        docs = unique_docs

        # Источники
        sources = []
        for doc in docs:
            file_name = doc.metadata.get('source_file')
            if not file_name:
                source_path = doc.metadata.get('source', '')
                file_name = os.path.basename(source_path) if source_path else 'Неизвестно'
            if file_name and file_name != 'Неизвестно':
                sources.append(file_name)
        sources = list(set(sources))

        # Контекст (ограничиваем)
        context = "\n\n".join([doc.page_content for doc in docs])
        if len(context) > 8000:
            context = context[:8000]

        # === Промпт ===
        prompt = f"""Ты — система извлечения фактов. Твоя задача — ответить на вопрос, используя ТОЛЬКО информацию из контекста.

ШАГ 1. Внимательно прочитай контекст.
ШАГ 2. Найди в контексте информацию, относящуюся к вопросу.
ШАГ 3. Если информация найдена — сформулируй краткий ответ (1-2 предложения).
ШАГ 4. Если информации нет — напиши ТОЛЬКО: "НЕТ ИНФОРМАЦИИ".

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""

        answer = ask_ollama(prompt)

        # === Фильтрация фрагментов для отображения ===
        if "нет информации" in answer.lower() or "не знаю" in answer.lower():
            fragments = []
        else:
            keywords = extract_keywords(answer)
            relevant_docs = filter_docs_by_keywords(docs, keywords)
            fragments = [doc.page_content[:300] + '...' for doc in relevant_docs]

        return jsonify({
            'answer': answer,
            'sources': sources,
            'fragments': fragments,
            'fragments_count': len(docs)
        })

    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)