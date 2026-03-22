from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import subprocess
import os
import re
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

retriever = None
vectorstore = None
available_files = set()

print("🔄 Загружаем базу данных...")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

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

    print(f"✅ База загружена. Файлов: {len(available_files)}")
except Exception as e:
    print(f"❌ Ошибка: {e}")


def ask_ollama(prompt):
    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'mistral:7b-instruct'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt, timeout=120)
        if process.returncode == 0:
            return stdout.strip()
        return f"Ошибка: {stderr}"
    except Exception as e:
        return f"Исключение: {e}"


def validate_answer(answer, context):
    """Проверяет, что все ключевые слова из ответа есть в контексте"""
    if not answer or "нет информации" in answer.lower():
        return False, "НЕТ ИНФОРМАЦИИ"

    words = re.findall(r'[А-Яа-яёЁA-Za-z]{4,}', answer)
    keywords = [w.lower() for w in words if w.lower() not in {'это', 'что', 'как', 'так', 'вот'}]

    if not keywords:
        return True, answer

    context_lower = context.lower()
    for kw in keywords:
        if kw not in context_lower:
            return False, "НЕТ ИНФОРМАЦИИ"
    return True, answer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/files')
def get_files():
    return jsonify(sorted(list(available_files)))


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '')
        selected_file = data.get('file', 'all')

        if not query:
            return jsonify({'error': 'Пустой запрос'}), 400

        # Приветствия
        greetings = ['привет', 'здравствуй', 'здравствуйте', 'добрый день', 'как дела']
        if any(g in query.lower() for g in greetings) and len(query.split()) <= 4:
            return jsonify({
                'answer': 'Привет! Задавайте вопросы по тексту.',
                'sources': [],
                'fragments': [],
                'fragments_count': 0,
                'keywords': []
            })

        if retriever is None:
            return jsonify({'error': 'База не загружена'}), 500

        # Поиск фрагментов
        if selected_file == 'all':
            docs = retriever.invoke(query)
        else:
            docs = vectorstore.similarity_search(query, k=20, filter={"source_file": selected_file})

        if not docs:
            return jsonify({
                'answer': 'НЕТ ИНФОРМАЦИИ',
                'sources': [],
                'fragments': [],
                'fragments_count': 0,
                'keywords': []
            })

        # Убираем дубликаты
        unique = []
        seen = set()
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        docs = unique

        # Контекст
        context = "\n\n".join([doc.page_content for doc in docs])
        if len(context) > 3000:
            context = context[:3000]

        prompt = f"""Ты — эксперт по текстам. Ответь на вопрос, используя ТОЛЬКО информацию из контекста.
Если в контексте нет ответа, скажи "НЕТ ИНФОРМАЦИИ".

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""

        answer = ask_ollama(prompt)
        is_valid, validated_answer = validate_answer(answer, context)
        if not is_valid:
            answer = validated_answer
            fragments = []
        else:
            fragments = [doc.page_content[:300] + '...' for doc in docs[:5]]

        sources = list(set([doc.metadata.get('source_file', 'Неизвестно') for doc in docs]))

        stop_words = {'это', 'что', 'как', 'так', 'вот', 'там', 'тут'}
        words = re.findall(r'[А-Яа-яёЁA-Za-z]{4,}', query)
        keywords = [w.lower() for w in words if w.lower() not in stop_words][:5]

        return jsonify({
            'answer': answer,
            'sources': sources,
            'fragments': fragments,
            'fragments_count': len(docs),
            'keywords': keywords
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)