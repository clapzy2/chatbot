from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import re
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

print("🔄 Загружаем базовую модель и адаптер...")

# Настройки 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Загружаем базовую модель Mistral-7B
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Загружаем LoRA-адаптер (дообученные веса)
model_dir = "./mistral-finetuned-final"
model = PeftModel.from_pretrained(base_model, model_dir)

print("✅ Модель загружена")

def ask_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in answer:
        answer = answer[len(prompt):].strip()
    return answer

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

        unique = []
        seen = set()
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        docs = unique

        context = "\n\n".join([doc.page_content for doc in docs])
        if len(context) > 2000:
            context = context[:2000]

        prompt = f"""Контекст:
{context}

Вопрос: {query}

Ответ:"""

        answer = ask_model(prompt)

        sources = list(set([doc.metadata.get('source_file', 'Неизвестно') for doc in docs]))
        fragments = [doc.page_content[:300] + '...' for doc in docs[:5]]

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