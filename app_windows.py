from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests, os, re, warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ═══════════════════════════════════════════════
#  НАСТРОЙКИ
# ═══════════════════════════════════════════════
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"
DB_DIR       = "./chroma_db"
EMBED_MODEL  = "intfloat/multilingual-e5-large"  # должна совпадать с ingest.py!

TOP_K         = 10     # фрагментов из базы на каждый вариант запроса
RERANK_TOP_K  = 5      # оставляем после rerank
MAX_CTX_CHARS = 5000   # увеличен для аналитических вопросов
MIN_RELEVANCE = 0.20   # порог для multilingual-e5 (выдаёт более высокие score)


# ═══════════════════════════════════════════════
#  ЗАГРУЗКА БАЗЫ
# ═══════════════════════════════════════════════
vectorstore     = None
available_files = set()

print("🔄 Загружаем базу...")
try:
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=emb)

    for meta in vectorstore.get()["metadatas"]:
        if not meta:
            continue
        fname = meta.get("source_file") or os.path.basename(meta.get("source", ""))
        if fname:
            available_files.add(fname)

    print(f"✅ База: {vectorstore._collection.count()} фрагментов, файлов: {len(available_files)}")
except Exception as e:
    print(f"❌ {e}")


# ═══════════════════════════════════════════════
#  OLLAMA
# ═══════════════════════════════════════════════
# Отключаем системный прокси для локальных запросов (частая проблема на Windows)
os.environ["NO_PROXY"]    = "localhost,127.0.0.1"
os.environ["no_proxy"]    = "localhost,127.0.0.1"

def call_ollama(prompt: str, temperature: float = 0.05, max_tokens: int = 600) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature":    temperature,
                    "top_p":          0.85,
                    "num_predict":    max_tokens,
                    "repeat_penalty": 1.1,
                    "stop":           ["ВОПРОС:", "КОНТЕКСТ:", "---", "\n\n\n"]
                }
            },
            timeout=150
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ОШИБКА_СОЕДИНЕНИЯ"
    except Exception as e:
        return f"ОШИБКА: {e}"


# ═══════════════════════════════════════════════
#  HYDE — расширение запроса
# ═══════════════════════════════════════════════
def expand_query(query: str) -> list[str]:
    """
    HyDE (Hypothetical Document Embeddings):
    Просим модель переформулировать вопрос тремя способами.
    Каждый вариант ищем отдельно — итог объединяем.
    Это сильно помогает при сложных аналитических вопросах.
    """
    prompt = f"""Перефразируй вопрос тремя разными способами для поиска в тексте.
Каждый вариант — отдельная строка. Без нумерации, без пояснений.

Вопрос: {query}

Варианты:"""

    result = call_ollama(prompt, temperature=0.4, max_tokens=120)

    if result.startswith("ОШИБКА"):
        return [query]  # fallback — используем оригинальный запрос

    variants = [line.strip() for line in result.split("\n") if line.strip()]
    # Всегда добавляем оригинал первым
    all_variants = [query] + variants[:3]
    return all_variants


# ═══════════════════════════════════════════════
#  ПОИСК: гибридный (семантика + ключевые слова) + HyDE
# ═══════════════════════════════════════════════
def search_docs(query: str, file_filter: str, use_hyde: bool = True):
    """
    1. Генерируем несколько вариантов запроса (HyDE)
    2. По каждому ищем в базе
    3. Объединяем, убираем дубли
    4. Keyword rerank — поднимаем фрагменты с нужными словами
    """
    kw_filter = {"source_file": file_filter} if file_filter != "all" else None

    # Варианты запроса
    queries = expand_query(query) if use_hyde else [query]

    # Собираем кандидатов из всех вариантов
    seen_ids  = set()
    candidates = []

    for q in queries:
        try:
            results = vectorstore.similarity_search_with_relevance_scores(
                q, k=TOP_K, filter=kw_filter
            )
            for doc, score in results:
                doc_id = doc.page_content[:100]  # псевдо-ID по началу текста
                if doc_id not in seen_ids and score >= MIN_RELEVANCE:
                    seen_ids.add(doc_id)
                    candidates.append((doc, score))
        except Exception as e:
            print(f"Ошибка поиска для '{q}': {e}")

    # Если ничего не нашли с порогом — без порога, лучшие 3
    if not candidates:
        try:
            results = vectorstore.similarity_search_with_relevance_scores(
                query, k=3, filter=kw_filter
            )
            candidates = [(d, s) for d, s in results]
        except Exception:
            return [], [], []

    # Keyword rerank
    query_words = set(re.findall(r"[А-Яа-яёЁA-Za-z]{3,}", query.lower()))

    def keyword_score(text: str) -> float:
        t = text.lower()
        hits = sum(1 for w in query_words if w in t)
        return hits / max(len(query_words), 1)

    scored = []
    for doc, sem_sc in candidates:
        kw_sc    = keyword_score(doc.page_content)
        combined = 0.65 * sem_sc + 0.35 * kw_sc
        scored.append((doc, combined, sem_sc))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:RERANK_TOP_K]

    docs      = [d for d, _, _ in top]
    scores    = [round(c, 3) for _, c, _ in top]
    sem_scores = [round(s, 3) for _, _, s in top]
    return docs, scores, sem_scores


# ═══════════════════════════════════════════════
#  ПРОМПТЫ
# ═══════════════════════════════════════════════
def make_answer_prompt(query: str, context: str) -> str:
    return f"""Ты — точный ассистент для работы с текстами. Отвечай СТРОГО по контексту.

Правила:
1. Используй ТОЛЬКО информацию из КОНТЕКСТА.
2. Если вопрос аналитический — синтезируй ответ из нескольких частей контекста.
3. Если вопрос с вариантами ответа (А/Б/В/Г) — выбери правильный вариант и кратко объясни.
4. Если информации нет — ответь ровно: НЕТ ИНФОРМАЦИИ
5. Не придумывай ничего от себя.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""


def make_verify_prompt(query: str, answer: str, context: str) -> str:
    return f"""Проверь: подтверждается ли ОТВЕТ информацией из КОНТЕКСТА?

КОНТЕКСТ:
{context}

ВОПРОС: {query}
ОТВЕТ: {answer}

Если ответ точный и подтверждается контекстом — напиши только: ПОДТВЕРЖДЕНО
Если в ответе есть что-то, чего нет в контексте — напиши только: НЕТ ИНФОРМАЦИИ"""


# ═══════════════════════════════════════════════
#  УТИЛИТЫ
# ═══════════════════════════════════════════════
REFUSAL = [
    "нет информации", "не упоминается", "не содержит", "не могу найти",
    "отсутствует", "нет данных", "no information", "not mentioned",
    "в контексте нет", "не найдено"
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in REFUSAL)

def build_context(docs) -> str:
    parts, total = [], 0
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        # Добавляем маркер источника — помогает модели при синтезе
        header = f"[Фрагмент {i+1} | {doc.metadata.get('source_file', '?')}]"
        block  = f"{header}\n{text}"

        if total + len(block) > MAX_CTX_CHARS:
            left = MAX_CTX_CHARS - total
            if left < 200:
                break
            cut   = block[:left].rfind(". ")
            block = block[:cut + 1] if cut > 50 else block[:left]

        parts.append(block)
        total += len(block)
        if total >= MAX_CTX_CHARS:
            break

    return "\n\n---\n\n".join(parts)


def empty_response(answer="НЕТ ИНФОРМАЦИИ"):
    return jsonify({
        "answer": answer, "verified": False,
        "sources": [], "fragments": [], "fragments_count": 0,
        "keywords": [], "scores": [], "sem_scores": [], "queries_used": []
    })


# ═══════════════════════════════════════════════
#  МАРШРУТЫ
# ═══════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/files")
def get_files():
    return jsonify(sorted(available_files))

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data          = request.get_json()
        query         = (data.get("query") or "").strip()
        selected_file = data.get("file", "all")

        if not query:
            return jsonify({"error": "Пустой запрос"}), 400

        # Приветствия — не лезем в базу
        greetings = ["привет", "здравствуй", "добрый", "hi", "hello"]
        if any(g in query.lower() for g in greetings) and len(query.split()) <= 5:
            return jsonify({
                "answer": "Привет! Задайте вопрос по загруженным текстам.",
                "verified": False, "sources": [], "fragments": [],
                "fragments_count": 0, "keywords": [], "scores": [],
                "sem_scores": [], "queries_used": []
            })

        if vectorstore is None:
            return jsonify({"error": "База данных не загружена"}), 500

        # ── Поиск ───────────────────────────────
        docs, scores, sem_scores = search_docs(query, selected_file, use_hyde=True)

        if not docs:
            return empty_response()

        context = build_context(docs)

        # ── Первый проход: ответ ─────────────────
        answer = call_ollama(make_answer_prompt(query, context))

        if answer == "ОШИБКА_СОЕДИНЕНИЯ":
            return jsonify({"error": "Ollama не запущена. Выполните: ollama serve"}), 503
        if answer.startswith("ОШИБКА:"):
            return jsonify({"error": answer}), 503

        # ── Второй проход: верификация ────────────
        verified = False
        if answer and not is_refusal(answer):
            # Для верификации берём только топ-3 фрагмента (меньше = точнее проверка)
            verify_ctx = build_context(docs[:3])
            verdict    = call_ollama(
                make_verify_prompt(query, answer, verify_ctx),
                temperature=0.0,
                max_tokens=15
            )
            verified = "подтверждено" in verdict.lower()
            if not verified:
                answer = "НЕТ ИНФОРМАЦИИ"

        if is_refusal(answer):
            answer   = "НЕТ ИНФОРМАЦИИ"
            verified = False

        # ── Формируем ответ ──────────────────────
        fragments = []
        if verified:
            fragments = [
                {
                    "text":  doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
                    "file":  doc.metadata.get("source_file", "?"),
                    "score": scores[i]
                }
                for i, doc in enumerate(docs[:4])
            ]

        sources   = list({doc.metadata.get("source_file", "?") for doc in docs})
        stop_w    = {"это", "что", "как", "так", "вот", "там", "тут", "для",
                     "или", "если", "было", "были", "чтобы", "который"}
        keywords  = [w.lower() for w in re.findall(r"[А-Яа-яёЁA-Za-z]{4,}", query)
                     if w.lower() not in stop_w][:6]

        return jsonify({
            "answer":          answer,
            "verified":        verified,
            "sources":         sources,
            "fragments":       fragments,
            "fragments_count": len(docs),
            "keywords":        keywords,
            "scores":          scores,
            "sem_scores":      sem_scores,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
