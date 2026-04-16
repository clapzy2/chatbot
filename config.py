"""
config.py - все настройки системы в одном месте.
Меняя этот файл, можно переключать режимы работы без изменения кода.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папки проекта
DOCS_DIR   = os.path.join(BASE_DIR, "docs")
DATA_DIR   = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")

# Режим работы LLM: "api" (облако) или "ollama" (локально)
LLM_MODE = "api"

# Настройки API (OpenRouter)
API_URL   = "https://openrouter.ai/api/v1/chat/completions"
API_KEY  = "sk-or-v1-..." # вставить в свой ключ
API_MODEL = "qwen/qwen3-32b" # можно брать разные модельки

# Настройки Ollama (локальный режим)
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# Параметры генерации
LLM_MAX_TOKENS     = 2048
LLM_TEMPERATURE    = 0.1
LLM_TOP_P         = 0.9
LLM_REPEAT_PENALTY = 1.15
LLM_CONTEXT_SIZE   = 32768

# Эмбеддинги
EMBEDDING_MODEL  = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"

# Реранкер
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
USE_RERANKER   = True
RERANK_TOP_K   = 7 # потесить еще

# ChromaDB
COLLECTION_NAME = "textbot_docs"

# Чанкинг
CHUNK_SIZE    = 1200 # потесить еще
CHUNK_OVERLAP = 200 # потесить еще

# Поиск
RETRIEVAL_TOP_K = 20
MIN_RELEVANCE   = 0.10
MAX_CTX_CHARS   = 8000

# HyDE
USE_HYDE      = True
HYDE_VARIANTS = 3

# Форматы файлов
SUPPORTED_FORMATS = [
    ".pdf", ".txt", ".epub", ".docx",
    ".md", ".fb2", ".fb2.zip", ".html", ".htm", # добавить еще
]

# Веб-интерфейс
GUI_PORT  = 7860
GUI_SHARE = False

# Системный промпт
SYSTEM_PROMPT = """Ты - точный ассистент для работы с текстами и учебными материалами.

Принципы:
1. ТОЧНОСТЬ - используй ТОЛЬКО факты из предоставленного контекста
2. ПОЛНОТА - внимательно проверь ВСЕ предоставленные фрагменты
3. СТРУКТУРА - чёткие логичные тексты с академической структурой
4. ЧЕСТНОСТЬ - если информации нет в контексте, прямо скажи об этом
5. ЯЗЫК - грамотный русский, академический стиль"""

# Промпты
PROMPTS = {
    "qa": """{system}

КОНТЕКСТ:
{context}

ВОПРОС: {topic}

Инструкция:
1. Внимательно прочитай ВСЕ фрагменты контекста
2. Найди информацию, относящуюся к вопросу
3. НЕ придумывай цитаты - текст должен быть дословно из контекста
4. Если ответа нет в контексте - напиши: НЕТ ИНФОРМАЦИИ

ОТВЕТ:""",

    "correction": """{system}

КОНТЕКСТ:
{context}

ПРЕДЫДУЩИЙ ВОПРОС: {prev_question}
ПРЕДЫДУЩИЙ ОТВЕТ: {prev_answer}
ЗАМЕЧАНИЕ ПОЛЬЗОВАТЕЛЯ: {correction}

Пользователь указал, что предыдущий ответ неправильный.
Перечитай ВСЕ фрагменты контекста и найди правильный ответ.
Если не можешь найти ответ - напиши: НЕТ ИНФОРМАЦИИ

ИСПРАВЛЕННЫЙ ОТВЕТ:""",

    "hyde": """Перефразируй вопрос {n} разными способами для поиска в тексте.
Каждый вариант - отдельная строка. Без нумерации, без пояснений.

Вопрос: {query}

Варианты:""",
}