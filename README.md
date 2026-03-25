# 📚 TextBot

RAG-бот для работы с учебными текстами. Задавай вопросы по загруженным документам — бот находит ответ в тексте и подтверждает цитатами.

## Возможности

- **Точные ответы** по загруженным текстам (PDF, TXT, EPUB, DOCX, FB2, HTML)
- **Контекстный чанкинг** — автоматически определяет разделы/главы в тексте
- **Фильтр по разделам** — "Ответь из Речи Федра" → ищет только в нужном разделе
- **HyDE** — переформулирует вопрос для лучшего поиска
- **Реранкер** — перерасположение результатов для максимальной точности
- **Стриминг** — ответ печатается в реальном времени
- **Мультипользовательский** — у каждого своя история чата
- **Мобильный** — работает с телефона через публичную ссылку

## Стек

| Компонент | Технология |
|-----------|-----------|
| LLM | Qwen3-32B через OpenRouter API |
| Эмбеддинги | BAAI/bge-m3 (мультиязычный) |
| Реранкер | BAAI/bge-reranker-v2-m3 |
| Векторная БД | ChromaDB |
| GUI | Gradio |

## Быстрый старт

### 1. Клонируй репозиторий
```bash
git clone https://github.com/clapzy2/chatbot.git
cd chatbot
```

### 2. Установи зависимости
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Получи API ключ
- Зарегистрируйся на [OpenRouter](https://openrouter.ai)
- Создай API ключ в [Settings → Keys](https://openrouter.ai/settings/keys)
- Вставь ключ в `config.py`:
```python
API_KEY = "sk-or-..."
```

### 4. Запусти
```bash
python main.py
```

Бот откроется в браузере на `http://localhost:7860`.
Публичная ссылка для друзей появится в консоли (если `GUI_SHARE = True`).

### 5. Загрузи тексты
- Положи файлы в папку `docs/`
- Или загрузи через вкладку "📖 Файлы" в интерфейсе
- Нажми "Индексировать"

## Структура проекта

```
chatbot/
├── config.py              # Конфигурация (модели, параметры, промпты)
├── main.py                # Gradio GUI + обработчики чата
├── ingest.py              # Индексация через командную строку (опционально)
├── requirements.txt       # Зависимости
├── docs/                  # Папка для текстов
├── data/
│   └── chromadb/          # Векторная база (создаётся автоматически)
└── src/
    ├── __init__.py
    ├── knowledge_base.py  # RAG: чанкинг, эмбеддинги, поиск, реранкинг
    ├── llm_engine.py      # LLM: API / Ollama / llama-cpp
    ├── document_generator.py  # Генерация DOCX/MD
    └── presentation_generator.py  # Генерация PPTX
```

## Конфигурация

Всё настраивается в `config.py`:

| Параметр | Описание | По умолчанию |
|----------|----------|-------------|
| `LLM_MODE` | `"api"` / `"ollama"` / `"llama_cpp"` | `"api"` |
| `API_MODEL` | Модель на OpenRouter | `"qwen/qwen3-32b"` |
| `CHUNK_SIZE` | Размер чанка в символах | `1200` |
| `RETRIEVAL_TOP_K` | Сколько кандидатов искать | `20` |
| `RERANK_TOP_K` | Сколько оставить после реранкинга | `7` |
| `USE_HYDE` | Включить HyDE | `True` |
| `GUI_SHARE` | Публичная ссылка | `True` |

## Локальный режим (без API)

Если не хочешь использовать API — поставь Ollama:

```bash
# Установи Ollama: https://ollama.com
ollama pull qwen3:8b
```

В `config.py`:
```python
LLM_MODE = "ollama"
OLLAMA_MODEL = "qwen3:8b"
```

## Лицензия

MIT
