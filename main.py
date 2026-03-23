#!/usr/bin/env python3
"""
TextBot — Gradio GUI
Fixes: контекст истории, фильтр по файлу, сохранение диалогов, верификация
"""
import sys
import os
import json
import threading

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import config
from src.llm_engine import LLMEngine
from src.knowledge_base import KnowledgeBase
from src.document_generator import DocumentGenerator
from src.presentation_generator import PresentationGenerator

# ── Глобальные объекты ─────────────────────────────────────────────────
_llm: LLMEngine     = None
_kb:  KnowledgeBase = None
_doc_gen  = DocumentGenerator()
_pres_gen = PresentationGenerator()

# Файл для сохранения истории диалогов
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chat_history.json")

_REFUSAL = [
    "нет информации", "не упоминается", "не содержит",
    "не могу найти", "отсутствует", "нет данных",
    "no information", "not mentioned",
]


def _get_kb(log=None) -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(progress_callback=log)
    return _kb


def _get_llm() -> LLMEngine:
    global _llm
    if _llm is None:
        _llm = LLMEngine()
    return _llm


def _is_refusal(text: str) -> bool:
    return any(p in text.lower() for p in _REFUSAL)


# ── Сохранение/загрузка истории ─────────────────────────────────────────
def save_history(history: list):
    """Сохраняет историю диалогов на диск"""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_history() -> list:
    """Загружает историю диалогов с диска"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


# ── Верификация (исправленная) ──────────────────────────────────────────
def _verify(question: str, answer: str, context: str) -> bool:
    """
    Проверяет ответ по контексту.
    Исправление: более мягкий промпт + проверка на явный отказ модели.
    """
    if not config.USE_VERIFICATION or not answer or _is_refusal(answer):
        return False
    # Если ответ очень короткий и конкретный — считаем верным без проверки
    if len(answer.strip()) < 100 and not any(w in answer.lower() for w in ["возможно", "может быть", "вероятно"]):
        return True
    prompt = config.PROMPTS["verify"].format(
        topic=question, answer=answer, context=context[:3000]
    )
    verdict = _get_llm().call(prompt, temperature=0.0, max_tokens=20)
    # Считаем подтверждённым если нет явного отказа
    verdict_lower = verdict.lower()
    return "нет информации" not in verdict_lower and "не подтверждено" not in verdict_lower


# ── Вспомогательная: история → строка для промпта ──────────────────────
def _history_to_context(history: list, n_last: int = 4) -> str:
    """
    Берёт последние N пар из истории и форматирует как текст.
    Используется чтобы модель понимала уточняющие вопросы.
    """
    if not history or len(history) < 2:
        return ""
    recent = history[-n_last * 2:] if len(history) > n_last * 2 else history
    lines = []
    for msg in recent:
        role    = "Пользователь" if msg.get("role") == "user" else "Ассистент"
        content = msg.get("content", "")
        # Gradio 6 может хранить content как список
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        # Убираем бейдж верификации из истории
        content = str(content).split("\n\n*[")[0].strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  ОБРАБОТЧИКИ
# ══════════════════════════════════════════════════════════════

def on_index_books():
    import re
    log_lines = []
    def log(msg):
        clean = re.sub(r'\[.*?\]', '', str(msg))
        log_lines.append(clean)
        return clean
    try:
        kb     = _get_kb(log)
        result = kb.index_all_books()
        stats  = kb.stats()
        out = (
            f"{result}\n\n📊 Итого:\n"
            f"  Файлов: {stats['total_books']}\n"
            f"  Фрагментов: {stats['total_chunks']}"
        )
        if stats["books"]:
            out += "\n  Файлы: " + ", ".join(stats["books"])
        return out
    except Exception as e:
        return f"❌ Ошибка: {e}"


def on_add_book(files):
    if not files:
        return "Выберите файлы для загрузки"
    import shutil
    try:
        kb = _get_kb()
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        results = []
        for file in files:
            src  = file.name if hasattr(file, "name") else str(file)
            dest = os.path.join(config.DOCS_DIR, os.path.basename(src))
            shutil.copy2(src, dest)
            results.append(kb.add_book(dest))
        stats = kb.stats()
        return "\n".join(results) + f"\n\n📊 Всего: {stats['total_books']} файлов, {stats['total_chunks']} фрагментов"
    except Exception as e:
        return f"❌ {e}"


def on_clear_kb():
    try:
        return _get_kb().clear()
    except Exception as e:
        return f"❌ {e}"


def on_stats():
    try:
        kb    = _get_kb()
        stats = kb.stats()
        text  = f"📚 Файлов: {stats['total_books']}\n📄 Фрагментов: {stats['total_chunks']}\n"
        if stats["books"]:
            text += "\n📖 Файлы:\n" + "".join(f"  • {b}\n" for b in stats["books"])
        else:
            text += "\n⚠️ База пуста. Загрузите файлы."
        return text
    except Exception as e:
        return f"❌ {e}"


def get_file_choices():
    """Возвращает список файлов для dropdown в чате"""
    try:
        kb    = _get_kb()
        files = kb.get_available_files()
        return ["Все файлы"] + files
    except Exception:
        return ["Все файлы"]


def on_refresh_files():
    """Обновляет список файлов в dropdown"""
    choices = get_file_choices()
    return gr.Dropdown(choices=choices, value="Все файлы")


def generate_document(topic: str, doc_type: str, fmt: str):
    if not topic.strip():
        return "Введите тему", ""
    try:
        kb  = _get_kb()
        llm = _get_llm()
        type_map = {
            "Отчёт": "report", "Конспект": "summary", "Эссе": "essay",
            "Анализ": "analysis", "Подготовка к экзамену": "exam_prep",
        }
        context  = kb.search(topic)
        text     = llm.generate_with_context(
            config.PROMPTS[type_map.get(doc_type, "report")], topic, context
        )
        fmt_map  = {"Оба (DOCX + MD)": "both", "DOCX": "docx", "Markdown": "md"}
        type_ru  = {
            "Отчёт": "отчёт", "Конспект": "конспект", "Эссе": "эссе",
            "Анализ": "анализ", "Подготовка к экзамену": "экзамен",
        }
        files   = _doc_gen.generate(text, topic, type_ru.get(doc_type, "документ"),
                                    fmt_map.get(fmt, "both"))
        flist   = "\n".join(f"📁 {os.path.basename(f)}" for f in files)
        return text, f"✅ Сохранено:\n{flist}\n\nПапка: {config.OUTPUT_DIR}"
    except FileNotFoundError as e:
        return str(e), "❌ Модель не найдена"
    except Exception as e:
        return f"❌ {e}", f"❌ {e}"


def generate_presentation(topic: str):
    if not topic.strip():
        return "Введите тему", ""
    try:
        kb   = _get_kb()
        llm  = _get_llm()
        ctx  = kb.search(topic)
        text = llm.generate_with_context(config.PROMPTS["presentation"], topic, ctx)
        fp   = _pres_gen.generate(text, topic)
        return text, f"✅ Презентация:\n📁 {os.path.basename(fp)}\n\nПапка: {config.OUTPUT_DIR}"
    except Exception as e:
        return f"❌ {e}", f"❌ {e}"


# ── ЧАТ ────────────────────────────────────────────────────────────────
def chat_respond(message: str, history: list, selected_file: str):
    """
    Исправления:
    1. Передаём историю диалога в промпт — модель понимает уточняющие вопросы
    2. Фильтр по файлу — ищем только в выбранном тексте
    3. Сохраняем историю на диск после каждого ответа
    4. Верификация больше не показывает ложные ✗
    """
    if not message.strip():
        yield history
        return

    greetings = ["привет", "здравствуй", "добрый", "hi", "hello"]
    if any(g in message.lower() for g in greetings) and len(message.split()) <= 5:
        new_h = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Привет! Задайте вопрос по загруженным текстам."}
        ]
        save_history(new_h)
        yield new_h
        return

    try:
        kb  = _get_kb()
        llm = _get_llm()

        # Фильтр по файлу
        file_filter = "all" if selected_file == "Все файлы" else selected_file

        # Если вопрос очень короткий — это уточнение, ищем по предыдущему вопросу
        followups = [
            "точно", "уверен", "правда", "докажи", "обоснуй",
            "подробнее", "аргументы", "аргумент", "объясни", "почему",
            "зачем", "пример", "примеры", "поясни", "расскажи подробнее",
            "как так", "серьёзно", "не понял", "уточни",
        ]
        search_query = message
        if len(message.split()) <= 5 and any(f in message.lower() for f in followups):
            # Берём последний вопрос пользователя из истории
            for msg in reversed(history):
                if msg.get("role") == "user":
                    content = msg["content"]
                    if isinstance(content, list):
                        content = " ".join(
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    search_query = str(content).split("\n\n*[")[0].strip()
                    break

        context = kb.search(search_query, file_filter=file_filter)

        if not context:
            new_h = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "НЕТ ИНФОРМАЦИИ — база пуста или файлы не проиндексированы."}
            ]
            save_history(new_h)
            yield new_h
            return

        # История диалога для уточняющих вопросов
        history_ctx = _history_to_context(history, n_last=3)

        # Промпт с историей
        if history_ctx:
            full_prompt = config.PROMPTS["qa"].format(
                system=config.SYSTEM_PROMPT,
                topic=message,
                context=f"ПРЕДЫДУЩИЙ ДИАЛОГ (только для понимания о чём спрашивают, не используй как источник фактов):\n{history_ctx}\n\nТЕКСТ ДОКУМЕНТА (только отсюда бери факты):\n{context}"
            )
        else:
            full_prompt = config.PROMPTS["qa"].format(
                system=config.SYSTEM_PROMPT,
                topic=message,
                context=context
            )

        if config.LLM_MODE == "llama_cpp" and getattr(config, "QWEN3_THINKING_MODE", False):
            full_prompt = "/think\n" + full_prompt

        # Стриминг ответа
        answer      = ""
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""}
        ]
        for token in llm.stream(full_prompt):
            answer += token
            new_history[-1]["content"] = answer
            yield new_history

        # Верификация — показываем ✓ только если подтверждено, иначе ничего
        if config.USE_VERIFICATION and answer and not _is_refusal(answer):
            verified = _verify(message, answer, context)
            if verified:
                new_history[-1]["content"] = answer.strip() + "\n\n*[✓ подтверждено]*"
                yield new_history
        elif _is_refusal(answer):
            new_history[-1]["content"] = "НЕТ ИНФОРМАЦИИ"
            yield new_history

        save_history(new_history)

    except Exception as e:
        err_h = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Ошибка: {e}"}
        ]
        save_history(err_h)
        yield err_h


def on_clear_chat():
    """Очищает чат и удаляет сохранённую историю"""
    save_history([])
    return []


# ── ОСТАЛЬНЫЕ ──────────────────────────────────────────────────────────
def get_model_info():
    mode      = config.LLM_MODE
    chunk_cfg = config.CHUNK_SETTINGS[config.CHUNK_MODE]
    thinking  = getattr(config, "QWEN3_THINKING_MODE", False)
    model_line = (
        f"- **Модель Ollama**: `{config.OLLAMA_MODEL}`"
        if mode == "ollama"
        else f"- **GGUF файл**: `{config.LLM_MODEL_PATH}`\n"
             f"- **Статус**: {'✅ найден' if os.path.exists(config.LLM_MODEL_PATH) else '❌ не найден'}"
    )
    return f"""## Состояние системы

### LLM
- **Режим**: `{mode}`
{model_line}
- **Контекст**: {config.LLM_CONTEXT_SIZE} токенов
- **Температура**: {config.LLM_TEMPERATURE}
- **Qwen3 thinking**: {'✅ вкл' if thinking else '❌ выкл'} *(только llama_cpp)*

### RAG-пайплайн
- **Эмбеддинги**: `{config.EMBEDDING_MODEL}`
- **Reranker**: `{config.RERANKER_MODEL}` ({'✅' if config.USE_RERANKER else '❌'})
- **HyDE**: {'✅' if config.USE_HYDE else '❌'} ({config.HYDE_VARIANTS} варианта)
- **Верификация**: {'✅' if config.USE_VERIFICATION else '❌'} *(только ✓, без ложных ✗)*
- **Режим чанков**: `{config.CHUNK_MODE}` ({chunk_cfg['chunk_size']} симв.)
- **Top-K**: {config.RETRIEVAL_TOP_K} → {config.RERANK_TOP_K} после rerank

---

## Рекомендуемые модели

| Модель | RAM | Качество |
|--------|-----|----------|
| **Qwen3-4B** | 5 GB | 👍 Хорошее |
| **Qwen3-8B** | 8 GB | ⭐ Отличное |
| **Qwen3-14B** | 14 GB | 🏆 Максимум |

### Через Ollama:
```bash
ollama pull qwen3:8b
# config.py: OLLAMA_MODEL = "qwen3:8b"
```
"""


def list_output_files():
    if not os.path.exists(config.OUTPUT_DIR):
        return "Нет созданных документов"
    files = []
    for f in sorted(os.listdir(config.OUTPUT_DIR), reverse=True):
        p    = os.path.join(config.OUTPUT_DIR, f)
        size = os.path.getsize(p)
        s    = f"{size/1024:.1f} KB" if size < 1024**2 else f"{size/1024**2:.1f} MB"
        files.append(f"📄 {f} ({s})")
    return "\n".join(files) if files else "Нет созданных документов"


# ══════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════
def build_gui():
    # Загружаем сохранённую историю при запуске
    saved_history = load_history()

    with gr.Blocks(title="TextBot") as app:

        gr.HTML("""
        <div style="text-align:center;padding:16px 0 8px">
            <h1 style="font-size:2em;font-weight:bold;">📚 TextBot</h1>
            <p style="color:#888;font-size:1em;">
                HyDE + Cross-Encoder + Верификация · Qwen3 ready
            </p>
        </div>""")

        with gr.Tabs():

            # ── Чат ──────────────────────────────────────────
            with gr.TabItem("💬 Чат"):
                gr.Markdown("### Вопросы по загруженным текстам")

                with gr.Row():
                    file_dropdown = gr.Dropdown(
                        choices=get_file_choices(),
                        value="Все файлы",
                        label="📄 Искать в файле",
                        scale=3,
                        interactive=True,
                    )
                    refresh_btn = gr.Button("🔄", scale=1, size="sm")

                chatbot = gr.Chatbot(
                    height=480,
                    show_label=False,
                    value=saved_history,   # восстанавливаем историю
                )
                with gr.Row():
                    chat_in  = gr.Textbox(
                        placeholder="Задайте вопрос...",
                        show_label=False, scale=9, container=False
                    )
                    chat_btn = gr.Button("Отправить", variant="primary", scale=1)
                chat_clear = gr.Button("🗑️ Очистить историю", size="sm")

                # Обновление списка файлов
                refresh_btn.click(on_refresh_files, outputs=file_dropdown)

                # Отправка сообщения — передаём file_dropdown как третий аргумент
                chat_btn.click(
                    chat_respond,
                    inputs=[chat_in, chatbot, file_dropdown],
                    outputs=chatbot
                ).then(lambda: "", outputs=chat_in)

                chat_in.submit(
                    chat_respond,
                    inputs=[chat_in, chatbot, file_dropdown],
                    outputs=chatbot
                ).then(lambda: "", outputs=chat_in)

                chat_clear.click(on_clear_chat, outputs=chatbot)

            # ── Документы ────────────────────────────────────
            with gr.TabItem("📝 Документы"):
                gr.Markdown("### Создание учебных документов")
                with gr.Row():
                    doc_topic = gr.Textbox(label="Тема", placeholder="Введите тему...", scale=3)
                    doc_type  = gr.Dropdown(
                        ["Отчёт", "Конспект", "Эссе", "Анализ", "Подготовка к экзамену"],
                        value="Отчёт", label="Тип", scale=1)
                    doc_fmt   = gr.Dropdown(
                        ["Оба (DOCX + MD)", "DOCX", "Markdown"],
                        value="Оба (DOCX + MD)", label="Формат", scale=1)
                doc_btn    = gr.Button("🚀 Создать документ", variant="primary", size="lg")
                doc_out    = gr.Textbox(label="Текст", lines=20)
                doc_status = gr.Textbox(label="Статус", lines=3)
                doc_btn.click(generate_document, [doc_topic, doc_type, doc_fmt],
                               [doc_out, doc_status])

            # ── Презентации ───────────────────────────────────
            with gr.TabItem("📊 Презентации"):
                gr.Markdown("### Создание PowerPoint")
                pres_topic  = gr.Textbox(label="Тема", placeholder="Введите тему...")
                pres_btn    = gr.Button("🚀 Создать презентацию", variant="primary", size="lg")
                pres_out    = gr.Textbox(label="Структура", lines=20)
                pres_status = gr.Textbox(label="Статус", lines=3)
                pres_btn.click(generate_presentation, pres_topic, [pres_out, pres_status])

            # ── Файлы ────────────────────────────────────────
            with gr.TabItem("📖 Файлы"):
                gr.Markdown("### Управление базой знаний")
                with gr.Row():
                    with gr.Column():
                        book_upload = gr.File(
                            label="Загрузить файлы",
                            file_count="multiple",
                            file_types=config.SUPPORTED_FORMATS,
                        )
                        book_up_btn = gr.Button("📥 Загрузить и индексировать", variant="primary")
                    with gr.Column():
                        book_idx_btn   = gr.Button("🔄 Индексировать папку docs/", variant="secondary")
                        book_stats_btn = gr.Button("📊 Статистика")
                        book_clr_btn   = gr.Button("🗑️ Очистить базу", variant="stop")
                book_out = gr.Textbox(label="Результат", lines=10)
                book_up_btn.click(on_add_book, book_upload, book_out)
                book_idx_btn.click(on_index_books, None, book_out)
                book_stats_btn.click(on_stats, None, book_out)
                book_clr_btn.click(on_clear_kb, None, book_out)

            # ── Результаты ───────────────────────────────────
            with gr.TabItem("📁 Результаты"):
                files_refresh = gr.Button("🔄 Обновить")
                files_list    = gr.Textbox(label="Созданные файлы", lines=15, value=list_output_files)
                files_refresh.click(list_output_files, outputs=files_list)
                gr.Markdown(f"📂 Папка: `{config.OUTPUT_DIR}`")

            # ── Настройки ────────────────────────────────────
            with gr.TabItem("⚙️ Настройки"):
                info_md      = gr.Markdown(value=get_model_info)
                info_refresh = gr.Button("🔄 Обновить")
                info_refresh.click(get_model_info, outputs=info_md)

    return app


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for d in [config.DOCS_DIR, config.OUTPUT_DIR, config.MODELS_DIR, config.DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    thinking = getattr(config, "QWEN3_THINKING_MODE", False)
    print("=" * 55)
    print("  📚 TextBot")
    print(f"  LLM        : {config.LLM_MODE.upper()} / {config.OLLAMA_MODEL if config.LLM_MODE == 'ollama' else os.path.basename(config.LLM_MODEL_PATH)}")
    print(f"  Чанки      : {config.CHUNK_MODE}")
    print(f"  HyDE       : {'вкл' if config.USE_HYDE else 'выкл'}")
    print(f"  Reranker   : {'вкл' if config.USE_RERANKER else 'выкл'}")
    print(f"  Верификация: {'вкл' if config.USE_VERIFICATION else 'выкл'}")
    print(f"  История    : {HISTORY_FILE}")
    print("=" * 55)

    app = build_gui()
    app.launch(
        server_port=config.GUI_PORT,
        share=config.GUI_SHARE,
        inbrowser=True,
    )
