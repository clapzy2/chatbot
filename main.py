#!/usr/bin/env python3
"""
TextBot v3.0 — Gradio GUI

- Отдельная история для каждого пользователя (gr.State)
- OpenRouter API
- Контекстный чанкинг + фильтр по разделам
- Коррекции и follow-ups
"""
import sys
import os
import gc
import json
import threading

os.environ["NO_PROXY"] = "localhost,127.0.0.1,api.groq.com,openrouter.ai"
os.environ["no_proxy"] = "localhost,127.0.0.1,api.groq.com,openrouter.ai"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import config
from src.llm_engine import LLMEngine
from src.knowledge_base import KnowledgeBase

# ── Глобальные объекты ─────────────────────────────────────────────────
_llm: LLMEngine     = None
_kb:  KnowledgeBase = None

_REFUSAL = [
    "нет информации", "не упоминается", "не содержит",
    "не могу найти", "отсутствует", "нет данных",
    "no information", "not mentioned",
]

_FOLLOWUPS = [
    "точно", "уверен", "правда", "докажи", "обоснуй",
    "подробнее", "аргументы", "аргумент", "объясни", "почему",
    "зачем", "пример", "примеры", "поясни", "расскажи подробнее",
    "как так", "серьёзно", "не понял", "уточни", "ещё",
    "а что", "а как", "а почему", "а зачем", "расскажи ещё",
    "продолжи", "дальше",
]

_CORRECTIONS = [
    "нет,", "неправильно", "неверно", "ошибка", "ты ошибся",
    "правильный ответ", "на самом деле", "не так", "неточно",
    "ты не прав", "это неправда", "некорректно", "нет это",
    "ответ неверный", "ответ неправильный", "нет правильный",
    "все-таки", "всё-таки", "однако нет", "а вот нет",
]


def _get_llm() -> LLMEngine:
    global _llm
    if _llm is None:
        _llm = LLMEngine()
    return _llm


def _get_kb(log=None) -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(progress_callback=log, llm_engine=_get_llm())
    elif _kb._llm is None:
        _kb.set_llm(_get_llm())
    return _kb


def _is_refusal(text: str) -> bool:
    lower = text.lower().strip()
    if not lower:
        return True
    if len(lower) < 150:
        return any(p in lower for p in _REFUSAL)
    return any(p in lower[:100] for p in _REFUSAL)


def _is_correction(text: str) -> bool:
    return any(c in text.lower().strip() for c in _CORRECTIONS)


def _is_followup(text: str) -> bool:
    return len(text.split()) <= 7 and any(f in text.lower().strip() for f in _FOLLOWUPS)


def _gradio_chatbot_kwargs() -> dict:
    major = gr.__version__.split(".")[0]
    return {"type": "messages"} if major == "5" else {}


def _extract_content(msg) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content).split("\n\n*[")[0].strip()


def _history_to_context(history: list, n_last: int = 4) -> str:
    if not history or len(history) < 2:
        return ""
    recent = history[-n_last * 2:] if len(history) > n_last * 2 else history
    lines = []
    for msg in recent:
        role = "Пользователь" if msg.get("role") == "user" else "Ассистент"
        content = _extract_content(msg)
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _get_last_qa(history: list):
    last_answer, last_question = "", ""
    for msg in reversed(history):
        role = msg.get("role", "")
        content = _extract_content(msg)
        if role == "assistant" and not last_answer:
            last_answer = content
        elif role == "user" and not last_question:
            last_question = content
            break
    return last_question, last_answer


def _verify(question: str, answer: str, context: str) -> bool:
    if not config.USE_VERIFICATION or not answer or _is_refusal(answer):
        return False
    if len(answer.strip()) < 100 and not any(w in answer.lower() for w in ["возможно", "может быть", "вероятно"]):
        return True
    prompt = config.PROMPTS["verify"].format(topic=question, answer=answer, context=context[:4000])
    verdict = _get_llm().call(prompt, temperature=0.0, max_tokens=30)
    return "нет информации" not in verdict.lower() and "не подтверждено" not in verdict.lower()


# ══════════════════════════════════════════════════════════════
#  ОБРАБОТЧИКИ
# ══════════════════════════════════════════════════════════════

def on_index_books():
    import re as _re
    log_lines = []
    def log(msg):
        log_lines.append(_re.sub(r'\[.*?\]', '', str(msg)))
        return log_lines[-1]
    try:
        kb = _get_kb(log)
        result = kb.index_all_books()
        stats = kb.stats()
        gc.collect()
        out = f"{result}\n\n📊 Итого:\n  Файлов: {stats['total_books']}\n  Фрагментов: {stats['total_chunks']}"
        if stats["books"]:
            out += "\n  Файлы: " + ", ".join(stats["books"])
        if stats.get("sections"):
            out += f"\n  Разделов: {len(stats['sections'])}"
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
            src = file.name if hasattr(file, "name") else str(file)
            dest = os.path.join(config.DOCS_DIR, os.path.basename(src))
            shutil.copy2(src, dest)
            results.append(kb.add_book(dest))
        gc.collect()
        stats = kb.stats()
        return "\n".join(results) + f"\n\n📊 Всего: {stats['total_books']} файлов, {stats['total_chunks']} фрагментов"
    except Exception as e:
        return f"❌ {e}"


def on_clear_kb():
    try:
        result = _get_kb().clear()
        gc.collect()
        return result
    except Exception as e:
        return f"❌ {e}"


def on_stats():
    try:
        stats = _get_kb().stats()
        text = f"📚 Файлов: {stats['total_books']}\n📄 Фрагментов: {stats['total_chunks']}\n"
        if stats["books"]:
            text += "\n📖 Файлы:\n" + "".join(f"  • {b}\n" for b in stats["books"])
        if stats.get("sections"):
            text += f"\n📑 Разделов: {len(stats['sections'])}"
        else:
            text += "\n⚠️ База пуста. Загрузите файлы."
        return text
    except Exception as e:
        return f"❌ {e}"


def get_file_choices():
    try:
        return ["Все файлы"] + _get_kb().get_available_files()
    except Exception:
        return ["Все файлы"]


def on_refresh_files():
    return gr.Dropdown(choices=get_file_choices(), value="Все файлы")


# ── ЧАТ (с отдельной историей для каждого пользователя) ────────────────
def chat_respond(message: str, history: list, selected_file: str):
    """
    history — это gr.Chatbot value, уникальный для каждой сессии браузера.
    Каждый пользователь видит только свою историю.
    """
    if not message.strip():
        yield history
        return

    greetings = ["привет", "здравствуй", "добрый", "hi", "hello"]
    if any(g in message.lower() for g in greetings) and len(message.split()) <= 5:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Привет! Задайте вопрос по загруженным текстам."}
        ]
        return

    try:
        kb  = _get_kb()
        llm = _get_llm()
        file_filter = "all" if selected_file == "Все файлы" else selected_file

        is_corr = _is_correction(message)
        is_fu   = _is_followup(message)
        search_query = message
        prev_question, prev_answer = "", ""

        if is_corr or is_fu:
            prev_question, prev_answer = _get_last_qa(history)
            if prev_question:
                search_query = prev_question

        section_filter = kb.find_section_in_query(message)
        context = kb.search(search_query, file_filter=file_filter, section_filter=section_filter)

        if not context:
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "НЕТ ИНФОРМАЦИИ — база пуста или файлы не проиндексированы."}
            ]
            return

        if is_corr and prev_question and prev_answer:
            full_prompt = config.PROMPTS["correction"].format(
                system=config.SYSTEM_PROMPT, context=context,
                prev_question=prev_question, prev_answer=prev_answer, correction=message,
            )
        else:
            history_ctx = _history_to_context(history, n_last=3)
            if history_ctx:
                full_prompt = config.PROMPTS["qa"].format(
                    system=config.SYSTEM_PROMPT, topic=message,
                    context=f"ПРЕДЫДУЩИЙ ДИАЛОГ:\n{history_ctx}\n\nТЕКСТ ДОКУМЕНТА:\n{context}"
                )
            else:
                full_prompt = config.PROMPTS["qa"].format(
                    system=config.SYSTEM_PROMPT, topic=message, context=context
                )

        answer = ""
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""}
        ]
        for token in llm.stream(full_prompt):
            answer += token
            new_history[-1]["content"] = answer
            yield new_history

        if config.USE_VERIFICATION and answer and not _is_refusal(answer):
            if _verify(message, answer, context):
                new_history[-1]["content"] = answer.strip() + "\n\n*[✓ подтверждено]*"
                yield new_history
        elif _is_refusal(answer):
            new_history[-1]["content"] = "НЕТ ИНФОРМАЦИИ"
            yield new_history

    except Exception as e:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Ошибка: {e}"}
        ]


# ── НАСТРОЙКИ ──────────────────────────────────────────────────────────
def get_model_info():
    mode = config.LLM_MODE
    if mode == "api":
        model_line = f"- **API модель**: `{getattr(config, 'API_MODEL', '?')}`\n- **API ключ**: {'✅ задан' if getattr(config, 'API_KEY', '') else '❌ не задан'}"
    elif mode == "ollama":
        model_line = f"- **Модель Ollama**: `{config.OLLAMA_MODEL}`"
    else:
        model_line = f"- **GGUF**: `{config.LLM_MODEL_PATH}`"
    return f"""## Состояние системы

### LLM
- **Режим**: `{mode}`
{model_line}

### RAG-пайплайн
- **Эмбеддинги**: `{config.EMBEDDING_MODEL}`
- **Реранкер**: `{config.RERANKER_MODEL}` ({'✅' if config.USE_RERANKER else '❌'})
- **HyDE**: {'✅' if config.USE_HYDE else '❌'} ({config.HYDE_VARIANTS} варианта)
- **Верификация**: {'✅' if config.USE_VERIFICATION else '❌'}
- **Чанк**: {config.CHUNK_SIZE} символов (overlap {config.CHUNK_OVERLAP})
- **Top-K**: {config.RETRIEVAL_TOP_K} → {config.RERANK_TOP_K} после rerank
"""


# ══════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════
def build_gui():
    chatbot_kwargs = _gradio_chatbot_kwargs()

    with gr.Blocks(title="TextBot") as app:

        gr.HTML("""
        <div style="text-align:center;padding:16px 0 8px">
            <h1 style="font-size:2em;font-weight:bold;">📚 TextBot</h1>
            <p style="color:#888;font-size:1em;">
                Умный ассистент по учебным текстам
            </p>
        </div>""")

        with gr.Tabs():

            with gr.TabItem("💬 Чат"):
                gr.Markdown("### Вопросы по загруженным текстам")

                with gr.Row():
                    file_dropdown = gr.Dropdown(
                        choices=get_file_choices(), value="Все файлы",
                        label="📄 Искать в файле", scale=3, interactive=True,
                    )
                    refresh_btn = gr.Button("🔄", scale=1, size="sm")

                chatbot = gr.Chatbot(height=480, show_label=False, **chatbot_kwargs)

                with gr.Row():
                    chat_in = gr.Textbox(
                        placeholder="Задайте вопрос...",
                        show_label=False, scale=9, container=False
                    )
                    chat_btn = gr.Button("Отправить", variant="primary", scale=1)
                chat_clear = gr.Button("🗑️ Очистить историю", size="sm")

                refresh_btn.click(on_refresh_files, outputs=file_dropdown)

                chat_btn.click(
                    chat_respond, inputs=[chat_in, chatbot, file_dropdown], outputs=chatbot
                ).then(lambda: "", outputs=chat_in)

                chat_in.submit(
                    chat_respond, inputs=[chat_in, chatbot, file_dropdown], outputs=chatbot
                ).then(lambda: "", outputs=chat_in)

                chat_clear.click(lambda: [], outputs=chatbot)

            with gr.TabItem("📖 Файлы"):
                gr.Markdown("### Управление базой знаний")
                with gr.Row():
                    with gr.Column():
                        book_upload = gr.File(label="Загрузить файлы", file_count="multiple",
                                              file_types=config.SUPPORTED_FORMATS)
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

            with gr.TabItem("⚙️ Настройки"):
                info_md = gr.Markdown(value=get_model_info)
                info_refresh = gr.Button("🔄 Обновить")
                info_refresh.click(get_model_info, outputs=info_md)

    return app


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for d in [config.DOCS_DIR, config.OUTPUT_DIR, config.MODELS_DIR, config.DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    mode = config.LLM_MODE
    if mode == "api":
        llm_label = f"API / {getattr(config, 'API_MODEL', '?')}"
    elif mode == "ollama":
        llm_label = f"OLLAMA / {config.OLLAMA_MODEL}"
    else:
        llm_label = f"LLAMA_CPP / {os.path.basename(config.LLM_MODEL_PATH)}"

    print("=" * 55)
    print("  📚 TextBot v3.0")
    print(f"  LLM        : {llm_label}")
    print(f"  Эмбеддинги : {config.EMBEDDING_MODEL}")
    print(f"  Реранкер   : {config.RERANKER_MODEL}")
    print(f"  Чанк       : {config.CHUNK_SIZE} симв.")
    print(f"  HyDE       : {'вкл' if config.USE_HYDE else 'выкл'}")
    print(f"  Верификация: {'вкл' if config.USE_VERIFICATION else 'выкл'}")
    print(f"  Share      : {'вкл' if config.GUI_SHARE else 'выкл'}")
    print(f"  Gradio     : v{gr.__version__}")
    print("=" * 55)

    app = build_gui()
    app.launch(
        server_port=config.GUI_PORT,
        share=config.GUI_SHARE,
        inbrowser=True,
    )
