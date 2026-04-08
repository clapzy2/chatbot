"""
knowledge_base.py — загрузка файлов, разбивка на разделы, индексация, поиск.
Основной модуль RAG-пайплайна.
"""
import os
import re
import gc
import glob
import hashlib
import sys
import zipfile
import tempfile
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Функции загрузки файлов разных форматов

def _load_txt(path):
    """Прочитать текстовый файл (пробуем разные кодировки)."""
    for enc in ["utf-8", "cp1251", "latin-1", "cp866"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Не удалось прочитать: {path}")

def _load_pdf(path):
    """Извлечь текст из PDF."""
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())

def _load_docx(path):
    """Извлечь текст из Word-документа."""
    from docx import Document
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

def _load_epub(path):
    """Извлечь текст из EPUB."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(path)
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)

def _load_fb2(path):
    """Извлечь текст из FB2."""
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")
    body = soup.find("body")
    return body.get_text(separator="\n") if body else soup.get_text(separator="\n")

def _load_fb2zip(path):
    """Распаковать ZIP и извлечь текст из FB2 внутри."""
    with zipfile.ZipFile(path, "r") as zf:
        fb2_files = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
        if not fb2_files:
            raise ValueError(f"Нет .fb2 внутри архива: {path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract(fb2_files[0], tmpdir)
            return _load_fb2(os.path.join(tmpdir, fb2_files[0]))

def _load_html(path):
    """Извлечь текст из HTML."""
    from bs4 import BeautifulSoup
    text = _load_txt(path)
    return BeautifulSoup(text, "html.parser").get_text(separator="\n")

# Словарь: расширение файла → функция загрузки
_LOADERS = {
    ".pdf": _load_pdf, ".txt": _load_txt, ".md": _load_txt,
    ".docx": _load_docx, ".epub": _load_epub, ".fb2": _load_fb2,
    ".fb2.zip": _load_fb2zip, ".html": _load_html, ".htm": _load_html,
}

def load_file(path):
    """Определить формат файла и извлечь текст."""
    lower = path.lower()
    if lower.endswith(".fb2.zip"):
        return _load_fb2zip(path)
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
    return loader(path)


# Определение разделов в тексте

def _detect_sections(text):
    """
    Разбивает текст на разделы по заголовкам.
    Заголовок определяется по паттернам: отступ, заглавные буквы,
    окружение пустыми строками, длина < 120 символов.
    Возвращает список пар: (название_раздела, текст_раздела)
    """
    lines = text.split("\n")
    header_patterns = [
        r'^\s{2,}[А-ЯЁA-Z][А-Яа-яёЁA-Za-z\s:–—\-,.]+\s*$',
        r'^\s*(Глава|Часть|Раздел|Chapter|Part|Section)\s+[\dIVXLCDMivxlcdm]+.*$',
        r'^\s*[А-ЯЁA-Z\s:–—\-]{10,}\s*$',
    ]
    sections = []
    current_header = ""
    current_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        # Проверяем, является ли строка заголовком
        is_header = False
        for pattern in header_patterns:
            if re.match(pattern, line) and len(stripped) < 120:
                prev_empty = (i == 0) or (i > 0 and not lines[i-1].strip())
                next_empty = (i == len(lines)-1) or (i < len(lines)-1 and not lines[i+1].strip())
                if prev_empty and next_empty:
                    is_header = True
                    break

        if is_header:
            # Сохраняем предыдущую секцию и начинаем новую
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_header, body))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Сохраняем последнюю секцию
    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_header, body))

    # Если заголовков не нашли — весь текст как одна секция
    if len(sections) <= 1:
        return [("", text)]
    return sections


# Основной класс — База Знаний

class KnowledgeBase:
    """Управляет загрузкой, индексацией и поиском по текстам."""

    def __init__(self, progress_callback=None, llm_engine=None):
        self._log = progress_callback or print
        self._reranker = None
        self._reranker_loaded = False
        self._llm = llm_engine
        self._init_embeddings()
        self._init_db()

    def _init_embeddings(self):
        """Загружаем модель BGE-M3 для превращения текста в вектор."""
        self._log("🔄 Загружаем embedding-модель...")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._log(f"✅ Эмбеддинги загружены: {config.EMBEDDING_MODEL}")

    def _init_db(self):
        """Подключаемся к ChromaDB (создаём коллекцию, если нет)."""
        import chromadb
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        try:
            self._col = self._client.get_collection(config.COLLECTION_NAME)
            self._log(f"📂 Коллекция: {self._col.count()} фрагментов")
        except Exception:
            self._col = self._client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._log("🆕 Новая коллекция создана")

    def _ensure_reranker(self):
        """Загружаем реранкер при первом использовании (ленивая загрузка)."""
        if self._reranker_loaded:
            return
        self._reranker_loaded = True
        if not config.USE_RERANKER:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANKER_MODEL)
            self._log(f"✅ Reranker: {config.RERANKER_MODEL}")
        except Exception as e:
            self._log(f"⚠️ Reranker недоступен: {e}")

    def set_llm(self, llm_engine):
        self._llm = llm_engine

    def _get_llm(self):
        if self._llm is None:
            from src.llm_engine import LLMEngine
            self._llm = LLMEngine()
        return self._llm

    @staticmethod
    def _md5(text):
        """Хеш текста для проверки дубликатов."""
        return hashlib.md5(text.strip().encode()).hexdigest()

    def _get_splitter(self):
        """Создаёт объект для разбивки текста на чанки."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n\n", "\n\n", "\n", ". ", "; ", " ", ""],
        )

    # Индексация файла
    def add_book(self, file_path):
        """Загрузить файл, разбить на чанки, добавить в ChromaDB."""
        filename = os.path.basename(file_path)
        lower = file_path.lower()
        supported = any(lower.endswith(fmt) for fmt in config.SUPPORTED_FORMATS)
        if not supported:
            return f"Формат не поддерживается: {filename}"

        self._log(f"Обрабатываем: {filename}")
        try:
            raw = load_file(file_path)
        except Exception as e:
            return f"Ошибка чтения {filename}: {e}"
        if not raw.strip():
            return f"Файл пуст: {filename}"

        # Определяем разделы ДО чистки текста
        sections = _detect_sections(raw)
        has_sections = len(sections) > 1
        if has_sections:
            names = [s[0] for s in sections if s[0]]
            self._log(f"Найдено {len(sections)} разделов: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")

        # Чистим текст от лишних пробелов и переносов
        cleaned = []
        for name, body in sections:
            body = re.sub(r'\n{3,}', '\n\n', body)
            body = re.sub(r' {2,}', ' ', body)
            cleaned.append((name, body))
        sections = cleaned

        splitter = self._get_splitter()
        existing_ids = set(self._col.get()["ids"]) if self._col.count() > 0 else set()
        new_chunks, new_ids, new_metas, seen = [], [], [], set()

        for section_name, section_text in sections:
            if not section_text.strip():
                continue
            chunks = splitter.split_text(section_text)
            for chunk in chunks:
                # Добавляем название раздела в начало чанка
                chunk_with_ctx = f"[{section_name}]\n{chunk}" if section_name else chunk
                h = self._md5(chunk_with_ctx)
                if h not in existing_ids and h not in seen:
                    seen.add(h)
                    new_chunks.append(chunk_with_ctx)
                    new_ids.append(h)
                    new_metas.append({
                        "source_file": filename,
                        "source": file_path,
                        "section": section_name or "",
                        "chunk_id": len(new_chunks) - 1,
                    })

        if not new_chunks:
            return f"⏭️ {filename} — уже в базе"

        # Добавляем в ChromaDB пачками по 32
        batch_size = 32
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            b_ids = new_ids[i:i+batch_size]
            b_metas = new_metas[i:i+batch_size]
            embeddings = self._embeddings.embed_documents(batch)
            self._col.add(ids=b_ids, embeddings=embeddings, documents=batch, metadatas=b_metas)
            pct = min(100, int((i + len(batch)) / len(new_chunks) * 100))
            self._log(f"  {filename}: {pct}%")

        section_info = f" ({len(sections)} разделов)" if has_sections else ""
        return f"✅ {filename}: добавлено {len(new_chunks)} фрагментов{section_info}"

    def index_all_books(self):
        """Проиндексировать все файлы из папки docs/."""
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        files = []
        for ext in config.SUPPORTED_FORMATS:
            files.extend(glob.glob(os.path.join(config.DOCS_DIR, f"**/*{ext}"), recursive=True))
        files = sorted(set(files))
        if not files:
            return f"❌ Нет файлов в docs/\nПоддерживаемые форматы: {', '.join(config.SUPPORTED_FORMATS)}"
        results = [f"📚 Найдено файлов: {len(files)}"]
        for fp in files:
            results.append(self.add_book(fp))
        gc.collect()
        results.append(f"\n📊 Итого в базе: {self._col.count()} фрагментов")
        return "\n".join(results)

    def clear(self):
        """Удалить всю коллекцию и создать новую."""
        self._client.delete_collection(config.COLLECTION_NAME)
        self._col = self._client.create_collection(name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        gc.collect()
        return "✅ База очищена"

    def stats(self):
        """Статистика: количество файлов, чанков, разделов."""
        if self._col.count() == 0:
            return {"total_chunks": 0, "total_books": 0, "books": [], "sections": []}
        data = self._col.get(include=["metadatas"])
        books, sections = set(), set()
        for m in data["metadatas"]:
            if m:
                books.add(m.get("source_file", "?"))
                s = m.get("section", "")
                if s:
                    sections.add(s)
        return {"total_chunks": self._col.count(), "total_books": len(books),
                "books": sorted(books), "sections": sorted(sections)}

    # HyDE: расширение запроса
    def _expand_query(self, query):
        """Генерируем 3 переформулировки вопроса через LLM."""
        if not config.USE_HYDE:
            return [query]
        try:
            prompt = config.PROMPTS["hyde"].format(n=config.HYDE_VARIANTS, query=query)
            result = self._get_llm().call(prompt, temperature=0.4, max_tokens=150)
            variants = [l.strip() for l in result.split("\n") if l.strip()]
            return [query] + variants[:config.HYDE_VARIANTS]
        except Exception:
            return [query]

    # Поиск в ChromaDB
    def _build_where_filter(self, file_filter="all", section_filter=None):
        """Строим фильтр для ChromaDB (по файлу и/или разделу)."""
        conditions = []
        if file_filter and file_filter != "all":
            conditions.append({"source_file": file_filter})
        if section_filter:
            conditions.append({"section": section_filter})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _raw_search(self, queries, kw_filter=None):
        """Для каждого варианта запроса ищем top-20 в ChromaDB."""
        seen, results = set(), []
        n_total = self._col.count()
        if n_total == 0:
            return []
        for q in queries:
            q_embed = self._embeddings.embed_query(q)
            kwargs = dict(
                query_embeddings=[q_embed],
                n_results=min(config.RETRIEVAL_TOP_K, n_total),
                include=["documents", "metadatas", "distances"]
            )
            if kw_filter:
                kwargs["where"] = kw_filter
            try:
                r = self._col.query(**kwargs)
            except Exception:
                continue
            for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
                h = self._md5(doc)
                relevance = max(0.0, 1.0 - dist)  # расстояние → сходство
                if h not in seen and relevance >= config.MIN_RELEVANCE:
                    seen.add(h)
                    results.append((doc, meta, relevance))
        return results

    # Реранкинг
    def _rerank_candidates(self, query, candidates):
        """Cross-encoder пересчитывает сходство, оставляет top-7."""
        self._ensure_reranker()
        docs = [c[0] for c in candidates]
        if self._reranker and len(docs) > 1:
            pairs = [[query, d] for d in docs]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:config.RERANK_TOP_K]]
        # Если реранкер не загружен — фильтруем по ключевым словам
        words = set(re.findall(r"[А-Яа-яёЁA-Za-z]{3,}", query.lower()))
        def kw(text):
            t = text.lower()
            return sum(1 for w in words if w in t) / max(len(words), 1)
        return sorted(candidates, key=lambda c: kw(c[0]), reverse=True)[:config.RERANK_TOP_K]

    # Сборка контекста для LLM
    def _build_context(self, candidates):
        """Склеиваем фрагменты в текст для отправки в LLM."""
        parts, total = [], 0
        for i, (doc, meta, score) in enumerate(candidates):
            fname = meta.get("source_file", "?")
            section = meta.get("section", "")
            label = f"{fname} | {section}" if section else fname
            block = f"[Фрагмент {i+1} | {label} | score: {score:.2f}]\n{doc.strip()}"
            if total + len(block) > config.MAX_CTX_CHARS:
                left = config.MAX_CTX_CHARS - total
                if left < 200:
                    break
                cut = block[:left].rfind(". ")
                block = block[:cut+1] if cut > 50 else block[:left]
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    # Распознавание раздела в запросе пользователя
    def get_available_sections(self):
        """Список всех разделов в базе."""
        if self._col.count() == 0:
            return []
        data = self._col.get(include=["metadatas"])
        return sorted({m.get("section", "") for m in data["metadatas"] if m and m.get("section")})

    def find_section_in_query(self, query):
        """
        Ищет в запросе название раздела (с учётом падежей русского языка).
        Например: "из Речи Федра" → находит раздел "Речь Федра: ..."
        """
        sections = self.get_available_sections()
        if not sections:
            return None
        query_lower = query.lower()

        # 1. Точное вхождение названия раздела в запрос
        best, best_len = None, 0
        for section in sections:
            if section.lower() in query_lower and len(section) > best_len:
                best, best_len = section, len(section)
                continue
            key_part = section.split(":")[0].strip().lower()
            if len(key_part) >= 5 and key_part in query_lower and len(key_part) > best_len:
                best, best_len = section, len(key_part)
        if best:
            return best

        # 2. Поиск по ключевым словам с учётом падежей (стемминг)
        skip_words = {
            "речь", "речи", "речей", "глава", "главы", "часть", "части",
            "раздел", "сцена", "эрот", "эрота", "эроте", "эроту", "эротом",
            "любовь", "любви", "стремление", "происхождение", "совершенства",
            "древнейшее", "целостности", "овладение", "благом", "панегирик",
            "природе", "разлит", "заключительная",
        }
        query_words = set(re.findall(r"[А-Яа-яёЁ]{3,}", query_lower))
        for section in sections:
            names = [w for w in re.findall(r"[А-Яа-яёЁ]{4,}", section.lower()) if w not in skip_words]
            for name in names:
                stem = name[:max(4, len(name)-2)]
                for qw in query_words:
                    if qw.startswith(stem) or name.startswith(qw[:max(4, len(qw)-2)]):
                        return section
        return None

    # Основной метод поиска (вызывается из чата)
    def search(self, query, file_filter="all", section_filter=None):
        """
        Полный RAG-пайплайн:
        1. HyDE — переформулировка запроса
        2. Поиск в ChromaDB — top-20
        3. Реранкинг — top-7
        4. Сборка контекста для LLM
        """
        if self._col.count() == 0:
            return "База пуста. Добавьте файлы и нажмите «Индексировать»."

        kw_filter = self._build_where_filter(file_filter, section_filter)
        queries = self._expand_query(query)
        cands = self._raw_search(queries, kw_filter)

        if not cands:
            # Fallback: ищем без фильтров
            q_e = self._embeddings.embed_query(query)
            fallback = dict(
                query_embeddings=[q_e],
                n_results=min(5, self._col.count()),
                include=["documents", "metadatas", "distances"]
            )
            if kw_filter:
                fallback["where"] = kw_filter
            r = self._col.query(**fallback)
            cands = [(d, m, max(0.0, 1.0 - dist))
                     for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]

        if not cands:
            return ""
        return self._build_context(self._rerank_candidates(query, cands))

    def get_available_files(self):
        """Список файлов в базе (для выпадающего списка)."""
        if self._col.count() == 0:
            return []
        data = self._col.get(include=["metadatas"])
        return sorted({m.get("source_file", "") for m in data["metadatas"] if m and m.get("source_file")})
