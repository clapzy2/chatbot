"""
KnowledgeBase v3.0

- Универсальный чанкинг (CHUNK_SIZE/CHUNK_OVERLAP из config)
- Контекстный чанкинг (заголовки разделов в чанках)
- Фильтр по разделу (find_section_in_query)
- BGE-M3 эмбеддинги, BGE-Reranker-v2-m3
- Ленивая загрузка реранкера
- Единый LLM (передаётся снаружи)
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


# ══════════════════════════════════════════════════════════════
#  ЗАГРУЗЧИКИ ФАЙЛОВ
# ══════════════════════════════════════════════════════════════

def _load_txt(path: str) -> str:
    for enc in ["utf-8", "cp1251", "latin-1", "cp866"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Не удалось прочитать: {path}")


def _load_pdf(path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    return "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def _load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_epub(path: str) -> str:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book  = epub.read_epub(path)
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _load_fb2(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml-xml")
    body = soup.find("body")
    return body.get_text(separator="\n") if body else soup.get_text(separator="\n")


def _load_fb2zip(path: str) -> str:
    with zipfile.ZipFile(path, "r") as zf:
        fb2_files = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
        if not fb2_files:
            raise ValueError(f"Нет .fb2 внутри архива: {path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract(fb2_files[0], tmpdir)
            extracted = os.path.join(tmpdir, fb2_files[0])
            return _load_fb2(extracted)


def _load_html(path: str) -> str:
    from bs4 import BeautifulSoup
    text = _load_txt(path)
    return BeautifulSoup(text, "html.parser").get_text(separator="\n")


_LOADERS = {
    ".pdf": _load_pdf, ".txt": _load_txt, ".md": _load_txt,
    ".docx": _load_docx, ".epub": _load_epub, ".fb2": _load_fb2,
    ".fb2.zip": _load_fb2zip, ".html": _load_html, ".htm": _load_html,
}


def load_file(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".fb2.zip"):
        return _load_fb2zip(path)
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
    return loader(path)


# ══════════════════════════════════════════════════════════════
#  ОПРЕДЕЛЕНИЕ РАЗДЕЛОВ В ТЕКСТЕ
# ══════════════════════════════════════════════════════════════

def _detect_sections(text: str) -> List[Tuple[str, str]]:
    """
    Разбивает текст на секции по заголовкам.
    Заголовок = строка с отступом 2+, начинается с заглавной,
    окружена пустыми строками, < 120 символов.
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
        is_header = False
        for pattern in header_patterns:
            if re.match(pattern, line) and len(stripped) < 120:
                prev_empty = (i == 0) or (i > 0 and not lines[i-1].strip())
                next_empty = (i == len(lines)-1) or (i < len(lines)-1 and not lines[i+1].strip())
                if prev_empty and next_empty:
                    is_header = True
                    break
        if is_header:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_header, body))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_header, body))

    if len(sections) <= 1:
        return [("", text)]
    return sections


# ══════════════════════════════════════════════════════════════
#  БАЗА ЗНАНИЙ
# ══════════════════════════════════════════════════════════════

class KnowledgeBase:

    def __init__(self, progress_callback=None, llm_engine=None):
        self._log       = progress_callback or print
        self._reranker   = None
        self._reranker_loaded = False
        self._llm       = llm_engine
        self._init_embeddings()
        self._init_db()

    def _init_embeddings(self):
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
    def _md5(text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def _get_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n\n", "\n\n", "\n", ". ", "; ", " ", ""],
        )

    # ── Индексация ────────────────────────────────────────────
    def add_book(self, file_path: str) -> str:
        filename = os.path.basename(file_path)
        lower    = file_path.lower()
        supported = any(lower.endswith(fmt) for fmt in config.SUPPORTED_FORMATS)
        if not supported:
            return f"⚠️ Формат не поддерживается: {filename}"

        self._log(f"📄 Обрабатываем: {filename}")
        try:
            raw = load_file(file_path)
        except Exception as e:
            return f"❌ Ошибка чтения {filename}: {e}"
        if not raw.strip():
            return f"⚠️ Файл пуст: {filename}"

        # Определяем разделы ДО чистки текста
        sections = _detect_sections(raw)
        has_sections = len(sections) > 1
        if has_sections:
            names = [s[0] for s in sections if s[0]]
            self._log(f"  📑 Найдено {len(sections)} разделов: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")

        # Чистка внутри каждой секции
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

        batch_size = 32
        for i in range(0, len(new_chunks), batch_size):
            batch   = new_chunks[i:i+batch_size]
            b_ids   = new_ids[i:i+batch_size]
            b_metas = new_metas[i:i+batch_size]
            embeddings = self._embeddings.embed_documents(batch)
            self._col.add(ids=b_ids, embeddings=embeddings, documents=batch, metadatas=b_metas)
            pct = min(100, int((i + len(batch)) / len(new_chunks) * 100))
            self._log(f"  {filename}: {pct}%")

        section_info = f" ({len(sections)} разделов)" if has_sections else ""
        return f"✅ {filename}: добавлено {len(new_chunks)} фрагментов{section_info}"

    def index_all_books(self) -> str:
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

    def clear(self) -> str:
        self._client.delete_collection(config.COLLECTION_NAME)
        self._col = self._client.create_collection(name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        gc.collect()
        return "✅ База очищена"

    def stats(self) -> dict:
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

    # ── HyDE ──────────────────────────────────────────────────
    def _expand_query(self, query: str) -> List[str]:
        if not config.USE_HYDE:
            return [query]
        try:
            prompt = config.PROMPTS["hyde"].format(n=config.HYDE_VARIANTS, query=query)
            result = self._get_llm().call(prompt, temperature=0.4, max_tokens=150)
            variants = [l.strip() for l in result.split("\n") if l.strip()]
            return [query] + variants[:config.HYDE_VARIANTS]
        except Exception:
            return [query]

    # ── Поиск ─────────────────────────────────────────────────
    def _build_where_filter(self, file_filter="all", section_filter=None):
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

    def _raw_search(self, queries: List[str], kw_filter=None) -> List[Tuple]:
        seen, results = set(), []
        n_total = self._col.count()
        if n_total == 0:
            return []
        for q in queries:
            q_embed = self._embeddings.embed_query(q)
            kwargs = dict(query_embeddings=[q_embed], n_results=min(config.RETRIEVAL_TOP_K, n_total),
                          include=["documents", "metadatas", "distances"])
            if kw_filter:
                kwargs["where"] = kw_filter
            try:
                r = self._col.query(**kwargs)
            except Exception:
                continue
            for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
                h = self._md5(doc)
                relevance = max(0.0, 1.0 - dist)
                if h not in seen and relevance >= config.MIN_RELEVANCE:
                    seen.add(h)
                    results.append((doc, meta, relevance))
        return results

    def _rerank_candidates(self, query: str, candidates: List[Tuple]) -> List[Tuple]:
        self._ensure_reranker()
        docs = [c[0] for c in candidates]
        if self._reranker and len(docs) > 1:
            pairs = [[query, d] for d in docs]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:config.RERANK_TOP_K]]
        words = set(re.findall(r"[А-Яа-яёЁA-Za-z]{3,}", query.lower()))
        def kw(text):
            t = text.lower()
            return sum(1 for w in words if w in t) / max(len(words), 1)
        return sorted(candidates, key=lambda c: kw(c[0]), reverse=True)[:config.RERANK_TOP_K]

    def _build_context(self, candidates: List[Tuple]) -> str:
        parts, total = [], 0
        for i, (doc, meta, score) in enumerate(candidates):
            fname   = meta.get("source_file", "?")
            section = meta.get("section", "")
            label   = f"{fname} | {section}" if section else fname
            block   = f"[Фрагмент {i+1} | {label} | score: {score:.2f}]\n{doc.strip()}"
            if total + len(block) > config.MAX_CTX_CHARS:
                left = config.MAX_CTX_CHARS - total
                if left < 200:
                    break
                cut = block[:left].rfind(". ")
                block = block[:cut+1] if cut > 50 else block[:left]
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    # ── Фильтр по разделу ─────────────────────────────────────
    def get_available_sections(self) -> List[str]:
        if self._col.count() == 0:
            return []
        data = self._col.get(include=["metadatas"])
        return sorted({m.get("section", "") for m in data["metadatas"] if m and m.get("section")})

    def find_section_in_query(self, query: str) -> Optional[str]:
        sections = self.get_available_sections()
        if not sections:
            return None
        query_lower = query.lower()

        # 1. Точное вхождение
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

        # 2. По именам собственным (с учётом падежей)
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

    # ── Публичные методы поиска ────────────────────────────────
    def search(self, query: str, file_filter="all", section_filter=None) -> str:
        if self._col.count() == 0:
            return "База пуста. Добавьте файлы и нажмите «Индексировать»."
        kw_filter = self._build_where_filter(file_filter, section_filter)
        queries = self._expand_query(query)
        cands = self._raw_search(queries, kw_filter)
        if not cands:
            q_e = self._embeddings.embed_query(query)
            fallback = dict(query_embeddings=[q_e], n_results=min(5, self._col.count()),
                            include=["documents", "metadatas", "distances"])
            if kw_filter:
                fallback["where"] = kw_filter
            r = self._col.query(**fallback)
            cands = [(d, m, max(0.0, 1.0-dist))
                     for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        if not cands:
            return ""
        return self._build_context(self._rerank_candidates(query, cands))

    def get_available_files(self) -> List[str]:
        if self._col.count() == 0:
            return []
        data = self._col.get(include=["metadatas"])
        return sorted({m.get("source_file", "") for m in data["metadatas"] if m and m.get("source_file")})
