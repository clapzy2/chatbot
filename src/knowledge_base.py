"""
KnowledgeBase — объединение обоих проектов + fb2.zip поддержка + Qwen3

Из Lama Loca:
  ✓ Cross-Encoder reranker (ms-marco-MiniLM-L-12-v2)
  ✓ Прямой chromadb.PersistentClient
  ✓ Поддержка PDF/EPUB/DOCX/TXT/FB2/FB2.ZIP/HTML/MD
  ✓ Пакетная индексация (batch_size=32)
  ✓ Префиксы passage:/query: для E5

Из твоего проекта:
  ✓ HyDE — переформулировки запроса
  ✓ Двухпроходная верификация
  ✓ Два режима чанков (precise / analytical)
  ✓ Дедупликация по MD5
  ✓ Keyword boost
"""
import os
import re
import glob
import hashlib
import sys
import zipfile
import tempfile
from typing import List, Tuple

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
    """Распаковывает fb2.zip во временную папку и читает fb2"""
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
    ".pdf":     _load_pdf,
    ".txt":     _load_txt,
    ".md":      _load_txt,
    ".docx":    _load_docx,
    ".epub":    _load_epub,
    ".fb2":     _load_fb2,
    ".fb2.zip": _load_fb2zip,   # ← новый формат
    ".html":    _load_html,
    ".htm":     _load_html,
}


def load_file(path: str) -> str:
    # Проверяем составное расширение .fb2.zip первым
    lower = path.lower()
    if lower.endswith(".fb2.zip"):
        return _load_fb2zip(path)
    ext = os.path.splitext(path)[1].lower()
    loader = _LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
    return loader(path)


# ══════════════════════════════════════════════════════════════
#  БАЗА ЗНАНИЙ
# ══════════════════════════════════════════════════════════════

class KnowledgeBase:

    def __init__(self, progress_callback=None):
        self._log      = progress_callback or print
        self._reranker = None
        self._llm      = None

        self._init_embeddings()
        self._init_db()
        self._init_reranker()

    # ── Инициализация ─────────────────────────────────────────
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
        self._log("✅ Эмбеддинги загружены")

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

    def _init_reranker(self):
        if not config.USE_RERANKER:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANKER_MODEL)
            self._log(f"✅ Reranker: {config.RERANKER_MODEL}")
        except Exception as e:
            self._log(f"⚠️ Reranker недоступен: {e}")

    def _get_llm(self):
        if self._llm is None:
            from src.llm_engine import LLMEngine
            self._llm = LLMEngine()
        return self._llm

    # ── Утилиты ───────────────────────────────────────────────
    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def _get_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        cfg = config.CHUNK_SETTINGS[config.CHUNK_MODE]
        return RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n\n", "\n\n", "\n", ". ", "; ", " ", ""],
        )

    # ── Индексация ────────────────────────────────────────────
    def add_book(self, file_path: str) -> str:
        filename = os.path.basename(file_path)
        lower    = file_path.lower()

        # Проверяем поддерживаемый формат
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

        # Чистка текста
        raw = re.sub(r'\n{3,}', '\n\n', raw)
        raw = re.sub(r' {2,}', ' ', raw)

        splitter = self._get_splitter()
        chunks   = splitter.split_text(raw)

        # Дедупликация по MD5
        existing_ids = set(self._col.get()["ids"]) if self._col.count() > 0 else set()
        new_chunks, new_ids, seen = [], [], set()
        for chunk in chunks:
            h = self._md5(chunk)
            if h not in existing_ids and h not in seen:
                seen.add(h)
                new_chunks.append(chunk)
                new_ids.append(h)

        if not new_chunks:
            return f"⏭️ {filename} — уже в базе"

        # Пакетная индексация с префиксом passage: для E5
        batch_size = 32
        for i in range(0, len(new_chunks), batch_size):
            batch     = new_chunks[i:i + batch_size]
            b_ids     = new_ids[i:i + batch_size]
            prefixed  = [f"passage: {c}" for c in batch]
            embeddings = self._embeddings.embed_documents(prefixed)
            metadatas  = [
                {"source_file": filename, "source": file_path, "chunk_id": i + j}
                for j in range(len(batch))
            ]
            self._col.add(
                ids=b_ids,
                embeddings=embeddings,
                documents=batch,       # оригинал без префикса
                metadatas=metadatas,
            )
            pct = min(100, int((i + len(batch)) / len(new_chunks) * 100))
            self._log(f"  {filename}: {pct}%")

        return f"✅ {filename}: добавлено {len(new_chunks)} фрагментов"

    def index_all_books(self) -> str:
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        files = []
        for ext in config.SUPPORTED_FORMATS:
            pattern = f"**/*{ext}" if not ext.startswith(".fb2.") else f"**/*{ext}"
            files.extend(glob.glob(
                os.path.join(config.DOCS_DIR, pattern), recursive=True
            ))
        # Убираем дубли (glob может дважды найти .fb2 если шаблоны пересекутся)
        files = sorted(set(files))

        if not files:
            exts = ", ".join(config.SUPPORTED_FORMATS)
            return f"❌ Нет файлов в docs/\nПоддерживаемые форматы: {exts}"

        results = [f"📚 Найдено файлов: {len(files)}"]
        for fp in files:
            results.append(self.add_book(fp))
        results.append(f"\n📊 Итого в базе: {self._col.count()} фрагментов")
        return "\n".join(results)

    def clear(self) -> str:
        self._client.delete_collection(config.COLLECTION_NAME)
        self._col = self._client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return "✅ База очищена"

    def stats(self) -> dict:
        if self._col.count() == 0:
            return {"total_chunks": 0, "total_books": 0, "books": []}
        data  = self._col.get(include=["metadatas"])
        books = {m.get("source_file", "?") for m in data["metadatas"] if m}
        return {
            "total_chunks": self._col.count(),
            "total_books":  len(books),
            "books":        sorted(books),
        }

    # ── HyDE ──────────────────────────────────────────────────
    def _expand_query(self, query: str) -> List[str]:
        if not config.USE_HYDE:
            return [query]
        try:
            prompt   = config.PROMPTS["hyde"].format(n=config.HYDE_VARIANTS, query=query)
            result   = self._get_llm().call(prompt, temperature=0.4, max_tokens=150)
            variants = [l.strip() for l in result.split("\n") if l.strip()]
            return [query] + variants[:config.HYDE_VARIANTS]
        except Exception:
            return [query]

    # ── Поиск ─────────────────────────────────────────────────
    def _raw_search(self, queries: List[str], kw_filter: dict = None) -> List[Tuple]:
        seen, results = set(), []
        for q in queries:
            q_embed = self._embeddings.embed_query(f"query: {q}")
            kwargs  = dict(
                query_embeddings=[q_embed],
                n_results=min(config.RETRIEVAL_TOP_K, max(self._col.count(), 1)),
                include=["documents", "metadatas", "distances"],
            )
            if kw_filter:
                kwargs["where"] = kw_filter
            try:
                r = self._col.query(**kwargs)
            except Exception:
                continue
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            ):
                h         = self._md5(doc)
                relevance = max(0.0, 1.0 - dist)
                if h not in seen and relevance >= config.MIN_RELEVANCE:
                    seen.add(h)
                    results.append((doc, meta, relevance))
        return results

    def _rerank_candidates(self, query: str, candidates: List[Tuple]) -> List[Tuple]:
        docs = [c[0] for c in candidates]
        if self._reranker and len(docs) > 1:
            pairs  = [[query, d] for d in docs]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:config.RERANK_TOP_K]]
        # Keyword boost fallback
        words = set(re.findall(r"[А-Яа-яёЁA-Za-z]{3,}", query.lower()))
        def kw(text):
            t = text.lower()
            return sum(1 for w in words if w in t) / max(len(words), 1)
        ranked = sorted(candidates, key=lambda c: kw(c[0]), reverse=True)
        return ranked[:config.RERANK_TOP_K]

    def _build_context(self, candidates: List[Tuple]) -> str:
        parts, total = [], 0
        for i, (doc, meta, score) in enumerate(candidates):
            fname = meta.get("source_file", "?")
            block = f"[Фрагмент {i+1} | {fname} | score: {score:.2f}]\n{doc.strip()}"
            if total + len(block) > config.MAX_CTX_CHARS:
                left = config.MAX_CTX_CHARS - total
                if left < 200:
                    break
                cut   = block[:left].rfind(". ")
                block = block[:cut + 1] if cut > 50 else block[:left]
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def search(self, query: str, file_filter: str = "all") -> str:
        if self._col.count() == 0:
            return "База пуста. Добавьте файлы в docs/ и нажмите «Индексировать»."
        kw_filter = {"source_file": file_filter} if file_filter != "all" else None
        queries   = self._expand_query(query)
        cands     = self._raw_search(queries, kw_filter)
        if not cands:
            # fallback без порога
            q_e = self._embeddings.embed_query(f"query: {query}")
            r   = self._col.query(
                query_embeddings=[q_e],
                n_results=min(5, self._col.count()),
                include=["documents", "metadatas", "distances"],
            )
            cands = [(d, m, max(0.0, 1.0 - dist))
                     for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        if not cands:
            return ""
        top = self._rerank_candidates(query, cands)
        return self._build_context(top)

    def search_with_meta(self, query: str, file_filter: str = "all"):
        """Возвращает (context, docs_list, scores) для Flask-сайдбара"""
        if self._col.count() == 0:
            return "", [], []
        kw_filter = {"source_file": file_filter} if file_filter != "all" else None
        queries   = self._expand_query(query)
        cands     = self._raw_search(queries, kw_filter)
        if not cands:
            q_e = self._embeddings.embed_query(f"query: {query}")
            r   = self._col.query(
                query_embeddings=[q_e],
                n_results=min(5, self._col.count()),
                include=["documents", "metadatas", "distances"],
            )
            cands = [(d, m, max(0.0, 1.0 - dist))
                     for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        if not cands:
            return "", [], []
        top    = self._rerank_candidates(query, cands)
        ctx    = self._build_context(top)
        docs   = [{"text": c[0], "file": c[1].get("source_file", "?"),
                   "score": round(c[2], 3)} for c in top]
        scores = [c[2] for c in top]
        return ctx, docs, scores

    def get_available_files(self) -> List[str]:
        if self._col.count() == 0:
            return []
        data  = self._col.get(include=["metadatas"])
        files = {m.get("source_file", "") for m in data["metadatas"] if m}
        return sorted(f for f in files if f)
