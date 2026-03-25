"""
ingest.py — отдельный скрипт индексации
Альтернатива кнопке «Индексировать» в GUI.
Запуск: python ingest.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.knowledge_base import KnowledgeBase


def main():
    print("=" * 50)
    print("  TextBot — Индексация документов")
    print(f"  Чанк         : {config.CHUNK_SIZE} символов")
    print(f"  Папка        : {config.DOCS_DIR}")
    print("=" * 50)
    print()

    os.makedirs(config.DOCS_DIR, exist_ok=True)

    kb     = KnowledgeBase()
    result = kb.index_all_books()
    print(result)

    print()
    print("✅ Готово! Запусти main.py для работы с ботом.")


if __name__ == "__main__":
    main()