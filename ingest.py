"""
ingest.py — запускается один раз, чтобы прочитать все файлы из docs/
и добавить их в базу знаний (ChromaDB).
"""
import os
import sys

# Добавляем корневую папку в пути, чтобы можно было импортировать config и src
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

    # Создаём папку docs/, если её нет
    os.makedirs(config.DOCS_DIR, exist_ok=True)

    # Создаём объект базы знаний (при этом загружаются эмбеддинги и открывается ChromaDB)
    kb     = KnowledgeBase()

    # Запускаем индексацию всех книг (файлов) из папки docs/
    result = kb.index_all_books()
    print(result)

    print()
    print("✅ Готово! Запусти main.py для работы с ботом.")


if __name__ == "__main__":
    main()