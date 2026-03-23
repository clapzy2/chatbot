# Философский чат-бот (RAG + Fine-tuning)

Чат-бот, который отвечает на вопросы по загруженным текстам, используя локальные модели. Проект демонстрирует два подхода: **RAG (Retrieval-Augmented Generation)** и **Fine-tuning**.

## Возможности

- Загрузка текстовых файлов (TXT, PDF)
- Векторный поиск по фрагментам (ChromaDB)
- Два режима работы:
  - **RAG-бот** (app_windows.py) — работает с Ollama, подходит для Windows без GPU
  - **Fine-tuned бот** (app_finetuned.py) — использует дообученную модель (требует GPU, запуск в WSL/Linux)
- Веб-интерфейс с выбором файла, историей запросов и подсветкой ключевых слов
- Полная локальная работа без интернета

## Требования

### Для RAG-бота (Windows)
- Python 3.11+
- Ollama (с моделью mistral:7b-instruct)
- 8+ ГБ RAM

### Для Fine-tuned бота (WSL/Linux)
- Python 3.11+
- NVIDIA GPU с 16+ ГБ VRAM
- Установленный WSL2 (для Windows) или нативная Linux-система
- Дообученная модель (создаётся скриптами fine-tuning)

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/clapzy2/chatbot.git
cd chatbot