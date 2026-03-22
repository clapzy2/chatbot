import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chardet

def detect_encoding(file_path):
    """Определяет кодировку файла"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

print("🔄 Загружаем модель эмбеддингов...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Проверяем, существует ли база
if os.path.exists("./chroma_db"):
    print("📂 База данных найдена. Загружаем существующую...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    print(f"✅ Загружено {vectorstore._collection.count()} фрагментов")
else:
    print("🆕 База данных не найдена. Будет создана новая...")
    vectorstore = None

# Папка docs
if not os.path.exists("docs"):
    os.makedirs("docs")
    print("✅ Создана папка 'docs'. Положите в неё текстовые файлы и запустите скрипт снова.")
    exit()

# Загружаем файлы
documents = []
for file in os.listdir("docs"):
    file_path = os.path.join("docs", file)

    if file.endswith(".txt"):
        encoding = detect_encoding(file_path)
        loader = TextLoader(file_path, encoding=encoding)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = file
        documents.extend(docs)
        print(f"📄 Загружен TXT: {file}")

    elif file.endswith(".pdf"):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = file
                doc.metadata["source"] = file_path
            documents.extend(docs)
            print(f"📄 Загружен PDF: {file} ({len(docs)} стр.)")
        except Exception as e:
            print(f"   ❌ Ошибка PDF {file}: {e}")

if not documents:
    print("❌ Нет файлов для загрузки")
    exit()

print(f"📊 Всего загружено новых документов: {len(documents)}")

# Разбиваем на фрагменты (те же параметры, что и раньше)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"🔪 Создано новых фрагментов: {len(chunks)}")

# Добавляем в базу
if vectorstore is None:
    print("💾 Создаем новую векторную базу данных...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
else:
    print("➕ Добавляем новые фрагменты в существующую базу...")
    vectorstore.add_documents(chunks)

vectorstore.persist()
print(f"✅ Готово! Теперь в базе {vectorstore._collection.count()} фрагментов")