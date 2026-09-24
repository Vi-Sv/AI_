## Блок 3: Продвинутый модуль RAG (Векторная база данных ChromaDB)
В этом блоке мы реализуем технологию RAG (Retrieval-Augmented Generation). Мы научим наш ИИ не просто генерировать текст из своей «головы», а предварительно искать точные совпадения и факты в ваших личных локальных документах (инструкциях, коде, заметках), загруженных в папку data/documents/. В качестве хранилища векторов мы применим легковесную базу данных ChromaDB.
------------------------------
## 1. Разработка парсера документов (rag/document_parser.py)
Так как большие документы (например, PDF-книга на 300 страниц или лог-файл на 50 000 строк) нельзя передать в ИИ целиком из-за ограничений контекстного окна и падения точности, их необходимо правильно подготовить.
Этот модуль сканирует текстовые файлы, разбивает их на небольшие смысловые фрагменты (чанги) с контролируемым перекрытием (overlap), чтобы контекст на стыках не терялся.
Создайте файл rag/document_parser.py и добавьте следующий код:

import osimport re
class DocumentParser:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Модуль для парсинга и сегментации локальных документов.
        :param chunk_size: Максимальное количество символов в одном фрагменте текста.
        :param chunk_overlap: Количество символов перекрытия между соседними фрагментами.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Очистка текста от избыточных пробельных символов и мусора."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> list[str]:
        """Разбивает монолитный текст на фрагменты (чанги) с перекрытием."""
        text = self.clean_text(text)
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        pointer = 0
        while pointer < len(text):
            # Берем фрагмент заданной длины
            end_pointer = min(pointer + self.chunk_size, len(text))
            chunk = text[pointer:end_pointer]
            chunks.append(chunk)
            
            # Сдвигаем указатель вперед с учетом перекрытия
            pointer += self.chunk_size - self.chunk_overlap
            
            # Страховка от бесконечного цикла, если шаг сдвига станет нулевым или отрицательным
            if pointer >= len(text) or (self.chunk_size - self.chunk_overlap) <= 0:
                break
                
        return chunks

    def load_directory(self, target_dir: str) -> list[dict]:
        """
        Сканирует папку и извлекает тексты из поддерживаемых файлов (.txt, .md, .py).
        Возвращает список словарей с текстом чанка и метаданными источника.
        """
        processed_data = []
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"[INFO] Создана пустая папка для документов: {target_dir}")
            return processed_data

        supported_extensions = ('.txt', '.md', '.py', '.json', '.csv', '.log')
        
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                            
                        chunks = self.split_into_chunks(content)
                        for idx, chunk in enumerate(chunks):
                            processed_data.append({
                                "text": chunk,
                                "metadata": {
                                    "source": file,
                                    "path": file_path,
                                    "chunk_id": idx
                                }
                            })
                    except Exception as e:
                        print(f"[ERROR] Ошибка при чтении файла {file_path}: {e}")
                        
        print(f"[SUCCESS] Считано файлов: {len(processed_data)} текстовых фрагментов.")
        return processed_data

------------------------------
## 2. Разработка векторного хранилища (rag/vector_store.py)
Этот модуль берет нарезанные фрагменты текста, преобразует их в математические векторы (эмбеддинги) с помощью встроенной локальной модели sentence-transformers и сохраняет в базу данных ChromaDB. При запросе пользователя модуль производит векторный (семантический) поиск, вычленяя самые похожие по смыслу куски текста.
Создайте файл rag/vector_store.py и добавьте следующий код:

import osimport jsonimport chromadbfrom chromadb.utils import embedding_functionsfrom rag.document_parser import DocumentParser
class VectorStoreManager:
    def __init__(self, config_path="config/settings.json"):
        """Управление локальной векторной базой данных ChromaDB."""
        self.db_path = "data/db"
        self.docs_path = "data/documents"
        self.collection_name = "local_knowledge_base"
        self.max_results = 3
        
        # Параметры парсера по умолчанию
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        self.load_config(config_path)
        
        # Инициализация встроенной локальной функции эмбеддингов (алгоритм MiniLM запускается прямо на CPU/GPU)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Инициализация клиента ChromaDB (работает в режиме постоянного сохранения на диск)
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def load_config(self, config_path: str):
        """Загрузка параметров RAG из конфигурационного файла."""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    rag_conf = config.get("rag", {})
                    self.chunk_size = rag_conf.get("chunk_size", self.chunk_size)
                    self.chunk_overlap = rag_conf.get("chunk_overlap", self.chunk_overlap)
                    self.max_results = rag_conf.get("max_results", self.max_results)
            except Exception as e:
                print(f"[WARNING] Ошибка загрузки конфига RAG: {e}. Используются дефолтные значения.")

    def rebuild_index(self):
        """Полная очистка базы данных и переиндексация документов из папки data/documents/."""
        print("[RAG] Начало процесса полной переиндексации локальных документов...")
        
        # Удаляем старую коллекцию, если она была, и создаем заново чистую
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass # Если коллекции не существовало
            
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        
        # Парсим документы
        parser = DocumentParser(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        parsed_chunks = parser.load_directory(self.docs_path)
        
        if not parsed_chunks:
            print("[RAG] Индексация завершена: Папка документов пуста. База данных очищена.")
            return

        # Подготавливаем списки для пакетной загрузки в ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(parsed_chunks):
            documents.append(item["text"])
            metadatas.append(item["metadata"])
            ids.append(f"doc_chunk_{idx}")
            
        # Добавляем данные порциями во избежание переполнения памяти
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            
        print(f"[RAG] Успешно проиндексировано и загружено {len(documents)} текстовых фрагментов в ChromaDB.")

    def query_knowledge_base(self, user_query: str) -> str:
        """
        Ищет наиболее релевантные контексты по запросу пользователя.
        Возвращает монолитную строку контекста для внедрения в промпт модели LLM.
        """
        # Если в базе нет записей, возвращаем пустую строку
        if self.collection.count() == 0:
            return ""
            
        try:
            results = self.collection.query(
                query_texts=[user_query],
                n_results=self.max_results
            )
            
            extracted_chunks = results.get('documents', [[]])[0]
            metadata_list = results.get('metadatas', [[]])[0]
            
            if not extracted_chunks:
                return ""
                
            context_string = "\n--- НАЙДЕННЫЙ ЛОКАЛЬНЫЙ КОНТЕКСТ ---\n"
            for chunk, meta in zip(extracted_chunks, metadata_list):
                source_file = meta.get('source', 'Неизвестный источник')
                context_string += f"[Источник: {source_file}]\n{chunk}\n\n"
                
            return context_string
        except Exception as e:
            print(f"[ERROR] Ошибка при выполнении векторного поиска: {e}")
            return ""

------------------------------
## 3. Скрипт для принудительного обновления базы
Вы можете запустить этот скрипт отдельно в любой момент, когда забросите новые файлы в папку data/documents/, чтобы ИИ мгновенно «изучил» их.
Создайте проверочный файл-утилиту в корне проекта sync_rag.py:

import osimport sys
# Добавляем текущую директорию в пути поиска Python, чтобы модули импортировались корректно
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag.vector_store import VectorStoreManager
if __name__ == "__main__":
    print("=== УТИЛИТА СИНХРОНИЗАЦИИ ЗНАНИЙ ИИ ===")
    
    # Гарантируем структуру папок
    os.makedirs("data/documents", exist_ok=True)
    
    # Инициализируем менеджер и пересобираем индексы
    rag_manager = VectorStoreManager()
    rag_manager.rebuild_index()
    
    print("Синхронизация успешно завершена. Теперь ИИ видит актуальные данные.")

------------------------------
Модуль локальной базы знаний (RAG) полностью готов и структурирован. Можем переходить к следующей стадии?
В Блоке 4 мы разработаем Графический интерфейс пользователя (ui/chat_window.py) с использованием фреймворка CustomTkinter. Это будет современное асинхронное окно чата с поддержкой темного режима, плавным посимвольным выводом текста (streaming) и индикацией работы локальной модели. Напишите «готов», и мы продолжим сборку.

