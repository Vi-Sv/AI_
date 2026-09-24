## Блок 2: Развертывание инфраструктуры, клиент LLM и управление контекстом памяти
В этом блоке мы подготовим операционную систему, настроим движок инференса, а также напишем два важнейших компонента ядра: клиент для связи с моделью и менеджер динамического контекста, который не позволит ИИ «забывать» нить разговора.
------------------------------
## 1. Подготовка окружения Windows и Ollama
Для того чтобы локальные нейросети работали с максимальной скоростью, скрипты должны обращаться к аппаратному ускорению (GPU). Движок Ollama берет эту задачу на себя, автоматически определяя архитектуру видеокарты (NVIDIA CUDA или AMD ROCm).
## Шаг 1. Настройка виртуального окружения Python
Откройте терминал в корневой папке вашего проекта (local-ai-agent/) и выполните команды для изоляции зависимостей:

# Создаем изолированное виртуальное окружение
python -m venv venv
# Активируем окружение в консоли Windows
call venv\Scripts\activate
# Обновляем менеджер пакетов pip до актуальной версии
python -m pip install --upgrade pip
# Устанавливаем все зависимости из нашего requirements.txt
pip install -r requirements.txt

## Шаг 2. Инициализация подсистемы ИИ (Ollama)

   1. Установите инсталлятор Ollama для Windows (Ollama-Installer.exe).
   2. По умолчанию Ollama работает как фоновая служба Windows. Убедитесь, что она запущена (в системном трее появится иконка ламы).
   3. Скачайте базовую модель. Для нашего проекта мы используем llama3.2:3b — она оптимизирована для работы на обычных ПК, потребляет около 2.5 ГБ видеопамяти и отлично справляется с инструкциями на русском языке. В консоли выполните:
   
   ollama pull llama3.2:3b
   
   4. Проверить доступность API можно, открыв браузер по адресу http://localhost:11434. Вы должны увидеть сообщение: "Ollama is running".

------------------------------
## 2. Разработка клиентского модуля API (core/llm_client.py)
Этот модуль инкапсулирует в себе все сетевые запросы к Ollama. Мы реализуем поддержку как обычной синхронной генерации, так и потоковой (Streaming), чтобы текст в интерфейсе выводился посимвольно в реальном времени, создавая эффект «живого» ответа.
Создайте файл core/llm_client.py и вставьте в него следующий код:

import jsonimport osimport requests

class OllamaClient:

    def __init__(self, config_path="config/settings.json"):
        """Инициализация клиента Ollama на основе конфигурационного файла проекта."""
        self.config_path = config_path
        self.base_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"
        self.load_config()

    def load_config(self):
        """Безопасная загрузка параметров из settings.json."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.base_url = config.get("ollama_url", self.base_url)
                    self.model_name = config.get("model_name", self.model_name)
            except Exception as e:
                print(
                    f"[ERROR] Не удалось прочитать конфиг {self.config_path}: {e}. Используются дефолты."
                )

    def is_server_online(self) -> bool:
        """Проверка доступности локального сервера Ollama."""
        try:
            response = requests.get(self.base_url, timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Синхронная генерация полного ответа (для фоновых задач и RAG)."""
        if not self.is_server_online():
            return "Ошибка: Локальный сервер ИИ (Ollama) не запущен. Пожалуйста, запустите Ollama."

        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.RequestException as e:
            return f"Ошибка при связи с ИИ-движком: {str(e)}"

    def stream_response(self, history: list, system_prompt: str = None):
        """Генератор для потоковой передачи данных (для вывода в GUI чата).

        Принимает историю в формате [{"role": "user", "content": "..."}, ...]
        """
        if not self.is_server_online():
            yield "Ошибка: Локальный сервер ИИ (Ollama) не запущен."
            return

        url = f"{self.base_url}/api/chat"

        # Формируем структуру запроса с учетом системного промпта
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        payload = {"model": self.model_name, "messages": messages, "stream": True}

        try:
            # Используем stream=True в requests для построчного чтения чанков
            response = requests.post(url, json=payload, stream=True, timeout=60)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    # Декодируем строку JSON от Ollama API
                    chunk = json.loads(line.decode("utf-8"))
                    message_chunk = chunk.get("message", {})
                    content = message_chunk.get("content", "")
                    if content:
                        yield content

                    # Проверяем флаг завершения генерации
                    if chunk.get("done", False):
                        break
        except requests.RequestException as e:
            yield f"\n[Ошибка потока данных: {str(e)}]"

------------------------------
## 3. Разработка модуля управления памятью (core/memory.py)
Обычные языковые модели не обладают встроенной памятью. Чтобы они помнили контекст беседы, им нужно при каждом новом запросе передавать историю предыдущих реплик. Однако бесконечно передавать историю нельзя: контекстное окно модели ограничено (у Llama 3.2 оно составляет 128k токенов, но для экономии ОЗУ мы жестко ограничим рабочую память фиксированным числом последних диалогов — скользящим окном).
Создайте файл core/memory.py и реализуйте в нем класс управления контекстом:

import jsonimport os

class ChatMemory:

    def __init__(self, max_turns: int = 10, history_file: str = "ai_history.txt"):
        """Класс для хранения, ротации и логирования контекста сессии диалога.

        :param max_turns: Максимальное количество запоминаемых пар "вопрос-ответ"
        """
        self.max_turns = max_turns
        self.history_file = history_file
        self.history = []  # Список словарей [{"role": "user/assistant", "content": "текст"}]

    def add_message(self, role: str, content: str):
        """Добавляет сообщение в оперативную память и обрезает контекст при превышении лимита."""
        if role not in ["user", "assistant", "system"]:
            raise ValueError(f"Недопустимая роль в истории диалога: {role}")

        self.history.append({"role": role, "content": content})

        # Ротация контекста: умножаем на 2, так как turn — это пара (user + assistant)
        if len(self.history) > (self.max_turns * 2):
            # Сохраняем системный промпт, если он был на 0-й позиции
            if self.history[0]["role"] == "system":
                self.history = [self.history[0]] + self.history[3:]
            else:
                self.history = self.history[2:]

    def get_context(self) -> list:
        """Возвращает текущую историю сообщений для отправки в модель."""
        return self.history

    def clear(self):
        """Очищает текущую оперативную память сессии."""
        self.history = []

    def save_to_disk(self, prompt: str, response: str):
        """Персистентное сохранение логов на диск для последующего анализа или аудита."""
        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(f"--- ЗАПРОС ПОЛЬЗОВАТЕЛЯ ---\n{prompt}\n")
                f.write(f"--- ОТВЕТ ИИ-АССИСТЕНТА ---\n{response}\n")
                f.write("=" * 60 + "\n\n")
        except IOError as e:
            print(f"[WARNING] Не удалось записать лог на диск: {e}")

------------------------------
Инфраструктурный слой и логика памяти готовы.
В Блоке 3 мы создадим Продвинутый модуль RAG (rag/vector_store.py и rag/document_parser.py) на базе ChromaDB. Он позволит нашему ИИ сканировать файлы из папки data/documents/, превращать их в векторы и отвечать пользователю, используя личные файлы как базу знаний.

