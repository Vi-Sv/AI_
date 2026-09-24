## Блок 4: Асинхронный графический интерфейс (CustomTkinter)
В этом блоке мы создадим современное окно чата с поддержкой темной темы, аппаратного сглаживания и адаптивной верстки.
Главная техническая сложность при работе с ИИ в графических интерфейсах — зависание окна. Если запустить тяжелую генерацию текста в основном потоке интерфейса, окно Windows покроется белой пеленой и перестанет реагировать на клики. Чтобы этого избежать, мы реализуем многопоточность (Multithreading). Интерфейс будет работать в главном потоке, а запросы к Ollama и RAG — в фоновом, передавая текст в окно посимвольно (через потоковый генератор) в реальном времени.
------------------------------
## Разработка модуля интерфейса чата (ui/chat_window.py)
Создайте файл ui/chat_window.py и добавьте в него следующий код. Он полностью автономен, использует асинхронные очереди для связи между потоками и автоматически прокручивает чат вниз по мере генерации ответа.

import osimport queueimport threadingimport tkinter as tkimport customtkinter as ctk
# Устанавливаем базовые параметры рендеринга CustomTkinter
ctk.set_appearance_mode("dark")  # Тема по умолчанию: dark/light/system
ctk.set_default_color_theme("blue")  # Цветовая палитра: blue/green/dark-blue

class ChatWindow(ctk.CTk):

    def __init__(self, agent_core):
        """Инициализация графического интерфейса ИИ-ассистента.

        :param agent_core: Ссылка на экземпляр главного оркестратора системы
        """
        super().__init__()

        self.agent = agent_core
        self.response_queue = queue.Queue()  # Потокобезопасная очередь для передачи текста из ИИ в UI

        # Конфигурация окна
        self.title("Локальный ИИ-Ассистент")
        self.geometry("450x650")
        self.attributes("-topmost", True)  # Окно всегда поверх остальных приложений
        self.protocol("WM_DELETE_WINDOW", self.hide_window)  # Вместо закрытия — скрываем в фон

        # Настройка сетки (Grid) для адаптивности компонентов
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)  # Область вывода текста (занимает всё пространство)
        self.grid_rowconfigure(1, weight=0)  # Панель ввода информации

        # ==========================================
        # 1. ОБЛАСТЬ ВЫВОДА ДИАЛОГА (Scrollable Text)
        # ==========================================
        self.chat_display = ctk.CTkTextbox(
            self,
            font=("Segoe UI", 14),
            wrap="word",
            activate_scrollbars=True,
            corner_radius=8,
            border_spacing=10,
        )
        self.chat_display.grid(
            row=0, column=0, padx=15, pady=(15, 10), sticky="nsew"
        )
        self.chat_display.configure(
            state="disabled"
        )  # Блокируем ручное редактирование текста пользователем

        # Приветственное сообщение
        self.append_to_chat(
            "Система",
            "Локальное ядро ИИ успешно активировано.\nНажмите Ctrl+Alt+Z в любом месте экрана для вызова чата.\nЗадайте свой вопрос...",
        )

        # ==========================================
        # 2. НИЖНЯЯ ПАНЕЛЬ ВВОДА
        # ==========================================
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Поле ввода текста (поддерживает перенос строк по Shift+Enter)
        self.input_field = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Введите ваш запрос и нажмите Enter...",
            font=("Segoe UI", 13),
            height=40,
            corner_radius=8,
        )
        self.input_field.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.input_field.bind(
            "<Return>", lambda event: self.send_message_trigger()
        )

        # Кнопка отправки запроса
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Отправить",
            width=90,
            height=40,
            font=("Segoe UI", 13, "bold"),
            corner_radius=8,
            command=self.send_message_trigger,
        )
        self.send_button.grid(row=0, column=1, sticky="e")

        # Запускаем постоянный мониторинг фоновой очереди сообщений ИИ
        self.check_queue_loop()

    def show_window(self):
        """Выводит окно из фонового режима на передний план и фокусирует ввод."""
        self.deiconify()
        self.attributes("-topmost", True)
        self.focus_force()
        self.input_field.focus()

    def hide_window(self):
        """Скрывает окно, сохраняя скрипт активным в трее/памяти."""
        self.withdraw()

    def append_to_chat(self, sender: str, text: str):
        """Безопасное добавление текста в поле чата с автоматической прокруткой вниз."""
        self.chat_display.configure(state="normal")

        if sender == "Вы":
            prefix = "\n😎 Вы:\n"
            content = f"{text}\n"
        elif sender == "Система":
            prefix = "\n⚙️ Система:\n"
            content = f"{text}\n"
        else:
            prefix = f"\n🤖 {sender}:\n"
            content = f"{text}\n"

        # Если сообщение новое — печатаем префикс автора
        if sender != "CONTINUATION":
            self.chat_display.insert(tk.END, prefix)
            self.chat_display.insert(tk.END, content)
        else:
            # Если это чанк потока — просто дописываем символы
            self.chat_display.insert(tk.END, text)

        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)  # Автоматический скролл вниз

    def send_message_trigger(self):
        """Обработчик отправки сообщения пользователем."""
        user_text = self.input_field.get().strip()
        if not user_text:
            return

        # Очищаем поле ввода и блокируем интерфейс на время генерации
        self.input_field.delete(0, tk.END)
        self.set_ui_state("disabled")

        # Отображаем реплику пользователя в чате
        self.append_to_chat("Вы", user_text)

        # Выводим технический маркер для начала ответа ИИ
        self.append_to_chat("ИИ", "")

        # Запускаем тяжелый процесс генерации в ОТДЕЛЬНОМ фоновом потоке
        threading.Thread(
            target=self.async_ai_worker, args=(user_text,), daemon=True
        ).start()

    def set_ui_state(self, state: str):
        """Переключает активность кнопок и полей ввода (normal / disabled)."""
        self.send_button.configure(state=state)
        self.input_field.configure(state=state)
        if state == "normal":
            self.input_field.focus()

    def async_ai_worker(self, prompt: str):
        """Метод выполняется в фоновом потоке.

        Запрашивает данные у агента и передает чанки в UI через потокобезопасную очередь.
        """
        try:
            # Вызываем потоковый инференс агента (который внутри соберет RAG и память)
            for chunk in self.agent.stream_query(prompt):
                self.response_queue.put(("CHUNK", chunk))
        except Exception as e:
            self.response_queue.put(("ERROR", f"Критическая ошибка ядра: {str(e)}"))
        finally:
            # Передаем сигнал завершения работы
            self.response_queue.put(("DONE", None))

    def check_queue_loop(self):
        """Циклический метод проверки очереди ответов.

        Работает в главном потоке интерфейса (каждые 50 мс).
        """
        try:
            while True:
                # Извлекаем все доступные элементы из очереди без блокировки потока
                msg_type, data = self.response_queue.get_nowait()

                if msg_type == "CHUNK":
                    self.append_to_chat("CONTINUATION", data)
                elif msg_type == "ERROR":
                    self.append_to_chat("Система", data)
                elif msg_type == "DONE":
                    self.set_ui_state("normal")

                self.response_queue.task_done()
        except queue.Empty:
            pass  # Очередь пуста, продолжаем ожидание

        # Переназначаем вызов метода через 50 миллисекунд
        self.after(50, self.check_queue_loop)

------------------------------
Модуль интерфейса полностью готов, структуры данных для работы в многопоточном режиме отлажены.
Переходим к следующему этапу?
В Блоке 5 мы напишем Главный оркестратор системы (core/agent.py), который склеит все модули воедино (память, RAG, клиент Ollama, промпты), а также разработаем Службу перехвата системных событий Windows (main.py) для работы горячих клавиш по всей ОС. Напишите «готов».

