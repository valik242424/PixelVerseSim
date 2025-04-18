import random
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLabel, QSlider, QPushButton  # <--- Додано QPushButton
)
# Додаємо QTimer і можливість роботи зі слотами/сигналами напряму
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor

# Імпортуємо константи та наш віджет сітки
from config import GRID_WIDTH, GRID_HEIGHT, VIEW_SIZE
from simulation_grid_widget import SimulationGridWidget
from entities import Entity, Wall, Bot

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симуляція сітки 100x100 з сутностями")
        self.setGeometry(100, 100, 900, 500)

        # 1. Створюємо дані для сітки та сутності (як і раніше)
        self.grid_data = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.entities_list = []

        # Додамо стіни
        for _ in range(50):
            r, c = random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1)
            if self.grid_data[r][c] is None:
                wall = Wall()
                self.grid_data[r][c] = wall

        # Додамо ботів
        self.bot_positions = {} # Словник для зберігання позицій ботів: {bot_instance: (r, c)}
        for i in range(10):
            while True: # Шукаємо вільне місце
                r, c = random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1)
                if self.grid_data[r][c] is None:
                    bot_color = random.choice([QColor(Qt.GlobalColor.red), QColor(Qt.GlobalColor.cyan), QColor(Qt.GlobalColor.magenta)])
                    bot = Bot(bot_id=f"B{i+1}", color=bot_color, energy=random.randint(50, 150))
                    self.grid_data[r][c] = bot
                    self.entities_list.append(bot)
                    self.bot_positions[bot] = (r, c) # Зберігаємо початкову позицію
                    break # Знайшли місце, виходимо з while

        # 2. Створюємо центральний віджет та головний лейаут (без змін)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 3. Ліва панель (Логи) - без змін
        log_area_container = QWidget()
        log_layout = QVBoxLayout(log_area_container)
        log_label = QLabel("Логи симуляції:")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_text_edit)
        log_area_container.setMinimumWidth(200)
        log_area_container.setMaximumWidth(300)
        main_layout.addWidget(log_area_container, 1)

        # 4. Центральна панель (Сітка симуляції) - без змін
        self.grid_widget = SimulationGridWidget(self.grid_data)
        main_layout.addWidget(self.grid_widget, 3)

        # 5. Права панель (Елементи керування) - ДОДАЄМО КНОПКИ
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_label = QLabel("Панель керування:")
        controls_layout.addWidget(controls_label)

        # --- Кнопки керування симуляцією ---
        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.step_button = QPushButton("Крок") # Додамо кнопку для одного кроку
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.step_button)
        controls_layout.addSpacing(20)
        # --- Кінець кнопок ---

        initial_row = self.grid_widget.view_row_offset
        initial_col = self.grid_widget.view_col_offset
        self.coordinates_label = QLabel(f"Координати вікна: ({initial_row}, {initial_col})")
        controls_layout.addWidget(self.coordinates_label)
        controls_layout.addSpacing(20)

        slider1_label = QLabel("Параметр 1:")
        slider1 = QSlider(Qt.Orientation.Horizontal)
        slider2_label = QLabel("Параметр 2:")
        slider2 = QSlider(Qt.Orientation.Horizontal)
        controls_layout.addWidget(slider1_label)
        controls_layout.addWidget(slider1)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(slider2_label)
        controls_layout.addWidget(slider2)
        controls_layout.addStretch()
        controls_container.setMinimumWidth(150)
        controls_container.setMaximumWidth(250)
        main_layout.addWidget(controls_container, 1)

        # --- Налаштування таймера симуляції ---
        self.timer = QTimer(self)
        self.timer.setInterval(200) # Інтервал в мілісекундах (наприклад, 200 мс = 5 кроків/сек)
        self.timer.timeout.connect(self.simulation_step)
        self.simulation_running = False # Прапорець, чи запущена симуляція

        # --- Підключення сигналів ---
        self.grid_widget.coordinatesChanged.connect(self.update_coordinates_label)
        self.grid_widget.cellClicked.connect(self.log_cell_click)
        # Підключаємо кнопки
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.step_button.clicked.connect(self.simulation_step) # Кнопка "Крок" теж викликає simulation_step

        # --- Початкові логи ---
        self.add_log("Симуляція запущена (з сутностями).")
        self.add_log(f"Розмір сітки: {GRID_WIDTH}x{GRID_HEIGHT}")
        self.add_log(f"Розмір видимого вікна: {VIEW_SIZE}x{VIEW_SIZE}")
        self.add_log("Використовуйте WASD для навігації.")
        self.add_log("Клікніть мишкою по клітинці для інформації про сутність.")
        self.add_log(f"Створено {len(self.entities_list)} ботів.")
        self.add_log("Натисніть 'Старт' для початку симуляції.") # Додано інструкцію

        self.grid_widget.setFocus()

    @Slot(str)
    def add_log(self, message):
        self.log_text_edit.append(message)

    @Slot(int, int, str)
    def log_cell_click(self, row, col, state_info):
        self.add_log(f"Клікнуто: [{row}, {col}], {state_info}")

    @Slot(int, int)
    def update_coordinates_label(self, row, col):
        self.coordinates_label.setText(f"Координати вікна: ({row}, {col})")

    # --- Методи керування симуляцією ---
    @Slot()
    def start_simulation(self):
        if not self.simulation_running:
            self.timer.start()
            self.simulation_running = True
            self.add_log("Симуляцію запущено.")
            self.start_button.setEnabled(False) # Вимикаємо кнопку "Старт"
            self.stop_button.setEnabled(True)   # Вмикаємо кнопку "Стоп"
            self.step_button.setEnabled(False)  # Вимикаємо кнопку "Крок"

    @Slot()
    def stop_simulation(self):
        if self.simulation_running:
            self.timer.stop()
            self.simulation_running = False
            self.add_log("Симуляцію зупинено.")
            self.start_button.setEnabled(True)  # Вмикаємо кнопку "Старт"
            self.stop_button.setEnabled(False) # Вимикаємо кнопку "Стоп"
            self.step_button.setEnabled(True)  # Вмикаємо кнопку "Крок"

    @Slot()
    def simulation_step(self):
        """Виконує один крок симуляції."""
        # self.add_log("Simulation step...") # Можна розкоментувати для відладки

        # Список змін, які потрібно застосувати до сітки: [(r, c, new_entity), ...]
        pending_changes = []
        # Зберігаємо нові позиції ботів цього кроку, щоб уникнути конфліктів в межах одного кроку
        next_bot_positions = {}

        # Ітеруємо по копії списку, бо боти можуть видалятися (в майбутньому)
        for bot in list(self.entities_list):
            if bot not in self.bot_positions: continue # Пропускаємо, якщо бота вже немає (на майбутнє)

            current_pos = self.bot_positions[bot]
            r, c = current_pos

            # Викликаємо update бота, передаючи сітку і його поточну позицію
            # Bot.update тепер поверне нову позицію (r, c) або None, якщо не рухався
            move_action = bot.update(self.grid_data, r, c)

            if move_action:
                # move_action - це (new_r, new_c)
                new_r, new_c = move_action

                # Перевірка, чи нова позиція не зайнята іншим ботом *на цьому ж кроці*
                target_occupied = False
                for pos in next_bot_positions.values():
                    if pos == (new_r, new_c):
                        target_occupied = True
                        break

                if not target_occupied:
                     # Якщо вільно, плануємо зміну
                    pending_changes.append((new_r, new_c, bot)) # Поставити бота на нове місце
                    pending_changes.append((r, c, None))       # Зробити старе місце порожнім
                    next_bot_positions[bot] = (new_r, new_c) # Записуємо нову позицію для перевірок іншими ботами
                else:
                    # Якщо клітинка вже заброньована іншим ботом на цьому кроці, бот залишається на місці
                    next_bot_positions[bot] = (r, c) # Залишається на старій позиції
            else:
                 # Якщо бот не рухався, його позиція не змінюється
                 next_bot_positions[bot] = (r, c)

        # Застосовуємо всі заплановані зміни до реальної сітки
        if pending_changes:
            for r_change, c_change, new_entity in pending_changes:
                 # Перевірка меж на всяк випадок (хоча update має це робити)
                 if 0 <= r_change < GRID_HEIGHT and 0 <= c_change < GRID_WIDTH:
                     self.grid_data[r_change][c_change] = new_entity

            # Оновлюємо словник позицій ботів
            self.bot_positions = next_bot_positions

            # Оновлюємо віджет сітки, щоб відобразити зміни
            self.grid_widget.update()

        # Якщо симуляція не запущена (тобто це був клік "Крок"),
        # переконуємось, що кнопки у правильному стані
        if not self.simulation_running:
             self.start_button.setEnabled(True)
             self.stop_button.setEnabled(False)
             self.step_button.setEnabled(True)