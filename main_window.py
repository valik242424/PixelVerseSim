import random
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLabel, QSlider, QPushButton
)
# Додаємо QTimer і можливість роботи зі слотами/сигналами напряму
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor

# Імпортуємо константи та наш віджет сітки
from config import GRID_WIDTH, GRID_HEIGHT, VIEW_SIZE
from simulation_grid_widget import SimulationGridWidget
# Переконаймося, що імпортуємо оновлені Bot, Wall, Food (якщо додали)
from entities import Entity, Wall, Bot , Food

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelVerseSim - Симуляція Піксельного Світу")
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
        pending_changes = []
        next_bot_positions = {} # {bot_instance: (r, c)}

        # Перемішуємо список ботів, щоб порядок оновлення був випадковим
        # Це може допомогти уникнути систематичних переваг у деяких ботів
        random.shuffle(self.entities_list)

        for bot in list(self.entities_list): # Ітеруємо по копії
            if bot not in self.bot_positions: continue # Якщо бота видалили

            current_r, current_c = self.bot_positions[bot]

            # --- Отримуємо дію від мозку бота ---
            # bot.update тепер повертає рядок дії ("move_north", "stay", etc.)
            action_name = bot.update(self.grid_data, current_r, current_c)

            if action_name is None: # На випадок неочікуваної помилки в update
                action_name = "stay"

            # --- Інтерпретуємо дію та розраховуємо нову позицію ---
            new_r, new_c = current_r, current_c # За замовчуванням залишаємось на місці
            moved = False

            if action_name == "move_north":
                new_r -= 1
                moved = True
            elif action_name == "move_east":
                new_c += 1
                moved = True
            elif action_name == "move_south":
                new_r += 1
                moved = True
            elif action_name == "move_west":
                new_c -= 1
                moved = True
            elif action_name == "stay":
                moved = False
            # else: # Невідома дія - ігноруємо
            #     print(f"Warning: Unknown action '{action_name}' from bot {bot.properties.get('id')}")
            #     moved = False


            # --- Перевірка можливості руху (на рівні симуляції) ---
            can_move = False
            if moved:
                # 1. Перевірка меж сітки
                if 0 <= new_r < GRID_HEIGHT and 0 <= new_c < GRID_WIDTH:
                    # 2. Перевірка, чи цільова клітинка зараз вільна (не стіна, не інший СТАТИЧНИЙ об'єкт)
                    #    Важливо: ми не перевіряємо тут на інших *рухомих* ботів,
                    #    оскільки це буде зроблено при перевірці конфліктів нижче.
                    target_cell = self.grid_data[new_r][new_c]
                    # Дозволяємо рух тільки в порожні клітинки (або на їжу - TODO)
                    if target_cell is None: # or isinstance(target_cell, Food):
                        can_move = True
                    # else: print(f"Bot {bot.properties.get('id')} blocked by {type(target_cell)}") # Відладка
                # else: print(f"Bot {bot.properties.get('id')} hit boundary") # Відладка

            # --- Перевірка конфліктів з іншими ботами на цьому кроці ---
            final_r, final_c = current_r, current_c # Де бот опиниться в кінці кроку

            if can_move:
                target_occupied_this_step = False
                # Перевіряємо, чи хтось вже "забронював" цільову клітинку
                for occupied_pos in next_bot_positions.values():
                    if occupied_pos == (new_r, new_c):
                        target_occupied_this_step = True
                        # print(f"Bot {bot.properties.get('id')} - target ({new_r},{new_c}) already claimed this step.") # Відладка
                        break

                if not target_occupied_this_step:
                    # Рух дозволено і немає конфлікту
                    final_r, final_c = new_r, new_c
                    # Плануємо зміни для сітки
                    pending_changes.append((new_r, new_c, bot)) # Поставити бота
                    pending_changes.append((current_r, current_c, None)) # Звільнити старе місце
                # else: Рух не вдався через конфлікт, бот залишається на місці (final_r, final_c = current_r, current_c)
            # else: Рух не вдався через стіну/межу (final_r, final_c = current_r, current_c)

            # Записуємо фінальну позицію бота для цього кроку
            next_bot_positions[bot] = (final_r, final_c)

            # --- TODO: Обробка взаємодій ---
            # Якщо бот опинився на клітинці з їжею (якщо can_move було True і target_cell був Food)
            # if final_r != current_r or final_c != current_c: # Якщо бот рухався
            #     final_cell_content = self.grid_data[final_r][final_c] # Що було в клітинці КУДИ він іде
            #     if isinstance(final_cell_content, Food):
            #          # З'їсти їжу
            #          eaten_energy = final_cell_content.properties.get('energy', 0)
            #          bot.properties['energy'] = min(MAX_ENERGY, bot.properties['energy'] + eaten_energy)
            #          print(f"Bot {bot.properties.get('id')} ate food! Energy: {bot.properties['energy']}")
            #          # Потрібно видалити їжу з pending_changes або оновити відповідний запис,
            #          # щоб їжа не була намальована після того, як її з'їли.
            #          # Або запланувати її видалення в pending_changes (краще)
            #          # Важливо: поточна логіка pending_changes може перезаписати їжу ботом.
            #          # Потрібно ретельніше продумати обробку взаємодій.


        # --- Застосування змін ---
        if pending_changes:
            num_bots_moved = sum(1 for r, c, ent in pending_changes if isinstance(ent, Bot))
            # print(f"Applying {len(pending_changes)} changes ({num_bots_moved} bots moved).") # Відладка

            for r_change, c_change, new_entity in pending_changes:
                 if 0 <= r_change < GRID_HEIGHT and 0 <= c_change < GRID_WIDTH:
                     self.grid_data[r_change][c_change] = new_entity
                 # else: print(f"Warning: Change out of bounds ({r_change},{c_change})") # Відладка

            self.bot_positions = next_bot_positions
            self.grid_widget.update()

        # --- Оновлення стану кнопок (як раніше) ---
        if not self.simulation_running:
             self.start_button.setEnabled(True)
             self.stop_button.setEnabled(False)
             self.step_button.setEnabled(True)