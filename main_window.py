import sys
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLabel, QSlider, QPushButton
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor, Qt as QtConstants # Використовуємо QtConstants як псевдонім

# Імпортуємо константи конфігурації
from config import GRID_WIDTH, GRID_HEIGHT, VIEW_SIZE

# Імпортуємо віджет сітки ТА новий рушій симуляції
from simulation_grid_widget import SimulationGridWidget
from simulation_engine import SimulationEngine

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelVerseSim v2 - Engine Powered")
        self.setGeometry(100, 100, 950, 550) # Трохи збільшив розмір вікна

        # --- 1. Створюємо та ініціалізуємо Рушій Симуляції ---
        self.engine = SimulationEngine(width=GRID_WIDTH, height=GRID_HEIGHT)
        # Параметри ініціалізації можна винести в конфіг або UI
        self.engine.initialize_world(num_walls=50, num_bots=15, num_food=35)
        initial_grid_data = self.engine.get_grid_data()

        # --- 2. Створюємо центральний віджет та головний лейаут ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 3. Ліва панель (Логи) ---
        log_area_container = QWidget()
        log_layout = QVBoxLayout(log_area_container)
        log_label = QLabel("Логи симуляції:")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_text_edit)
        log_area_container.setMinimumWidth(200)
        log_area_container.setMaximumWidth(350) # Трохи ширше для логів
        main_layout.addWidget(log_area_container, 1) # Пропорція 1

        # --- 4. Центральна панель (Сітка симуляції) ---
        self.grid_widget = SimulationGridWidget(initial_grid_data)
        main_layout.addWidget(self.grid_widget, 3) # Пропорція 3 (більше місця для сітки)

        # --- 5. Права панель (Елементи керування) ---
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_label = QLabel("Панель керування:")
        controls_layout.addWidget(controls_label)

        # --- Кнопки ---
        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.step_button = QPushButton("Крок")
        self.reset_button = QPushButton("Скинути")
        self.stop_button.setEnabled(False) # Починаємо зупиненими

        # Групуємо основні кнопки керування разом
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.step_button)
        controls_layout.addLayout(button_layout)
        controls_layout.addWidget(self.reset_button) # Кнопка скидання окремо нижче
        controls_layout.addSpacing(15) # Відступ після кнопок

        # --- Лічильник Кроків ---
        self.step_counter_label = QLabel(f"Крок: {self.engine.current_step}") # Початкове значення 0
        controls_layout.addWidget(self.step_counter_label)
        controls_layout.addSpacing(15)

        # --- Координати вікна перегляду ---
        initial_row, initial_col = self.grid_widget.get_view_offset()
        self.coordinates_label = QLabel(f"Координати вікна: ({initial_row}, {initial_col})")
        controls_layout.addWidget(self.coordinates_label)
        controls_layout.addSpacing(15)

        # --- Слайдер швидкості ---
        slider_label = QLabel("Швидкість:")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)   # мін. інтервал мс (найшвидше)
        self.speed_slider.setMaximum(1000) # макс. інтервал мс (найповільніше)
        self.speed_slider.setValue(200)    # Початковий інтервал
        self.speed_slider.setInvertedAppearance(True) # Рух вправо = швидше
        controls_layout.addWidget(slider_label)
        controls_layout.addWidget(self.speed_slider)

        controls_layout.addStretch() # Розтягувач, щоб притиснути елементи догори
        controls_container.setMinimumWidth(180) # Змінив мінімальну ширину
        controls_container.setMaximumWidth(250)
        main_layout.addWidget(controls_container, 1) # Пропорція 1

        # --- 6. Налаштування таймера симуляції ---
        self.timer = QTimer(self)
        self.timer.setInterval(self.speed_slider.value()) # Беремо інтервал зі слайдера
        self.timer.timeout.connect(self.simulation_step_triggered_by_timer)
        self.simulation_running = False

        # --- 7. Підключення сигналів ---
        self.grid_widget.coordinatesChanged.connect(self.update_coordinates_label)
        self.grid_widget.cellClicked.connect(self.handle_cell_click)
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.step_button.clicked.connect(self.execute_single_step)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.speed_slider.valueChanged.connect(self.update_timer_interval)

        # --- 8. Початкові логи ---
        self.add_log("Симуляція (v2) ініціалізована з SimulationEngine.")
        self.add_log(f"Розмір сітки: {GRID_WIDTH}x{GRID_HEIGHT}. Вікно: {VIEW_SIZE}x{VIEW_SIZE}")
        self.add_log("WASD - навігація, Клік - інфо.")
        self.add_log("Використовуйте кнопки Старт/Стоп/Крок/Скинути.")
        self.add_log(f"Початковий стан: Крок {self.engine.current_step}") # Логуємо початковий крок

        self.update_control_buttons_state() # Встановлюємо початковий стан кнопок
        self.grid_widget.setFocus() # Фокус на віджеті сітки для керування

    # --- Слоти для обробки сигналів ---

    @Slot(str)
    def add_log(self, message):
        """Додає повідомлення в текстове поле логів."""
        self.log_text_edit.append(message)

    @Slot(int, int)
    def handle_cell_click(self, grid_r, grid_c):
        """Обробляє клік по клітинці, запитуючи інформацію з рушія."""
        entity = self.engine.get_entity_at(grid_r, grid_c)
        if entity:
            state_info = entity.get_state_info()
            self.add_log(f"Клік: [{grid_r}, {grid_c}] -> {state_info}")
        else:
            self.add_log(f"Клік: [{grid_r}, {grid_c}] -> Порожньо")

    @Slot(int, int)
    def update_coordinates_label(self, view_row, view_col):
        """Оновлює мітку з координатами вікна перегляду."""
        self.coordinates_label.setText(f"Координати вікна: ({view_row}, {view_col})")

    @Slot(int)
    def update_timer_interval(self, value):
        """Оновлює інтервал таймера відповідно до слайдера."""
        self.timer.setInterval(value)

    # --- Методи керування симуляцією ---

    def _update_simulation_view(self):
        """Оновлює відображення сітки та лічильник кроків."""
        updated_grid = self.engine.get_grid_data()
        self.grid_widget.update_data(updated_grid)
        self.grid_widget.update() # Викликає paintEvent
        self.step_counter_label.setText(f"Крок: {self.engine.current_step}") # Оновлює лейбл

    @Slot()
    def start_simulation(self):
        """Запускає автоматичне виконання кроків симуляції."""
        if not self.simulation_running:
            self.simulation_running = True
            self.timer.start()
            self.update_control_buttons_state()
            self.add_log("Симуляцію запущено.")

    @Slot()
    def stop_simulation(self):
        """Зупиняє автоматичне виконання кроків симуляції."""
        if self.simulation_running:
            self.simulation_running = False
            self.timer.stop()
            self.update_control_buttons_state()
            self.add_log("Симуляцію зупинено.")

    @Slot()
    def execute_single_step(self):
        """Виконує рівно один крок симуляції."""
        if self.simulation_running:
             self.stop_simulation() # Зупиняємо авто-режим, якщо він був активний

        # Викликаємо крок рушія
        step_logs = self.engine.step()
        for log_msg in step_logs: # Додаємо логи від рушія
            self.add_log(log_msg)

        # Оновлюємо відображення (сітка + лічильник)
        self._update_simulation_view()

    @Slot()
    def simulation_step_triggered_by_timer(self):
        """Виконує крок симуляції, викликаний таймером."""
        step_logs = self.engine.step()
        for log_msg in step_logs:
            self.add_log(log_msg)

        # Оновлюємо відображення (сітка + лічильник)
        self._update_simulation_view()

    @Slot()
    def reset_simulation(self):
        """Скидає симуляцію до початкового стану."""
        self.stop_simulation()
        self.add_log("--- С К И Д А Н Н Я   С И М У Л Я Ц І Ї ---")
        self.engine.initialize_world() # Реініціалізуємо світ (крок стає 0)

        # Оновлюємо відображення (сітка + лічильник)
        self._update_simulation_view() # Покаже "Крок: 0"

        self.log_text_edit.clear() # Очищуємо поле логів
        self.add_log("Симуляцію скинуто до початкового стану.")
        self.update_control_buttons_state() # Оновлюємо кнопки

    def update_control_buttons_state(self):
        """Оновлює стан активності кнопок керування."""
        running = self.simulation_running
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.step_button.setEnabled(not running)
        self.reset_button.setEnabled(True) # Завжди активна

    # --- Обробка закриття вікна (опціонально, для коректної зупинки) ---
    # def closeEvent(self, event):
    #     """Обробляє подію закриття вікна."""
    #     self.stop_simulation() # Зупиняємо симуляцію перед виходом
    #     # Тут можна додати збереження стану в майбутньому
    #     super().closeEvent(event)