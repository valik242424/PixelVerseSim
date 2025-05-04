import sys
import random
import os # Потрібен для роботи зі шляхами збереження/завантаження
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLabel, QSlider, QPushButton, QCheckBox, QFileDialog, QSpacerItem, QSizePolicy # Додали QCheckBox, QFileDialog, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor, Qt as QtConstants

from config import GRID_WIDTH, GRID_HEIGHT, VIEW_SIZE
from simulation_grid_widget import SimulationGridWidget
from simulation_engine import SimulationEngine
from entities import Bot

# Шлях за замовчуванням для збереження/завантаження моделей
DEFAULT_MODEL_PATH_PREFIX = "models/bot_model"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelVerseSim v2 - RL Training") # Оновив заголовок
        self.setGeometry(100, 100, 1000, 600) # Зробив вікно трохи ширшим

        # --- 1. Створюємо та ініціалізуємо Рушій Симуляції ---
        self.engine = SimulationEngine(width=GRID_WIDTH, height=GRID_HEIGHT)
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
        log_area_container.setMinimumWidth(250) # Трохи ширше
        log_area_container.setMaximumWidth(400)
        main_layout.addWidget(log_area_container, 1)

        # --- 4. Центральна панель (Сітка симуляції) ---
        self.grid_widget = SimulationGridWidget(initial_grid_data)
        main_layout.addWidget(self.grid_widget, 3)

        # --- 5. Права панель (Елементи керування) ---
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_label = QLabel("Панель керування:")
        controls_layout.addWidget(controls_label)

        # --- Кнопки керування симуляцією ---
        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.step_button = QPushButton("Крок")
        self.reset_button = QPushButton("Скинути")
        self.stop_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.step_button)
        controls_layout.addLayout(button_layout)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addSpacing(15)

        # --- Лічильник Кроків ---
        self.step_counter_label = QLabel(f"Крок: {self.engine.current_step}")
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
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(200)
        self.speed_slider.setInvertedAppearance(True)
        controls_layout.addWidget(slider_label)
        controls_layout.addWidget(self.speed_slider)
        controls_layout.addSpacing(25) # Більший відступ

        # --- Керування Навчанням ---
        training_label = QLabel("Навчання (RL):")
        controls_layout.addWidget(training_label)

        # Чекбокс для ввімкнення/вимкнення навчання
        self.training_checkbox = QCheckBox("Режим тренування")
        self.training_checkbox.setChecked(self.engine._is_training) # Встановлюємо початковий стан з рушія
        controls_layout.addWidget(self.training_checkbox)

        # Кнопки збереження/завантаження
        self.save_models_button = QPushButton("Зберегти моделі")
        self.load_models_button = QPushButton("Завантажити моделі")
        model_button_layout = QHBoxLayout()
        model_button_layout.addWidget(self.save_models_button)
        model_button_layout.addWidget(self.load_models_button)
        controls_layout.addLayout(model_button_layout)

        # Додаємо розтягувач, щоб притиснути все догори
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        controls_layout.addItem(spacer)

        controls_container.setMinimumWidth(200) # Збільшив мінімальну ширину
        controls_container.setMaximumWidth(300)
        main_layout.addWidget(controls_container, 1)

        # --- 6. Налаштування таймера симуляції ---
        self.timer = QTimer(self)
        self.timer.setInterval(self.speed_slider.value())
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
        # Нові сигнали для навчання
        self.training_checkbox.stateChanged.connect(self.toggle_training_mode)
        self.save_models_button.clicked.connect(self.save_models)
        self.load_models_button.clicked.connect(self.load_models)

        # --- 8. Початкові логи ---
        self.add_log("Симуляція (v2 - RL) ініціалізована.")
        self.add_log(f"Розмір сітки: {GRID_WIDTH}x{GRID_HEIGHT}. Вікно: {VIEW_SIZE}x{VIEW_SIZE}")
        self.add_log("WASD - навігація, Клік - інфо.")
        self.add_log("Керування: Старт/Стоп/Крок/Скинути.")
        self.add_log(f"Режим тренування: {'Увімкнено' if self.engine._is_training else 'Вимкнено'}")
        self.add_log(f"Початковий стан: Крок {self.engine.current_step}")

        self.update_control_buttons_state()
        self.grid_widget.setFocus()

    # --- Слоти для обробки сигналів ---

    @Slot(str)
    def add_log(self, message):
        self.log_text_edit.append(message)
        # Автопрокрутка донизу (опціонально)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())


    @Slot(int, int)
    def handle_cell_click(self, grid_r, grid_c):
        entity = self.engine.get_entity_at(grid_r, grid_c)
        if entity:
            state_info = entity.get_state_info()
            # Додамо інфо про режим тренування бота, якщо це бот
            if isinstance(entity, Bot):
                 training_status = "Training" if entity.agent.is_training else "Evaluating"
                 state_info += f" (Mode: {training_status})"
            self.add_log(f"Клік: [{grid_r}, {grid_c}] -> {state_info}")
        else:
            self.add_log(f"Клік: [{grid_r}, {grid_c}] -> Порожньо")

    @Slot(int, int)
    def update_coordinates_label(self, view_row, view_col):
        self.coordinates_label.setText(f"Координати вікна: ({view_row}, {view_col})")

    @Slot(int)
    def update_timer_interval(self, value):
        self.timer.setInterval(value)

    # --- Слоти для керування навчанням ---

    @Slot(int)
    def toggle_training_mode(self, state):
        """Вмикає/вимикає режим навчання для всіх ботів."""
        is_training = state == Qt.CheckState.Checked.value # Перевіряємо стан чекбокса
        self.engine.set_training_mode(is_training)
        mode_str = "Увімкнено" if is_training else "Вимкнено"
        self.add_log(f"Режим тренування {mode_str}.")
        # Можливо, варто скинути симуляцію при зміні режиму? Або ні? Поки не будемо.

    @Slot()
    def save_models(self):
        """Зберігає моделі поточних ботів."""
        # Зупиняємо симуляцію на час збереження (про всяк випадок)
        was_running = self.simulation_running
        if was_running:
            self.stop_simulation()

        # Використовуємо діалог для вибору префіксу файлу
        # Початкова директорія - директорія зі скриптом + /models
        initial_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(initial_dir, exist_ok=True) # Створюємо, якщо немає

        # Пропонуємо стандартний префікс, але дозволяємо змінити
        # Зауваження: QFileDialog.getSaveFileName повертає (filePath, selectedFilter)
        file_path_prefix, _ = QFileDialog.getSaveFileName(
            self,
            "Зберегти префікс моделі",
            os.path.join(initial_dir, "bot_model"), # Пропонований шлях з префіксом
            "Model Prefix (*)" # Фільтр не дуже важливий тут
        )

        if file_path_prefix: # Якщо користувач не натиснув Cancel
            self.add_log(f"Збереження моделей з префіксом: {file_path_prefix}...")
            try:
                self.engine.save_models(file_path_prefix)
                self.add_log("Моделі успішно збережено.")
            except Exception as e:
                self.add_log(f"Помилка збереження моделей: {e}")
        else:
            self.add_log("Збереження моделей скасовано.")

        # Відновлюємо симуляцію, якщо вона була запущена
        if was_running:
            self.start_simulation()
        self.grid_widget.setFocus() # Повертаємо фокус

    @Slot()
    def load_models(self):
        """Завантажує моделі для поточних ботів."""
        was_running = self.simulation_running
        if was_running:
            self.stop_simulation()

        initial_dir = os.path.join(os.path.dirname(__file__), "models")

        # Вибираємо префікс існуючих файлів
        # Тут логіка трохи інша - нам потрібен префікс, а не конкретний файл
        # Можна попросити користувача вибрати ОДИН файл моделі, а ми витягнемо префікс
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Виберіть файл моделі для завантаження (префікс буде визначено автоматично)",
            initial_dir,
            "PyTorch Model Files (*.pth);;All Files (*)"
        )

        if file_path:
            # Визначаємо префікс зі шляху (видаляємо "_BotID.pth")
            try:
                base_name = os.path.basename(file_path)
                # Знаходимо останнє підкреслення перед ".pth"
                parts = base_name.split('_')
                if len(parts) > 1 and parts[-1].endswith(".pth"):
                    prefix_part = "_".join(parts[:-1])
                    path_prefix = os.path.join(os.path.dirname(file_path), prefix_part)
                    self.add_log(f"Завантаження моделей з префіксом: {path_prefix}...")
                    self.engine.load_models(path_prefix)
                    self.add_log("Спроба завантаження моделей завершена.")
                    # Оновлюємо відображення, щоб побачити нових (можливо, розумніших) ботів
                    self._update_simulation_view()
                else:
                     self.add_log("Не вдалося визначити префікс з імені файлу. Очікуваний формат: prefix_BotID.pth")

            except Exception as e:
                self.add_log(f"Помилка завантаження моделей: {e}")
        else:
            self.add_log("Завантаження моделей скасовано.")

        if was_running:
            self.start_simulation()
        self.grid_widget.setFocus()

    # --- Методи керування симуляцією ---

    def _update_simulation_view(self):
        updated_grid = self.engine.get_grid_data()
        self.grid_widget.update_data(updated_grid)
        self.grid_widget.update()
        self.step_counter_label.setText(f"Крок: {self.engine.current_step}")

    @Slot()
    def start_simulation(self):
        if not self.simulation_running:
            self.simulation_running = True
            self.timer.start()
            self.update_control_buttons_state()
            self.add_log("Симуляцію запущено.")

    @Slot()
    def stop_simulation(self):
        if self.simulation_running:
            self.simulation_running = False
            self.timer.stop()
            self.update_control_buttons_state()
            self.add_log("Симуляцію зупинено.")

    @Slot()
    def execute_single_step(self):
        if self.simulation_running:
             self.stop_simulation()

        step_logs = self.engine.step()
        for log_msg in step_logs:
            self.add_log(log_msg)
        self._update_simulation_view()

    @Slot()
    def simulation_step_triggered_by_timer(self):
        step_logs = self.engine.step()
        # Можливо, не логувати кожен крок при швидкій симуляції?
        # Або логувати тільки важливі події (смерть, їжа)?
        # Поки що логуємо все:
        for log_msg in step_logs:
            self.add_log(log_msg)
        self._update_simulation_view()

    @Slot()
    def reset_simulation(self):
        self.stop_simulation()
        self.add_log("--- С К И Д А Н Н Я   С И М У Л Я Ц І Ї ---")
        # Запам'ятовуємо поточний стан чекбокса тренування
        is_training_checked = self.training_checkbox.isChecked()
        # Рушій сам встановить режим тренування при ініціалізації
        self.engine.initialize_world() # Перестворює світ і ботів
        # Встановлюємо режим тренування відповідно до чекбокса
        self.engine.set_training_mode(is_training_checked)
        self.training_checkbox.setChecked(is_training_checked) # Оновлюємо чекбокс про всяк випадок

        self._update_simulation_view() # Покаже "Крок: 0"
        self.log_text_edit.clear()
        self.add_log("Симуляцію скинуто до початкового стану.")
        mode_str = "Увімкнено" if is_training_checked else "Вимкнено"
        self.add_log(f"Режим тренування: {mode_str}")
        self.update_control_buttons_state()
        self.grid_widget.setFocus()

    def update_control_buttons_state(self):
        running = self.simulation_running
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.step_button.setEnabled(not running)
        self.reset_button.setEnabled(True)
        # Кнопки збереження/завантаження активні, коли симуляція зупинена? Чи завжди?
        # Поки що зробимо їх завжди активними, але зупинятимемо симуляцію під час операції.
        self.save_models_button.setEnabled(True)
        self.load_models_button.setEnabled(True)
        # Чекбокс тренування теж завжди активний
        self.training_checkbox.setEnabled(True)

    # def closeEvent(self, event):
    #     self.stop_simulation()
    #     super().closeEvent(event)