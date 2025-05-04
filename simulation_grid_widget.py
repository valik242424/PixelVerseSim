from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QPainter, QColor, QKeyEvent, QBrush, QPen, QMouseEvent
from PySide6.QtCore import Qt, QRect, Signal, Slot

# Забираємо залежність від конфігів розміру сітки, вона тепер не потрібна тут
from config import VIEW_SIZE, CELL_SIZE_PX
# Забираємо пряму залежність від Entity, хоча знаємо, що в grid_data будуть об'єкти з методом get_color()
# from entities import Entity # Не імпортуємо напряму

class SimulationGridWidget(QWidget):
    # Сигнали:
    coordinatesChanged = Signal(int, int) # (view_row_offset, view_col_offset)
    cellClicked = Signal(int, int)        # (grid_row, grid_col) - БЕЗ state_info

    def __init__(self, initial_grid_data, parent=None):
        super().__init__(parent)
        # Важливо: grid_data тепер просто посилання, яке оновлюється ззовні
        self.grid_data = initial_grid_data
        # Отримуємо розміри з переданих даних (безпечніше)
        self.grid_rows = len(self.grid_data)
        self.grid_cols = len(self.grid_data[0]) if self.grid_rows > 0 else 0

        self.view_rows = VIEW_SIZE
        self.view_cols = VIEW_SIZE

        # Початкове зміщення (можна зробити параметром або взяти з конфігу)
        self.view_row_offset = max(0, (self.grid_rows // 2) - (self.view_rows // 2))
        self.view_col_offset = max(0, (self.grid_cols // 2) - (self.view_cols // 2))

        # Прапорці стану для клавіш руху (без змін)
        self.key_w_pressed = False
        self.key_a_pressed = False
        self.key_s_pressed = False
        self.key_d_pressed = False

        # Кольори визначаємо тут
        self.empty_color = QColor(Qt.GlobalColor.white)
        self.default_color = QColor(Qt.GlobalColor.lightGray) # Колір поза межами сітки
        self.grid_line_color = QColor(Qt.GlobalColor.black)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Робимо віджет розтягуваним
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Встановлюємо розумний мінімальний розмір
        self.setMinimumSize(self.view_cols * 10, self.view_rows * 10)

    def calculate_cell_size(self):
        """Розраховує розмір клітинки на основі поточного розміру віджета."""
        widget_w = self.width()
        widget_h = self.height()
        # Забезпечуємо, щоб ділення не було на нуль, якщо view_cols/rows не задані
        cell_w = max(1, widget_w // self.view_cols if self.view_cols > 0 else widget_w)
        cell_h = max(1, widget_h // self.view_rows if self.view_rows > 0 else widget_h)
        # Клітинки квадратні, беремо менший розмір
        cell_size = min(cell_w, cell_h)
        return cell_size

    @Slot() # Явно позначаємо як слот, хоча для paintEvent це не обов'язково
    def update(self):
        """Перевизначаємо update, щоб переконатись, що він викликає QWidget.update()"""
        super().update() # Викликаємо стандартний метод перемальовки

    def paintEvent(self, event):
        """Малює видиму частину сітки."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False) # Чіткі пікселі

        cell_size = self.calculate_cell_size()
        if cell_size == 0: return # Немає що малювати

        total_view_width = self.view_cols * cell_size
        total_view_height = self.view_rows * cell_size

        # Малюємо клітинки
        for r_view in range(self.view_rows):
            for c_view in range(self.view_cols):
                grid_r = self.view_row_offset + r_view
                grid_c = self.view_col_offset + c_view

                entity_color = self.default_color # Колір за замовчуванням (поза сіткою)

                if 0 <= grid_r < self.grid_rows and 0 <= grid_c < self.grid_cols:
                    entity = self.grid_data[grid_r][grid_c] # Отримуємо вміст
                    if entity is None:
                        entity_color = self.empty_color # Порожня клітинка
                    elif hasattr(entity, 'get_color'): # Перевіряємо наявність методу
                        # Отримуємо колір як (R, G, B) і створюємо QColor
                        rgb = entity.get_color()
                        if isinstance(rgb, (tuple, list)) and len(rgb) == 3:
                            entity_color = QColor(rgb[0], rgb[1], rgb[2])
                        else:
                            # Якщо get_color повернув щось не те, використовуємо сірий
                            print(f"Warning: Entity at ({grid_r},{grid_c}) returned invalid color: {rgb}")
                            entity_color = QColor(128, 128, 128) # Сірий
                    else:
                        # Якщо об'єкт не має get_color, теж малюємо сірим
                         print(f"Warning: Entity at ({grid_r},{grid_c}) has no get_color method: {type(entity)}")
                         entity_color = QColor(128, 128, 128) # Сірий

                # Розрахунок координат для малювання
                x = c_view * cell_size
                y = r_view * cell_size
                rect = QRect(x, y, cell_size, cell_size)

                painter.setBrush(QBrush(entity_color))
                # Малюємо тонку рамку для кожної клітинки (можна вимкнути для швидкості)
                if cell_size > 2: # Малюємо лінії, тільки якщо клітинки достатньо великі
                     painter.setPen(QPen(self.grid_line_color, 1))
                else:
                     painter.setPen(Qt.PenStyle.NoPen) # Без рамки для дуже маленьких клітинок
                painter.drawRect(rect)

        # Малюємо товстішу рамку навколо всього видимого вікна
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self.grid_line_color, 2)) # Чорна рамка товщиною 2
        # Малюємо трохи всередині, щоб не обрізалась
        painter.drawRect(0, 0, total_view_width - 1, total_view_height - 1)

    @Slot(list) # Приймає новий стан сітки
    def update_data(self, new_grid_data):
        """Оновлює внутрішнє посилання на дані сітки."""
        self.grid_data = new_grid_data
        # Оновлюємо розміри на випадок, якщо вони змінились (хоча не повинні)
        self.grid_rows = len(self.grid_data)
        self.grid_cols = len(self.grid_data[0]) if self.grid_rows > 0 else 0
        # Тут НЕ викликаємо self.update() автоматично.
        # MainWindow вирішує, коли перемалювати після оновлення даних.

    def set_view_position(self, row, col):
        """Встановлює нову позицію лівого верхнього кута вікна перегляду."""
        # Обмежуємо зміщення межами сітки
        new_row_offset = max(0, min(row, self.grid_rows - self.view_rows))
        new_col_offset = max(0, min(col, self.grid_cols - self.view_cols))

        if new_row_offset != self.view_row_offset or new_col_offset != self.view_col_offset:
            self.view_row_offset = new_row_offset
            self.view_col_offset = new_col_offset
            self.update() # Перемальовуємо після зміни зміщення
            self.coordinatesChanged.emit(self.view_row_offset, self.view_col_offset) # Повідомляємо MainWindow

    def move_view(self, dr, dc):
        """Зміщує вікно перегляду на dr рядків і dc стовпців."""
        self.set_view_position(self.view_row_offset + dr, self.view_col_offset + dc)

    def get_view_offset(self):
        """Повертає поточне зміщення вікна перегляду (row, col)."""
        return self.view_row_offset, self.view_col_offset

    # --- Обробка клавіатури (без змін відносно попередньої версії) ---
    def keyPressEvent(self, event: QKeyEvent):
        accepted = False
        if not event.isAutoRepeat():
            key = event.key()
            if key == Qt.Key.Key_W: self.key_w_pressed = True; accepted = True
            elif key == Qt.Key.Key_A: self.key_a_pressed = True; accepted = True
            elif key == Qt.Key.Key_S: self.key_s_pressed = True; accepted = True
            elif key == Qt.Key.Key_D: self.key_d_pressed = True; accepted = True
            # TODO: Додати обробку +/- для масштабування (зміни VIEW_SIZE?)

        if self.key_w_pressed or self.key_a_pressed or self.key_s_pressed or self.key_d_pressed:
             self.process_movement()
             accepted = True

        if accepted:
            event.accept()
        else:
            super().keyPressEvent(event)

    def process_movement(self):
        dr = 0; dc = 0
        if self.key_w_pressed: dr -= 1
        if self.key_s_pressed: dr += 1
        if self.key_a_pressed: dc -= 1
        if self.key_d_pressed: dc += 1
        if dr != 0 or dc != 0:
            self.move_view(dr, dc)

    def keyReleaseEvent(self, event: QKeyEvent):
        accepted = False
        if not event.isAutoRepeat():
            key = event.key()
            if key == Qt.Key.Key_W: self.key_w_pressed = False; accepted = True
            elif key == Qt.Key.Key_A: self.key_a_pressed = False; accepted = True
            elif key == Qt.Key.Key_S: self.key_s_pressed = False; accepted = True
            elif key == Qt.Key.Key_D: self.key_d_pressed = False; accepted = True

        if accepted:
            event.accept()
        else:
            super().keyReleaseEvent(event)

    # --- Обробка миші ---
    def mousePressEvent(self, event: QMouseEvent):
        """Обробляє клік миші."""
        if event.button() == Qt.MouseButton.LeftButton:
            cell_size = self.calculate_cell_size()
            if cell_size == 0: return

            click_x = event.position().x()
            click_y = event.position().y()

            view_c = int(click_x // cell_size)
            view_r = int(click_y // cell_size)

            # Перевіряємо, чи клік був у межах видимих стовпців/рядків
            if 0 <= view_r < self.view_rows and 0 <= view_c < self.view_cols:
                grid_r = self.view_row_offset + view_r
                grid_c = self.view_col_offset + view_c

                # Перевіряємо, чи глобальні координати в межах сітки
                if 0 <= grid_r < self.grid_rows and 0 <= grid_c < self.grid_cols:
                    # Просто відправляємо глобальні координати кліку
                    self.cellClicked.emit(grid_r, grid_c)
                    event.accept()
                    return # Клік оброблено

        super().mousePressEvent(event) # Передаємо далі, якщо не ліва кнопка або поза межами