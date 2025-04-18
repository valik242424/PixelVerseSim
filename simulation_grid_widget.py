# simulation_grid_widget.py
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QPainter, QColor, QKeyEvent, QBrush, QPen
from PySide6.QtCore import Qt, QRect, Signal

from config import GRID_WIDTH, GRID_HEIGHT, VIEW_SIZE, CELL_SIZE_PX
from entities import Entity # <--- Імпортуємо базовий клас Entity

class SimulationGridWidget(QWidget):
    viewMoved = Signal(str) # Можливо, цей сигнал вже не потрібен? Залишимо поки що.
    coordinatesChanged = Signal(int, int)
    # Змінимо сигнатуру cellClicked, щоб вона передавала сам об'єкт Entity (або None)
    # Але для простоти логування залишимо рядок, як і було
    cellClicked = Signal(int, int, str) # (рядок, стовпець, інформація_про_стан)

    def __init__(self, grid_data, parent=None):
        super().__init__(parent)
        self.grid_data = grid_data # Тепер це сітка з None або Entity
        self.grid_rows = len(grid_data)
        self.grid_cols = len(grid_data[0]) if self.grid_rows > 0 else 0
        self.view_rows = VIEW_SIZE
        self.view_cols = VIEW_SIZE

        self.view_row_offset = 0
        self.view_col_offset = 0

        # Словник self.colors більше не потрібен тут, колір береться з Entity
        # self.colors = { ... } # <--- ВИДАЛИТИ або закоментувати

        self.empty_color = QColor(Qt.GlobalColor.white) # <--- Колір для порожніх клітинок (None)
        self.default_color = QColor(Qt.GlobalColor.lightGray) # <--- Колір для клітинок поза межами сітки

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(self.view_cols * CELL_SIZE_PX, self.view_rows * CELL_SIZE_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def calculate_cell_size(self):
        widget_w = self.width()
        widget_h = self.height()
        cell_w = max(1, widget_w // self.view_cols)
        cell_h = max(1, widget_h // self.view_rows)
        cell_size = min(cell_w, cell_h)
        return cell_size

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        cell_size = self.calculate_cell_size()
        if cell_size == 0: return

        for r in range(self.view_rows):
            for c in range(self.view_cols):
                grid_r = self.view_row_offset + r
                grid_c = self.view_col_offset + c

                color = self.default_color # Колір за замовчуванням (поза сіткою)
                entity = None # Сутність у цій клітинці

                if 0 <= grid_r < self.grid_rows and 0 <= grid_c < self.grid_cols:
                    entity = self.grid_data[grid_r][grid_c] # Отримуємо вміст клітинки
                    if entity is None:
                        color = self.empty_color # Порожня клітинка
                    elif isinstance(entity, Entity): # Перевіряємо, чи це наша сутність
                        color = entity.color # Беремо колір з сутності
                    # else: можна додати обробку несподіваних даних у сітці

                x = c * cell_size
                y = r * cell_size
                rect = QRect(x, y, cell_size, cell_size)

                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.drawRect(rect)

        # Малюємо рамку навколо видимого вікна
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(0, 0, self.view_cols * cell_size -1, self.view_rows * cell_size -1 )

    def set_view_position(self, row, col):
        new_row_offset = max(0, min(row, self.grid_rows - self.view_rows))
        new_col_offset = max(0, min(col, self.grid_cols - self.view_cols))

        if new_row_offset != self.view_row_offset or new_col_offset != self.view_col_offset:
            old_row, old_col = self.view_row_offset, self.view_col_offset
            self.view_row_offset = new_row_offset
            self.view_col_offset = new_col_offset
            self.update()
            # Випромінюємо сигнал про зміну координат (для мітки)
            self.coordinatesChanged.emit(self.view_row_offset, self.view_col_offset)
            # Випромінюємо сигнал для логу (можна додати більше інформації)
            self.viewMoved.emit(f"View moved from ({old_row},{old_col}) to ({self.view_row_offset},{self.view_col_offset})")


    def move_view(self, dr, dc):
        self.set_view_position(self.view_row_offset + dr, self.view_col_offset + dc)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        moved = False
        if key == Qt.Key.Key_W:
            self.move_view(-1, 0); moved = True
        elif key == Qt.Key.Key_S:
            self.move_view(1, 0); moved = True
        elif key == Qt.Key.Key_A:
            self.move_view(0, -1); moved = True
        elif key == Qt.Key.Key_D:
            self.move_view(0, 1); moved = True

        if moved:
            event.accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        cell_size = self.calculate_cell_size()
        if cell_size == 0: return

        click_x = event.position().x()
        click_y = event.position().y()

        view_c = int(click_x // cell_size)
        view_r = int(click_y // cell_size)

        grid_r = self.view_row_offset + view_r
        grid_c = self.view_col_offset + view_c

        state_info = "Out of bounds" # За замовчуванням, якщо клік поза сіткою

        if 0 <= view_r < self.view_rows and 0 <= view_c < self.view_cols:
             if 0 <= grid_r < self.grid_rows and 0 <= grid_c < self.grid_cols:
                entity = self.grid_data[grid_r][grid_c] # Отримуємо сутність (або None)
                if entity is None:
                    state_info = "Empty cell"
                elif isinstance(entity, Entity):
                    state_info = entity.get_state_info() # Викликаємо метод сутності
                else:
                    state_info = f"Unknown data: {entity}" # На випадок помилки

                self.cellClicked.emit(grid_r, grid_c, state_info) # Випромінюємо сигнал

        super().mousePressEvent(event)