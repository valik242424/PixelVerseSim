import sys
from PySide6.QtWidgets import QApplication

# Імпортуємо клас головного вікна
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())