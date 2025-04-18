# entities.py
import random
from PySide6.QtGui import QColor, Qt
# Імпортуємо константи розмірів сітки з config.py для перевірки меж
from config import GRID_WIDTH, GRID_HEIGHT

class Entity:
    """Базовий клас для всіх об'єктів на сітці."""
    def __init__(self, entity_type="generic", color=QColor(Qt.GlobalColor.gray), properties=None):
        """
        Ініціалізатор базової сутності.

        Args:
            entity_type (str): Рядок, що ідентифікує тип сутності (напр., "wall", "bot").
            color (QColor): Колір для візуалізації сутності на сітці.
            properties (dict, optional): Словник з додатковими властивостями сутності.
                                         За замовчуванням None (створюється порожній словник).
        """
        self.entity_type = entity_type
        self.color = color
        # Створюємо копію словника properties або новий порожній, щоб уникнути спільного змінюваного стану
        self.properties = properties.copy() if properties is not None else {}

    def get_state_info(self):
        """
        Повертає рядок з інформацією про стан сутності для логування або відображення.
        Формує рядок з типу та властивостей сутності.
        """
        # Створюємо рядок з властивостей виду "key1: value1, key2: value2"
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        if prop_str:
            # Якщо є властивості, повертаємо тип і властивості
            return f"Type: {self.entity_type}, Properties: {prop_str}"
        else:
            # Якщо властивостей немає, повертаємо тільки тип
            return f"Type: {self.entity_type}"

    def update(self, grid_data, row, col):
        """
        Метод для оновлення стану сутності на кожному кроці симуляції.
        Базовий клас нічого не робить і має бути перевизначений у похідних класах,
        якщо сутність має якусь поведінку (наприклад, рух, взаємодія).

        Args:
            grid_data (list[list[Entity or None]]): Посилання на всю сітку симуляції.
                                                    Дозволяє сутності "бачити" своє оточення.
            row (int): Поточний рядок сутності на сітці.
            col (int): Поточний стовпець сутності на сітці.

        Returns:
            Any: Результат оновлення. Формат залежить від похідного класу.
                 Для рухомих сутностей може повертати нові координати або інформацію про зміни.
                 Базова реалізація повертає None.
        """
        # Базова реалізація нічого не робить
        return None

# --- Похідні класи ---

class Wall(Entity):
    """Клас для стін - нерухомих перешкод."""
    def __init__(self):
        """Ініціалізує стіну з типом "wall" і темно-сірим кольором."""
        super().__init__(entity_type="wall", color=QColor(Qt.GlobalColor.darkGray))
        # Стіни не мають активної поведінки, тому метод update не перевизначається.
        # Додаткові властивості також не потрібні за замовчуванням.

class Bot(Entity):
    """Клас для рухомих ботів."""
    def __init__(self, bot_id, color=QColor(Qt.GlobalColor.red), energy=100):
        """
        Ініціалізує бота.

        Args:
            bot_id (str): Унікальний ідентифікатор бота.
            color (QColor, optional): Колір бота. За замовчуванням червоний.
            energy (int, optional): Початковий рівень енергії бота. За замовчуванням 100.
        """
        # Викликаємо ініціалізатор базового класу, передаючи тип "bot", колір
        # та словник з початковими властивостями: id та energy.
        super().__init__(entity_type="bot", color=color, properties={"id": bot_id, "energy": energy})

    def update(self, grid_data, row, col):
        """
        Виконує один крок логіки бота: спроба зробити випадковий крок.
        Перевіряє межі сітки та наявність перешкод у цільовій клітинці.

        Args:
            grid_data (list[list[Entity or None]]): Поточний стан сітки симуляції.
            row (int): Поточний рядок бота.
            col (int): Поточний стовпець бота.

        Returns:
            tuple[int, int] or None: Повертає кортеж (new_r, new_c) з новими координатами,
                                     якщо рух можливий і вдалий.
                                     Повертає None, якщо бот не може рухатися (межа, перешкода)
                                     або вирішив залишитися на місці (якщо така логіка додана).
        """
        # Можливі варіанти зміщення: вгору, вниз, вліво, вправо
        possible_moves = [
            (-1, 0),  # Вгору (зменшення рядка)
            (1, 0),   # Вниз (збільшення рядка)
            (0, -1),  # Вліво (зменшення стовпця)
            (0, 1)    # Вправо (збільшення стовпця)
            # Можна додати сюди (0, 0), якщо бот може вирішити залишитися на місці
        ]

        # Вибираємо випадкове зміщення з можливих
        dr, dc = random.choice(possible_moves)

        # Розраховуємо потенційні нові координати
        new_r, new_c = row + dr, col + dc

        # --- Перевірки можливості руху ---

        # 1. Перевірка виходу за межі сітки
        # Використовуємо GRID_WIDTH і GRID_HEIGHT, імпортовані з config.py
        if not (0 <= new_r < GRID_HEIGHT and 0 <= new_c < GRID_WIDTH):
            # print(f"Bot {self.properties.get('id')} at ({row},{col}) tried move to ({new_r},{new_c}) - Out of bounds") # Для відладки
            return None # Рух неможливий, вихід за межі

        # 2. Перевірка, чи цільова клітинка вільна (тобто містить None)
        target_cell = grid_data[new_r][new_c]
        if target_cell is not None:
            # Клітинка зайнята іншою сутністю (стіною або іншим ботом)
            # entity_type = getattr(target_cell, 'entity_type', 'Unknown') # Безпечне отримання типу
            # print(f"Bot {self.properties.get('id')} at ({row},{col}) tried move to ({new_r},{new_c}) - Blocked by {entity_type}") # Для відладки
            return None # Рух неможливий, клітинка зайнята

        # --- Рух можливий ---
        # Якщо всі перевірки пройдені, бот може переміститися в нову клітинку.
        # print(f"Bot {self.properties.get('id')} moving from ({row},{col}) to ({new_r},{new_c})") # Для відладки

        # Повертаємо нові координати, куди бот хоче переміститися
        return (new_r, new_c)

    def get_state_info(self):
        """
        Перевизначаємо метод для можливого додавання специфічної для бота інформації в майбутньому.
        Наразі просто викликає базову реалізацію.
        """
        # Можна додати більше інформації, наприклад, поточну енергію:
        # base_info = super().get_state_info()
        # return f"{base_info}"
        # Поки що залишимо як є, базова реалізація показує ID та енергію.
        return super().get_state_info()

# Можна додавати інші класи сутностей тут, наприклад:
# class Food(Entity):
#     def __init__(self, energy_value=10):
#         super().__init__(entity_type="food", color=QColor(Qt.GlobalColor.yellow), properties={"energy": energy_value})
#         # Їжа зазвичай не має методу update, вона просто існує