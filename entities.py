# refactored_code/entities.py
import random
import torch
from PySide6.QtGui import QColor, Qt # Залежність від GUI ще є через QColor, подумаємо про це пізніше
from config import GRID_WIDTH, GRID_HEIGHT, MAX_ENERGY

# Імпортуємо ТІЛЬКИ необхідне з bot_brain (мозок і дії)
# INPUT_SIZE тепер визначається тут, бо він залежить від типів клітинок
from bot_brain import BotBrainNet, ACTIONS

# --- Константи для кодування входу (ТУТ) ---
CELL_TYPE_EMPTY = 0
CELL_TYPE_WALL = 1
CELL_TYPE_BOT = 2
# TODO: Розглянути додавання CELL_TYPE_FOOD = 3
NUM_CELL_TYPES = 3 # Food поки кодується як Empty

# ВИЗНАЧЕННЯ РОЗМІРУ ВХОДУ ТУТ
# Розмір входу залежить від кодування оточення
DIRECTIONS = ["north", "east", "south", "west"]
NUM_DIRECTIONS = len(DIRECTIONS)
# Розмір: К-ть напрямків * К-ть типів + 1 (енергія)
INPUT_SIZE = NUM_DIRECTIONS * NUM_CELL_TYPES + 1

# --- Функція для підготовки вхідного вектора (ТУТ) ---
def prepare_input_vector(surroundings, energy):
    """
    Готує вхідний вектор для нейромережі на основі оточення та енергії бота.
    Оточення - словник {напрямок: вміст_клітинки}.
    Енергія - float.
    Повертає torch.Tensor або None при помилці.
    """
    input_features = []

    for direction in DIRECTIONS:
        cell_content = surroundings.get(direction, None)
        one_hot = [0.0] * NUM_CELL_TYPES

        if cell_content is None:
            one_hot[CELL_TYPE_EMPTY] = 1.0
        elif isinstance(cell_content, Wall): # Wall визначений нижче
            one_hot[CELL_TYPE_WALL] = 1.0
        elif isinstance(cell_content, Bot):   # Bot визначений нижче
            one_hot[CELL_TYPE_BOT] = 1.0
        elif isinstance(cell_content, Food):  # Food визначений нижче
            # Поки що кодуємо їжу як порожню клітинку для мозку
            one_hot[CELL_TYPE_EMPTY] = 1.0
        else: # Невідомий об'єкт або об'єкт Entity без підкласу
             one_hot[CELL_TYPE_EMPTY] = 1.0

        input_features.extend(one_hot)

    # Перевірка розміру після оточення
    expected_surrounding_features = NUM_DIRECTIONS * NUM_CELL_TYPES
    if len(input_features) != expected_surrounding_features:
         print(f"Error: Incorrect number of features after encoding surroundings: {len(input_features)}, expected {expected_surrounding_features}")
         return None

    # Додавання нормалізованої енергії
    normalized_energy = max(0.0, min(1.0, energy / MAX_ENERGY))
    input_features.append(normalized_energy)

    # Перевірка фінального розміру вектора
    if len(input_features) != INPUT_SIZE:
        print(f"Error: Final input vector size is incorrect: {len(input_features)}, expected {INPUT_SIZE}")
        return None

    return torch.tensor(input_features, dtype=torch.float32)

# --- Базовий клас Entity ---
class Entity:
    def __init__(self, entity_type="generic", color=QColor(Qt.GlobalColor.gray), properties=None):
        self.entity_type = entity_type
        # Зберігаємо колір як рядок або кортеж (R, G, B), щоб уникнути залежності від Qt тут
        # Перетворимо QColor на кортеж RGB
        if isinstance(color, QColor):
            self.color_repr = (color.red(), color.green(), color.blue())
        else:
            # Якщо передали щось інше (напр. вже кортеж), використовуємо як є
            self.color_repr = color if color else (128, 128, 128) # Сірий за замовчуванням

        self.properties = properties.copy() if properties is not None else {}

    def get_state_info(self):
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        if prop_str:
            return f"Type: {self.entity_type}, Properties: {prop_str}"
        else:
            return f"Type: {self.entity_type}"

    def get_color(self):
        """Повертає колір як кортеж (R, G, B)."""
        return self.color_repr

    def update(self, grid_data, row, col):
        """Метод для оновлення стану сутності (перевизначається у нащадків)."""
        return None # За замовчуванням нічого не робить

# --- Клас Wall ---
class Wall(Entity):
    def __init__(self):
        super().__init__(entity_type="wall", color=QColor(Qt.GlobalColor.darkGray)) # QColor тут ок, бо це кінцевий клас

# --- Клас Food ---
class Food(Entity):
    def __init__(self, energy_value=25):
         super().__init__(entity_type="food", color=QColor(Qt.GlobalColor.yellow), properties={"energy": energy_value})

# --- Клас Bot ---
class Bot(Entity):
    def __init__(self, bot_id, color=QColor(Qt.GlobalColor.red), energy=100):
        super().__init__(entity_type="bot", color=color, properties={"id": bot_id, "energy": energy})
        # Мозок створюється тут, використовує INPUT_SIZE, визначений вище
        self.brain = BotBrainNet(input_size=INPUT_SIZE) # INPUT_SIZE з цього файлу
        self.brain.eval() # Переводимо мозок в режим оцінки
        self.logged_death = False # Прапорець логування смерті

    def update(self, grid_data, row, col):
        """
        Оновлює стан бота на основі оточення та внутрішнього стану.
        Повертає назву дії ('move_north', 'stay', etc.).
        Зменшує енергію за дію.
        """
        # --- 1. Збір інформації про оточення ---
        surroundings = {}
        for direction, (dr, dc) in zip(DIRECTIONS, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
            nr, nc = row + dr, col + dc
            if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                surroundings[direction] = grid_data[nr][nc]
            else:
                # Якщо вихід за межі, вважаємо це стіною для мозку
                surroundings[direction] = Wall()

        # --- 2. Отримання ПОТОЧНОЇ енергії ---
        current_energy = self.properties.get('energy', 0.0)

        # --- Перевірка, чи є енергія для дій ---
        if current_energy <= 0:
            if not self.logged_death:
                # Повідомлення про смерть логується в SimulationEngine, тут тільки стан
                # print(f"Bot {self.properties.get('id')} ran out of energy!") # Видаляємо прямий print
                self.logged_death = True
            return "stay" # Мертві не ходять

        # Скидаємо прапорець смерті, якщо енергія знову є (на випадок відновлення)
        if self.logged_death:
            self.logged_death = False

        # --- 3. Підготовка вхідного вектора ---
        # Викликаємо функцію, визначену в цьому ж файлі
        input_vector = prepare_input_vector(surroundings, current_energy)
        if input_vector is None:
            print(f"Warning: Bot {self.properties.get('id')} failed to prepare input vector. Staying put.")
            return "stay"

        # --- 4. Отримання оцінок дій від мозку ---
        try:
            with torch.no_grad():
                action_scores = self.brain(input_vector)
        except Exception as e:
            print(f"Error during brain forward pass for bot {self.properties.get('id')}: {e}")
            return "stay"

        # --- 5. Вибір найкращої дії ---
        chosen_action_index = torch.argmax(action_scores).item()
        chosen_action_name = ACTIONS[chosen_action_index] # ACTIONS імпортовано з bot_brain

        # --- 6. Зменшення енергії за крок/дію ---
        # Зменшуємо енергію *завжди* за один крок (навіть якщо стоїть)
        energy_cost = 1.0 # Базова вартість кроку
        # Можна додати вартість за рух:
        # if chosen_action_name != "stay":
        #     energy_cost += 0.5
        new_energy = max(0, current_energy - energy_cost)
        self.properties['energy'] = new_energy

        # Перевірка, чи енергія ТІЛЬКИ ЩО закінчилась (для логування в engine)
        if new_energy <= 0 and not self.logged_death:
             self.logged_death = True
             # Повідомлення про смерть буде в SimulationEngine

        # --- 7. Повернення назви дії ---
        return chosen_action_name

    # get_state_info успадковується від Entity і працює як треба