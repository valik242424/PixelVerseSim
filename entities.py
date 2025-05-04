import random
import torch
from PySide6.QtGui import QColor, Qt # Залежність від GUI ще є через QColor, подумаємо про це пізніше
from config import GRID_WIDTH, GRID_HEIGHT, MAX_ENERGY

# TODO: Подумати, де краще визначати ці константи (config?)
from bot_brain import BotBrainNet, ACTIONS, GRU_HIDDEN_SIZE # Додали GRU_HIDDEN_SIZE

# --- Константи ---
VIEW_RADIUS = 2 # <--- НОВЕ: Радіус огляду (5x5 = 2*2+1)
CELL_TYPE_EMPTY = 0
CELL_TYPE_WALL = 1
CELL_TYPE_BOT = 2
CELL_TYPE_FOOD = 3  # <--- НОВЕ: Окремий тип для їжі
NUM_CELL_TYPES = 4  # <--- НОВЕ: Кількість типів

DIRECTIONS = ["north", "east", "south", "west"] # Це більше не використовується для поля зору
NUM_DIRECTIONS = len(DIRECTIONS) # Не використовується для INPUT_SIZE

# РОЗРАХУНОК НОВОГО INPUT_SIZE
FIELD_SIZE = (2 * VIEW_RADIUS + 1) # 5
INPUT_SIZE = (FIELD_SIZE ** 2) * NUM_CELL_TYPES + 1 # (5*5)*4 + 1 = 101

# --- Функція prepare_input_vector (ОНОВЛЕНА для 5x5) ---
def prepare_input_vector(grid_data, bot_row, bot_col, energy):
    """
    Готує вхідний вектор для нейромережі (поле зору 5x5 + енергія).

    Args:
        grid_data (list[list]): Повна сітка симуляції.
        bot_row (int): Поточний рядок бота.
        bot_col (int): Поточний стовпець бота.
        energy (float): Поточна енергія бота.

    Returns:
        torch.Tensor або None при помилці.
    """
    input_features = []
    grid_height = len(grid_data)
    grid_width = len(grid_data[0]) if grid_height > 0 else 0

    # Ітеруємо по полю зору 5x5 (від -2 до +2 відносно бота)
    for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
        for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            nr, nc = bot_row + dr, bot_col + dc # Абсолютні координати клітинки

            one_hot = [0.0] * NUM_CELL_TYPES # [empty, wall, bot, food]

            # Перевірка меж світу
            if 0 <= nr < grid_height and 0 <= nc < grid_width:
                cell_content = grid_data[nr][nc]
                if cell_content is None:
                    one_hot[CELL_TYPE_EMPTY] = 1.0
                elif isinstance(cell_content, Wall):
                    one_hot[CELL_TYPE_WALL] = 1.0
                elif isinstance(cell_content, Bot):
                     # Важливо: не розрізняти *цього* бота від інших
                     # Можна додати окремий тип "self", але це ускладнить
                     # Поки що всі боти - це CELL_TYPE_BOT
                    one_hot[CELL_TYPE_BOT] = 1.0
                elif isinstance(cell_content, Food):
                    one_hot[CELL_TYPE_FOOD] = 1.0
                else: # Невідомий об'єкт вважаємо порожнім
                    one_hot[CELL_TYPE_EMPTY] = 1.0
            else:
                # Все за межами світу вважаємо стіною
                one_hot[CELL_TYPE_WALL] = 1.0

            input_features.extend(one_hot) # Додаємо one-hot вектор до загального списку

    # Перевірка розміру після поля зору (має бути 5*5 * 4 = 100)
    expected_view_features = (FIELD_SIZE ** 2) * NUM_CELL_TYPES
    if len(input_features) != expected_view_features:
         print(f"Error: Incorrect features after view encoding: {len(input_features)}, expected {expected_view_features}")
         return None

    # Додавання нормалізованої енергії
    normalized_energy = max(0.0, min(1.0, energy / MAX_ENERGY))
    input_features.append(normalized_energy)

    # Перевірка фінального розміру вектора (має бути 101)
    if len(input_features) != INPUT_SIZE:
        print(f"Error: Final input vector size incorrect: {len(input_features)}, expected {INPUT_SIZE}")
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

# --- Клас Bot (ОНОВЛЕНИЙ для GRU) ---
class Bot(Entity):
    def __init__(self, bot_id, color=(255, 0, 0), energy=100): # Використовуємо RGB кортеж для кольору
        # INPUT_SIZE (101) розраховано вище
        # GRU_HIDDEN_SIZE імпортовано з bot_brain
        # Передаємо INPUT_SIZE в конструктор Entity (не використовується, але для повноти)
        super().__init__(entity_type="bot", color=color, properties={"id": bot_id, "energy": energy})

        # Створюємо мозок з правильним INPUT_SIZE
        self.brain = BotBrainNet(input_size=INPUT_SIZE, gru_hidden_size=GRU_HIDDEN_SIZE)
        self.brain.eval() # Переводимо в режим оцінки

        # --- Ініціалізація прихованого стану ---
        # Зберігатимемо стан у форматі (num_layers, batch_size, hidden_size) = (1, 1, GRU_HIDDEN_SIZE)
        self.hidden_state = torch.zeros(1, 1, GRU_HIDDEN_SIZE)

        self.logged_death = False

    def update(self, grid_data, row, col):
        """
        Оновлює стан бота, використовуючи мозок з GRU.
        Повертає назву дії ('move_north', 'stay', etc.).
        Зменшує енергію за дію.
        """
        # --- 1. Отримання поточної енергії ---
        current_energy = self.properties.get('energy', 0.0)

        # --- 2. Перевірка стану (мертвий?) ---
        if current_energy <= 0:
            if not self.logged_death: self.logged_death = True
            # Важливо: скинути прихований стан, якщо бот помер?
            # Або залишити як є, якщо він може "воскреснути"? Поки залишаємо.
            return "stay"
        if self.logged_death: self.logged_death = False # Скидання прапорця, якщо ожив

        # --- 3. Підготовка вхідного вектора (нове поле зору) ---
        # Передаємо grid_data повністю, бо функція сама вибере потрібні клітинки
        input_vector = prepare_input_vector(grid_data, row, col, current_energy)
        if input_vector is None:
            print(f"Warning: Bot {self.properties.get('id')} failed to prepare input vector. Staying put.")
            return "stay"

        # --- 4. Отримання передбачення від мозку (з GRU) ---
        try:
            # Передаємо поточний вектор і ПОПЕРЕДНІЙ прихований стан
            current_h_state = self.hidden_state
            with torch.no_grad():
                # Мережа повертає (оцінки_дій, наступний_стан)
                action_scores, next_h_state = self.brain(input_vector, current_h_state)

            # Оновлюємо збережений прихований стан для НАСТУПНОГО кроку
            # detach() - щоб градієнти не текли між кроками симуляції під час навчання
            self.hidden_state = next_h_state.detach()

        except Exception as e:
            print(f"Error during brain forward pass for bot {self.properties.get('id')}: {e}")
            # У разі помилки, можливо, варто скинути прихований стан?
            self.hidden_state = torch.zeros(1, 1, GRU_HIDDEN_SIZE)
            return "stay"

        # --- 5. Вибір найкращої дії ---
        chosen_action_index = torch.argmax(action_scores).item()
        chosen_action_name = ACTIONS[chosen_action_index]

        # --- 6. Зменшення енергії ---
        energy_cost = 1.0
        new_energy = max(0, current_energy - energy_cost)
        self.properties['energy'] = new_energy
        if new_energy <= 0 and not self.logged_death:
             self.logged_death = True

        # --- 7. Повернення назви дії ---
        return chosen_action_name

    # get_state_info успадковується
    # get_color успадковується