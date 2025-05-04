import random
import torch
from PySide6.QtGui import QColor, Qt # Залежність від GUI ще є через QColor
from config import GRID_WIDTH, GRID_HEIGHT, MAX_ENERGY

# Імпортуємо необхідне для RL
from dqn_agent import DQNAgent, device # Додали DQNAgent та device
from bot_brain import ACTIONS, GRU_HIDDEN_SIZE # ACTIONS потрібні для перетворення індексу в назву
from config import (GRID_WIDTH, GRID_HEIGHT, MAX_ENERGY,
                    VIEW_RADIUS, CELL_TYPE_EMPTY, CELL_TYPE_WALL,
                    CELL_TYPE_BOT, CELL_TYPE_FOOD, NUM_CELL_TYPES,
                    FIELD_SIZE, INPUT_SIZE,
                    ENERGY_COST_STEP)

# --- Функція prepare_input_vector  ---
def prepare_input_vector(grid_data, bot_row, bot_col, energy):
    input_features = []
    grid_height = len(grid_data)
    grid_width = len(grid_data[0]) if grid_height > 0 else 0

    # Використовуємо VIEW_RADIUS, NUM_CELL_TYPES з config
    for dr in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
        for dc in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            nr, nc = bot_row + dr, bot_col + dc
            one_hot = [0.0] * NUM_CELL_TYPES # Використовуємо NUM_CELL_TYPES
            if 0 <= nr < grid_height and 0 <= nc < grid_width:
                cell_content = grid_data[nr][nc]
                # Використовуємо CELL_TYPE_* з config
                if cell_content is None: one_hot[CELL_TYPE_EMPTY] = 1.0
                elif isinstance(cell_content, Wall): one_hot[CELL_TYPE_WALL] = 1.0
                elif isinstance(cell_content, Bot): one_hot[CELL_TYPE_BOT] = 1.0
                elif isinstance(cell_content, Food): one_hot[CELL_TYPE_FOOD] = 1.0
                else: one_hot[CELL_TYPE_EMPTY] = 1.0
            else:
                one_hot[CELL_TYPE_WALL] = 1.0
            input_features.extend(one_hot)

    # Використовуємо FIELD_SIZE, NUM_CELL_TYPES з config
    expected_view_features = (FIELD_SIZE ** 2) * NUM_CELL_TYPES
    if len(input_features) != expected_view_features:
         print(f"Error: Incorrect features after view encoding: {len(input_features)}, expected {expected_view_features}")
         return None

    normalized_energy = max(0.0, min(1.0, energy / MAX_ENERGY))
    input_features.append(normalized_energy)

    # Використовуємо INPUT_SIZE з config
    if len(input_features) != INPUT_SIZE:
        print(f"Error: Final input vector size incorrect: {len(input_features)}, expected {INPUT_SIZE}")
        return None

    return torch.tensor(input_features, dtype=torch.float32)


# --- Базовий клас Entity  ---
class Entity:
    def __init__(self, entity_type="generic", color=QColor(Qt.GlobalColor.gray), properties=None):
        self.entity_type = entity_type
        if isinstance(color, QColor):
            self.color_repr = (color.red(), color.green(), color.blue())
        else:
            self.color_repr = color if color else (128, 128, 128)
        self.properties = properties.copy() if properties is not None else {}

    def get_state_info(self):
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return f"Type: {self.entity_type}" + (f", Properties: {prop_str}" if prop_str else "")

    def get_color(self):
        return self.color_repr

    def update(self, grid_data, row, col):
        return None # За замовчуванням нічого не робить

# --- Клас Wall  ---
class Wall(Entity):
    def __init__(self):
        super().__init__(entity_type="wall", color=QColor(Qt.GlobalColor.darkGray))

# --- Клас Food  ---
class Food(Entity):
    def __init__(self, energy_value=25):
         super().__init__(entity_type="food", color=QColor(Qt.GlobalColor.yellow), properties={"energy": energy_value})

# --- Клас Bot (ОНОВЛЕНИЙ для DQNAgent) ---
class Bot(Entity):
    def __init__(self, bot_id, color=(255, 0, 0), energy=100):
        super().__init__(entity_type="bot", color=color, properties={"id": bot_id, "energy": energy})
        self.agent = DQNAgent(bot_id=bot_id)
        self.last_state = None
        self.last_action_index = None
        self.logged_death = False

    def update(self, grid_data, row, col):
        """
        Оновлює стан бота: готує спостереження, отримує дію від агента,
        витрачає енергію.
        Повертає назву дії ('move_north', 'stay', etc.).
        """
        current_energy = self.properties.get('energy', 0.0)

        if current_energy <= 0:
            if not self.logged_death: self.logged_death = True
            self.last_state = None
            self.last_action_index = None
            return "stay"
        if self.logged_death: self.logged_death = False

        # --- 1. Підготовка поточного стану (спостереження) ---
        state_tensor = prepare_input_vector(grid_data, row, col, current_energy)
        if state_tensor is None:
            print(f"Warning: Bot {self.properties.get('id')} failed to prepare input vector. Staying put.")
            self.last_state = None
            self.last_action_index = None
            return "stay"

        self.last_state = state_tensor

        # --- 2. Отримання дії від агента ---
        action_index_tensor = self.agent.select_action(state_tensor)
        action_index = action_index_tensor.item()
        self.last_action_index = action_index_tensor

        # --- 3. Витрата енергії за крок ---
        # Використовуємо константу з config.py
        energy_cost = ENERGY_COST_STEP # <--- ЗМІНЕНО
        new_energy = max(0, current_energy - energy_cost)
        self.properties['energy'] = new_energy
        if new_energy <= 0 and not self.logged_death:
             self.logged_death = True

        # --- 4. Повернення назви дії для рушія ---
        chosen_action_name = ACTIONS[action_index]
        return chosen_action_name

    def store_experience(self, next_state_tensor, reward, done):
        """
        Метод, який викликається рушієм ПІСЛЯ виконання дії,
        щоб зберегти повний перехід (s, a, r, s', done) в буфері агента.
        """
        # Перевіряємо, чи був валідний попередній стан і дія
        if self.last_state is not None and self.last_action_index is not None:
            # Передаємо дані в агент (він сам розбереться з CPU/GPU)
            self.agent.store_transition(
                self.last_state,
                self.last_action_index,
                next_state_tensor, # Наступний стан (тензор)
                reward,            # Винагорода (число)
                done               # Чи завершився епізод (bool)
            )
        else:
            # Якщо не було попереднього стану/дії (напр., перший крок після смерті/ресету),
            # то і зберігати нічого.
            pass

        # Скидаємо last_state і last_action_index після збереження,
        # щоб уникнути повторного збереження того ж переходу.
        # self.last_state = None
        # self.last_action_index = None
        # Або краще не скидати? Якщо бот зробить крок, а потім помре до наступного update,
        # то ми не збережемо його останній досвід. Краще залишити їх,
        # store_experience буде викликатися тільки якщо бот був живий на початку кроку.

    # --- Методи для керування агентом ззовні ---
    def set_training_mode(self, is_training: bool):
        """Перемикає режим навчання агента."""
        self.agent.set_training_mode(is_training)

    def save_model(self, path):
        """Зберігає модель агента."""
        self.agent.save_model(path)

    def load_model(self, path):
        """Завантажує модель агента."""
        self.agent.load_model(path)

    def reset_state(self):
        """Скидає внутрішній стан бота (наприклад, при ресеті симуляції)."""
        # Скидаємо прихований стан GRU в агенті
        self.agent.policy_hidden_state = torch.zeros(1, 1, self.agent.gru_hidden_size, device=device)
        # Скидаємо прапорець смерті
        self.logged_death = False
        # Скидаємо останній стан/дію
        self.last_state = None
        self.last_action_index = None
        print(f"[{self.properties.get('id')}] State reset.")