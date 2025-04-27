import random
import torch
from PySide6.QtGui import QColor, Qt
from config import GRID_WIDTH, GRID_HEIGHT
from bot_brain import BotBrainNet, prepare_input_vector, ACTIONS, MAX_ENERGY

# --- Базовий клас Entity ---
class Entity:
    def __init__(self, entity_type="generic", color=QColor(Qt.GlobalColor.gray), properties=None):
        self.entity_type = entity_type
        self.color = color
        self.properties = properties.copy() if properties is not None else {}

    def get_state_info(self):
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        if prop_str:
            return f"Type: {self.entity_type}, Properties: {prop_str}"
        else:
            return f"Type: {self.entity_type}"

    def update(self, grid_data, row, col):
        """
        Повертає рядок з назвою дії ('move_north', 'stay', etc.) або None.
        Базовий клас нічого не робить.
        """
        return None

# --- Клас Wall ---
class Wall(Entity):
    def __init__(self):
        super().__init__(entity_type="wall", color=QColor(Qt.GlobalColor.darkGray))

# --- Клас Bot ---
class Bot(Entity):
    def __init__(self, bot_id, color=QColor(Qt.GlobalColor.red), energy=100):
        super().__init__(entity_type="bot", color=color, properties={"id": bot_id, "energy": energy})
        self.brain = BotBrainNet()
        self.brain.eval()
        # --- Додаємо прапорець для відстеження логування смерті ---
        self.logged_death = False # Спочатку вважаємо, що про смерть не повідомляли

    def update(self, grid_data, row, col):
        # --- 1. Збір інформації про оточення (як раніше) ---
        surroundings = {}
        relative_coords = {
            "north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)
        }
        for direction, (dr, dc) in relative_coords.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                surroundings[direction] = grid_data[nr][nc]
            else:
                surroundings[direction] = Wall()

        # --- 2. Отримання ПОТОЧНОЇ енергії ---
        current_energy = self.properties.get('energy', 0.0)

        # --- Перевірка, чи є взагалі енергія для дій ---
        if current_energy <= 0:
            # Якщо енергії вже немає, перевіряємо, чи логували смерть
            if not self.logged_death:
                print(f"Bot {self.properties.get('id')} ran out of energy!")
                self.logged_death = True # Позначили, що залогували
            return "stay" # Немає енергії - стоїмо на місці

        # --- Якщо енергія є, продовжуємо ---

        # --- 3. Підготовка вхідного вектора (як раніше) ---
        input_vector = prepare_input_vector(surroundings, current_energy)
        if input_vector is None:
            print(f"Warning: Bot {self.properties.get('id')} failed to prepare input vector. Staying put.")
            return "stay"

        # --- 4. Отримання оцінок дій від мозку (як раніше) ---
        try:
            with torch.no_grad():
                action_scores = self.brain(input_vector)
        except Exception as e:
            print(f"Error during brain forward pass for bot {self.properties.get('id')}: {e}")
            return "stay"

        # --- 5. Вибір найкращої дії (як раніше) ---
        chosen_action_index = torch.argmax(action_scores).item()
        chosen_action_name = ACTIONS[chosen_action_index]

        # --- 6. Зменшення енергії ---
        # Зменшуємо енергію *після* прийняття рішення
        new_energy = max(0, current_energy - 1)
        self.properties['energy'] = new_energy

        # --- Перевірка, чи енергія ТІЛЬКИ ЩО закінчилась ---
        if new_energy <= 0 and not self.logged_death:
             # Якщо енергія стала <= 0 *на цьому кроці* і ми ще не логували
             print(f"Bot {self.properties.get('id')} ran out of energy!")
             self.logged_death = True # Позначили, що залогували

        # --- Скидання прапорця, якщо енергія відновилась (на майбутнє) ---
        # Цей блок спрацює, якщо бот якось отримає енергію (напр. з'їсть їжу)
        if new_energy > 0 and self.logged_death:
            # Якщо енергія знову позитивна, а ми раніше логували смерть
            self.logged_death = False # Скидаємо прапорець, дозволяємо логувати знову

        # --- 7. Повернення назви дії ---
        return chosen_action_name

    def get_state_info(self):
        """Повертає базову інформацію + поточну енергію."""
        base_info = super().get_state_info()
        # Перезаписуємо properties у рядку, щоб включити оновлену енергію
        prop_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return f"Type: {self.entity_type}, Properties: {prop_str}"


# --- Можна додати клас Food, якщо потрібно ---
class Food(Entity):
    def __init__(self, energy_value=25):
         super().__init__(entity_type="food", color=QColor(Qt.GlobalColor.yellow), properties={"energy": energy_value})