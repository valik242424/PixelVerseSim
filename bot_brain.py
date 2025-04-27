import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Може знадобитися для підготовки даних
from config import MAX_ENERGY

# Імпортуємо класи сутностей, щоб розпізнавати їх в оточенні
# Припускаємо, що entities.py знаходиться в тому ж каталозі
try:
    from entities import Wall, Bot, Entity # Додаємо Entity для загальних випадків
except ImportError:
    print("Warning: Could not import entity classes. Input preparation might fail.")
    # Визначимо фіктивні класи, якщо імпорт не вдався, щоб уникнути помилок далі
    class Entity: pass
    class Wall(Entity): pass
    class Bot(Entity): pass


# --- Константи для архітектури мережі та дій ---

# Розміри шарів
INPUT_SIZE = 13   # 4 напрямки * 3 типи (one-hot) + 1 енергія (нормалізована)
HIDDEN_SIZE = 16  # Кількість нейронів у прихованому шарі
OUTPUT_SIZE = 5   # Кількість можливих дій (N, E, S, W, Stay)

# Можливі дії, які може вибрати бот
ACTIONS = ["move_north", "move_east", "move_south", "move_west", "stay"]
ACTION_MAP = {name: i for i, name in enumerate(ACTIONS)} # Для зручності

# Індекси для one-hot encoding типів клітинок
CELL_TYPE_EMPTY = 0
CELL_TYPE_WALL = 1
CELL_TYPE_BOT = 2
NUM_CELL_TYPES = 3 # Кількість типів клітинок для one-hot encoding


# --- Клас Нейромережі ---

class BotBrainNet(nn.Module):
    """
    Нейромережа, що представляє "мозок" бота.
    Приймає стан оточення та внутрішній стан (енергію) і видає оцінки для кожної можливої дії.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__() # Викликаємо конструктор батьківського класу nn.Module
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Визначаємо шари мережі
        # Перший повністю зв'язаний шар: вхід -> прихований
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # Другий повністю зв'язаний шар: прихований -> вихід
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        # Функцію активації ReLU можна визначити тут або використовувати F.relu у forward
        # self.relu = nn.ReLU()

    def forward(self, x):
        """
        Визначає пряме поширення сигналу через мережу.

        Args:
            x (torch.Tensor): Вхідний тензор розміром (batch_size, input_size) або (input_size).

        Returns:
            torch.Tensor: Вихідний тензор з оцінками дій розміром (batch_size, output_size) або (output_size).
        """
        # Переконуємось, що вхід є тензором float
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        # Якщо вхід одновимірний (один приклад), додаємо розмірність батча
        if x.dim() == 1:
            x = x.unsqueeze(0) # Перетворює (input_size) на (1, input_size)

        # Прохід через перший шар і функцію активації ReLU
        hidden_output = F.relu(self.fc1(x))
        # Прохід через другий (вихідний) шар (без активації - Linear output)
        output = self.fc2(hidden_output)

        return output.squeeze(0) # Повертаємо одновимірний тензор, якщо на вході був один приклад


# --- Функція для підготовки вхідного вектора ---

def prepare_input_vector(surroundings, energy):
    """
    Готує вхідний вектор для нейромережі на основі оточення та енергії бота.

    Args:
        surroundings (dict): Словник, де ключі - напрямки ("north", "east", "south", "west"),
                             а значення - вміст відповідної клітинки (None, Wall, Bot, ...).
        energy (float): Поточний рівень енергії бота.

    Returns:
        torch.Tensor: Вхідний вектор (тензор) для нейромережі розміром INPUT_SIZE.
                      Або None, якщо виникла помилка.
    """
    input_features = []

    # 1. Кодування оточення (One-Hot Encoding)
    directions = ["north", "east", "south", "west"]
    for direction in directions:
        cell_content = surroundings.get(direction, None) # Отримуємо вміст клітинки
        # Створюємо one-hot вектор для цієї клітинки [is_empty, is_wall, is_bot]
        one_hot = [0.0] * NUM_CELL_TYPES # Починаємо з [0.0, 0.0, 0.0]

        if cell_content is None:
            one_hot[CELL_TYPE_EMPTY] = 1.0 # [1.0, 0.0, 0.0]
        elif isinstance(cell_content, Wall):
            one_hot[CELL_TYPE_WALL] = 1.0  # [0.0, 1.0, 0.0]
        elif isinstance(cell_content, Bot):
            one_hot[CELL_TYPE_BOT] = 1.0   # [0.0, 0.0, 1.0]
        # Додайте сюди elif для інших типів (Food тощо), якщо потрібно
        # else: # Невідомий об'єкт - кодуємо як порожнє або окремий тип? Поки як порожнє.
        #     one_hot[CELL_TYPE_EMPTY] = 1.0

        input_features.extend(one_hot) # Додаємо one-hot вектор до загального списку

    # Перевірка розміру після кодування оточення (має бути 4 * 3 = 12)
    if len(input_features) != NUM_CELL_TYPES * len(directions):
         print(f"Error: Incorrect number of features after encoding surroundings: {len(input_features)}")
         # Можливо, варто повернути None або кинути виняток
         # return None

    # 2. Додавання нормалізованої енергії
    normalized_energy = max(0.0, min(1.0, energy / MAX_ENERGY)) # Обмежуємо значення від 0 до 1
    input_features.append(normalized_energy)

    # Перевірка фінального розміру вектора
    if len(input_features) != INPUT_SIZE:
        print(f"Error: Final input vector size is incorrect: {len(input_features)}, expected {INPUT_SIZE}")
        return None # Повертаємо None у разі помилки

    # 3. Перетворення на тензор PyTorch
    return torch.tensor(input_features, dtype=torch.float32)


# --- Приклад використання (для тестування цього файлу) ---
if __name__ == "__main__":
    print("Testing Bot Brain Network...")

    # 1. Створюємо екземпляр мережі
    brain = BotBrainNet()
    print(f"Network architecture:\n{brain}")

    # 2. Готуємо фіктивні вхідні дані
    # Припустимо, на півночі стіна, на сході порожньо, на півдні бот, на заході порожньо
    dummy_surroundings = {
        "north": Wall(),
        "east": None,
        "south": Bot(bot_id="B2"), # Потрібно створити фіктивний екземпляр
        "west": None
    }
    dummy_energy = 85.0

    print(f"\nInput state: Surroundings={ {k: type(v).__name__ for k, v in dummy_surroundings.items()} }, Energy={dummy_energy}")

    # 3. Готуємо вхідний вектор
    input_vector = prepare_input_vector(dummy_surroundings, dummy_energy)

    if input_vector is not None:
        print(f"\nPrepared input tensor (size {input_vector.shape}):\n{input_vector}")

        # 4. Робимо передбачення (пряме поширення)
        # Мережа повинна бути в режимі evaluation для передбачень (особливо якщо є шари Dropout/BatchNorm)
        brain.eval()
        with torch.no_grad(): # Вимикаємо обчислення градієнтів для економії пам'яті/швидкості
            action_scores = brain(input_vector)

        print(f"\nOutput action scores (size {action_scores.shape}):\n{action_scores}")

        # 5. Визначаємо найкращу дію
        best_action_index = torch.argmax(action_scores).item() # .item() для отримання Python числа
        best_action_name = ACTIONS[best_action_index]

        print(f"\nBest action index: {best_action_index}, Best action name: '{best_action_name}'")
    else:
        print("\nFailed to prepare input vector.")