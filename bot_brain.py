# refactored_code/bot_brain.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Константи для архітектури мережі та дій ---
# INPUT_SIZE тепер визначається в entities.py
# Потрібно знати тільки розмір виходу та прихованого шару
HIDDEN_SIZE = 16  # Можна винести в config.py пізніше
OUTPUT_SIZE = 5   # Кількість можливих дій

# Можливі дії, які може вибрати бот
ACTIONS = ["move_north", "move_east", "move_south", "move_west", "stay"]
ACTION_MAP = {name: i for i, name in enumerate(ACTIONS)} # Для зручності

# --- Клас Нейромережі ---
class BotBrainNet(nn.Module):
    """
    Нейромережа, що представляє "мозок" бота.
    Приймає стан оточення та внутрішній стан (енергію) і видає оцінки для кожної можливої дії.
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        """
        Тепер приймає input_size як аргумент, бо він визначається в entities.py.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """ Пряме поширення. """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0) # Додаємо batch dimension

        hidden_output = F.relu(self.fc1(x))
        output = self.fc2(hidden_output)

        # Якщо на вході був один приклад, повертаємо теж одновимірний тензор
        return output.squeeze(0) if output.shape[0] == 1 else output