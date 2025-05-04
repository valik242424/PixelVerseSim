import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Константи для архітектури ---
# INPUT_SIZE визначається в entities.py
# HIDDEN_SIZE тепер може бути розміром виходу препроцесингу або GRU
# Додамо розмір прихованого стану GRU
GRU_HIDDEN_SIZE = 32 # Можна зробити більшим (64, 128)
# Чи використовувати шар препроцесингу перед GRU?
USE_PREPROCESSING_LAYER = True
PREPROCESS_SIZE = 64 # Розмір виходу шару препроцесингу (якщо використовується)

OUTPUT_SIZE = 5 # Кількість дій (залишається)

ACTIONS = ["move_north", "move_east", "move_south", "move_west", "stay"]
ACTION_MAP = {name: i for i, name in enumerate(ACTIONS)}

# --- Клас Нейромережі з GRU ---
class BotBrainNet(nn.Module):
    """
    Нейромережа з GRU шаром для обробки послідовності спостережень (покроково).
    """
    def __init__(self, input_size, output_size=OUTPUT_SIZE,
                 gru_hidden_size=GRU_HIDDEN_SIZE,
                 use_preprocessing=USE_PREPROCESSING_LAYER,
                 preprocess_size=PREPROCESS_SIZE):
        super().__init__()
        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        self.use_preprocessing = use_preprocessing
        self.preprocess_size = preprocess_size

        # --- Шар препроцесингу (опціональний) ---
        if self.use_preprocessing:
            self.fc_preprocess = nn.Linear(self.input_size, self.preprocess_size)
            # Вхід для GRU - це вихід препроцесингу
            gru_input_feature_size = self.preprocess_size
        else:
            # Вхід для GRU - це безпосередньо вхід мережі
            gru_input_feature_size = self.input_size
            self.fc_preprocess = None # Щоб позначити відсутність

        # --- GRU Шар ---
        # batch_first=True означає, що тензори будуть (batch, seq, feature)
        # num_layers=1 - один шар GRU
        self.gru = nn.GRU(input_size=gru_input_feature_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=1,
                          batch_first=True)

        # --- Вихідний Шар ---
        # Приймає вихід GRU (розміром gru_hidden_size)
        self.fc_out = nn.Linear(self.gru_hidden_size, self.output_size)

    def forward(self, x, h_prev):
        """
        Виконує пряме поширення для одного кроку.

        Args:
            x (torch.Tensor): Вхідний вектор спостереження (розмір input_size).
            h_prev (torch.Tensor): Попередній прихований стан GRU
                                     (розмір (num_layers, batch_size, gru_hidden_size)).
                                     Для нас batch_size=1, num_layers=1.

        Returns:
            tuple: (action_scores, h_next)
                action_scores (torch.Tensor): Оцінки для кожної дії (розмір output_size).
                h_next (torch.Tensor): Новий прихований стан GRU
                                        (розмір (num_layers, batch_size, gru_hidden_size)).
        """
        # --- Перевірка та підготовка входу x ---
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        # Переконуємося, що x має розмірність (batch_size=1, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0) # (input_size) -> (1, input_size)
        # На цьому етапі x має форму (1, input_size)

        # --- Препроцесинг (якщо використовується) ---
        if self.use_preprocessing and self.fc_preprocess:
            processed_x = F.relu(self.fc_preprocess(x)) # (1, input_size) -> (1, preprocess_size)
        else:
            processed_x = x # (1, input_size) або (1, preprocess_size якщо він == input_size)

        # --- Підготовка до GRU ---
        # GRU очікує вхід у форматі (batch, seq, feature)
        # Оскільки ми обробляємо 1 крок, seq_len = 1
        gru_input = processed_x.unsqueeze(1) # (1, feature_size) -> (1, 1, feature_size)
                                             # Де feature_size = preprocess_size або input_size

        # --- Перевірка та підготовка прихованого стану h_prev ---
        # h_prev має бути (num_layers, batch_size, hidden_size) = (1, 1, gru_hidden_size)
        if h_prev is None: # Початковий стан (перший крок)
            # Створюємо нульовий тензор правильної форми
            h_prev = torch.zeros(1, 1, self.gru_hidden_size, device=x.device) # device=x.device - важливо!
        elif h_prev.shape != (1, 1, self.gru_hidden_size):
             # Якщо форма неправильна (напр., передали з неправильним batch_size або num_layers)
             print(f"Warning: Incorrect h_prev shape. Expected (1, 1, {self.gru_hidden_size}), got {h_prev.shape}. Reinitializing.")
             h_prev = torch.zeros(1, 1, self.gru_hidden_size, device=x.device)


        # --- Прохід через GRU ---
        # self.gru повертає:
        #   output: (batch, seq, num_directions * hidden_size) -> (1, 1, gru_hidden_size)
        #   h_n:    (num_layers * num_directions, batch, hidden_size) -> (1, 1, gru_hidden_size)
        gru_output, h_next = self.gru(gru_input, h_prev)

        # --- Обробка виходу GRU ---
        # Нам потрібен вихід останнього (єдиного) часового кроку
        # gru_output має форму (1, 1, gru_hidden_size)
        # Вибираємо вихід для batch=0, seq=0
        gru_output_last_step = gru_output.squeeze(1) # (1, 1, H) -> (1, H)

        # --- Прохід через вихідний шар ---
        action_scores = self.fc_out(gru_output_last_step) # (1, H) -> (1, output_size)

        # Повертаємо оцінки (без вимірності batch) і новий прихований стан (з усіма вимірностями)
        return action_scores.squeeze(0), h_next # (output_size), (1, 1, gru_hidden_size)